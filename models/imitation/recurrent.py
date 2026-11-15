"""Episode-local recurrent policy, padded chunks and truncated imitation updates."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .network import ResidualVisualEncoder, ObservationFeatureGate
from .dataset import BINARY_ACTION_COLUMNS

RECURRENT_ARCHITECTURE_VERSION = 'shionn_imitation_v5_gru_binned'


class RecurrentBinnedPolicy(nn.Module):
    """[B,T,12,180,320] -> action logits [B,T,C], state [1,B,H]."""
    def __init__(self, dx_classes, dy_classes, hidden_size=256, feature_gate=False):
        super().__init__()
        if hidden_size < 1 or min(dx_classes, dy_classes) < 3:
            raise ValueError('Invalid recurrent hidden size or mouse bins')
        self.hidden_size = hidden_size
        self.trunk = ResidualVisualEncoder()
        self.feature_gate = ObservationFeatureGate() if feature_gate else nn.Identity()
        self.memory = nn.GRU(512, hidden_size, batch_first=True)
        self.binary_heads = nn.ModuleDict({k: nn.Linear(hidden_size, 2) for k in BINARY_ACTION_COLUMNS})
        self.mouse_dx_head = nn.Linear(hidden_size, dx_classes)
        self.mouse_dy_head = nn.Linear(hidden_size, dy_classes)

    def forward(self, frames, hidden=None, valid=None, return_features=False):
        if frames.ndim != 5 or frames.shape[1] < 1:
            raise ValueError('Expected nonempty [B,T,C,H,W] sequence')
        b, t = frames.shape[:2]
        if hidden is None:
            hidden = frames.new_zeros(1, b, self.hidden_size)
        if hidden.shape != (1, b, self.hidden_size):
            raise ValueError('Hidden state shape mismatch')
        if valid is not None and valid.shape != (b, t):
            raise ValueError('Validity mask shape mismatch')
        outputs = []
        # Sequential encoding bounds temporary CNN work; autograd retains only
        # the explicitly supervised segment. Invalid padding cannot alter state.
        for i in range(t):
            features = self.feature_gate(self.trunk(frames[:, i]))
            _, candidate = self.memory(features[:, None], hidden.contiguous())
            hidden = candidate if valid is None else torch.where(valid[:, i][None, :, None], candidate, hidden)
            outputs.append(hidden[0])
        features = torch.stack(outputs, 1)
        result = {k: head(features) for k, head in self.binary_heads.items()}
        result['mouse_dx_logits'] = self.mouse_dx_head(features)
        result['mouse_dy_logits'] = self.mouse_dy_head(features)
        if return_features:
            return result, hidden, features
        return result, hidden

    def step(self, frame, hidden=None):
        output, hidden = self(frame[:, None], hidden)
        return {k: v[:, 0] for k, v in output.items()}, hidden


GOAL_CONDITIONED_ARCHITECTURE_VERSION = 'shionn_imitation_v5_gru_goal_binned'
GOAL_FEATURE_DIM = 11
GOAL_KIND_COUNT = 2
GOAL_KIND_EMBEDDING_DIM = 8


class GoalConditionedRecurrentPolicy(nn.Module):
    """Optional visible-waypoint conditioning over the existing recurrent policy.

    ``goals`` is [B,T,4]: normalized x/y, visibility, target-kind ID. The stable
    goal identity is managed by the caller and deliberately is not inferred
    from the camera view. Outputs add zero-initialized residual logits, so a
    warm-started candidate exactly matches its source recurrent policy.
    """
    def __init__(self, dx_classes, dy_classes, hidden_size=256, feature_gate=False,
                 adapter_size=128):
        super().__init__()
        if adapter_size < 1:
            raise ValueError('Goal adapter size must be positive')
        self.hidden_size = hidden_size
        self.base = RecurrentBinnedPolicy(
            dx_classes, dy_classes, hidden_size=hidden_size,
            feature_gate=feature_gate,
        )
        self.target_kind_embedding = nn.Embedding(
            GOAL_KIND_COUNT, GOAL_KIND_EMBEDDING_DIM
        )
        self.goal_adapter = nn.Sequential(
            nn.Linear(hidden_size + GOAL_FEATURE_DIM, adapter_size),
            nn.Tanh(),
        )
        from .dataset import BINARY_ACTION_COLUMNS
        self.binary_goal_heads = nn.ModuleDict({
            key: nn.Linear(adapter_size, 2) for key in BINARY_ACTION_COLUMNS
        })
        self.mouse_dx_goal_head = nn.Linear(adapter_size, dx_classes)
        self.mouse_dy_goal_head = nn.Linear(adapter_size, dy_classes)
        for head in (*self.binary_goal_heads.values(), self.mouse_dx_goal_head,
                     self.mouse_dy_goal_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode_goal(self, goals):
        if goals.ndim != 3 or goals.shape[-1] != 4:
            raise ValueError('Expected goals shaped [B,T,4]')
        xy = goals[..., :2]
        visible = goals[..., 2:3]
        kind = goals[..., 3].long()
        if torch.any(~torch.isfinite(goals)):
            raise ValueError('Goal values must be finite')
        if torch.any(goals[..., 3] != kind):
            raise ValueError('Target kind IDs must be integers')
        if torch.any((xy < -1) | (xy > 1)):
            raise ValueError('Visible goal coordinates must be normalized to [-1,1]')
        if torch.any((visible != 0) & (visible != 1)):
            raise ValueError('Goal visibility must be binary')
        if torch.any((visible == 0) & torch.any(xy != 0, dim=-1, keepdim=True)):
            raise ValueError('Invisible goal coordinates must be canonical zeros')
        if torch.any((kind < 0) | (kind >= GOAL_KIND_COUNT)):
            raise ValueError('Unknown target kind ID')
        return torch.cat((xy, visible, self.target_kind_embedding(kind)), dim=-1)

    def forward(self, frames, goals, hidden=None, valid=None):
        if goals.shape[:2] != frames.shape[:2]:
            raise ValueError('Goal sequence batch/time dimensions must match frames')
        base_output, hidden, state_features = self.base(
            frames, hidden=hidden, valid=valid, return_features=True,
        )
        goal_features = self.encode_goal(goals)
        residual = self.goal_adapter(torch.cat((state_features, goal_features), dim=-1))
        result = dict(base_output)
        for key, head in self.binary_goal_heads.items():
            result[key] = result[key] + head(residual)
        result['mouse_dx_logits'] = result['mouse_dx_logits'] + self.mouse_dx_goal_head(residual)
        result['mouse_dy_logits'] = result['mouse_dy_logits'] + self.mouse_dy_goal_head(residual)
        return result, hidden

    def step(self, frame, goal, hidden=None):
        if goal.ndim == 2:
            goal = goal[:, None]
        output, hidden = self(frame[:, None], goal, hidden=hidden)
        return {key: value[:, 0] for key, value in output.items()}, hidden


def initialize_goal_policy_from_recurrent(goal_policy, source_state_dict):
    """Warm-start and freeze the nested baseline; leave goal weights untouched."""
    goal_policy.base.load_state_dict(source_state_dict, strict=True)
    goal_policy.base.requires_grad_(False)


class SequenceDataset:
    """Nonoverlapping supervised chunks with optional preceding burn-in.

    Wrap an already split/trimmed BehaviorCloningDataset. Frame history always
    comes from that same episode. Left burn-in and right tail padding are masked.
    Hidden state starts at zero per chunk; burn-in is context, never a loss target.
    """
    def __init__(self, base, sequence_length=8, burn_in=4):
        if sequence_length < 1 or burn_in < 0:
            raise ValueError('Invalid sequence length or burn-in')
        self.base, self.sequence_length, self.burn_in = base, sequence_length, burn_in
        self.chunks = []
        offset = 0
        for ep, (frames, _) in enumerate(base.episodes):
            first = base.first_action_frames[ep]
            for start in range(first, len(frames), sequence_length):
                self.chunks.append((offset, first, start, len(frames)))
            offset += len(frames) - first

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        offset, first, start, end = self.chunks[index]
        total = self.burn_in + self.sequence_length
        frames = np.zeros((total, 12, 180, 320), dtype=np.uint8)
        targets = np.zeros((total, 10), dtype=np.float32)
        valid = np.zeros(total, dtype=bool)
        for i in range(total):
            frame = start - self.burn_in + i
            if first <= frame < end:
                frames[i], targets[i] = self.base[offset + frame - first]
                valid[i] = True
        return frames, targets, valid


def run_recurrent_epoch(model, loader, optimizer, scaler, device, train,
                        binary_class_weights, mouse_scale, global_step=0,
                        on_step=None, skip_batches=0, log_every=0, mouse_bins=None,
                        brightness=0., contrast=0., horizontal_flip_probability=0.,
                        label_smoothing=0., max_grad_norm=0.):
    from .train_bc import behavior_cloning_loss, augment_batch
    model.train(train)
    totals, count = {}, 0
    burn = loader.dataset.burn_in
    for batch_index, (frames, targets, valid) in enumerate(loader):
        if batch_index < skip_batches:
            continue
        frames = frames.to(device=device, dtype=torch.float32).div_(255.)
        targets, valid = targets.to(device), valid.to(device)
        if train:
            # One sampled augmentation per sequence, shared across its timesteps.
            for b in range(len(frames)):
                flat = frames[b].reshape(1, -1, *frames.shape[-2:])
                # Use the established label transform on all time labels with a
                # shared RNG draw by augmenting each sequence as one image stack.
                flip = horizontal_flip_probability and bool(torch.rand((), device=device) < horizontal_flip_probability)
                frames[b:b+1] = augment_batch(flat, targets[b:b+1, 0], brightness=brightness, contrast=contrast)[0].reshape_as(frames[b:b+1])
                if flip:
                    frames[b] = frames[b].flip(-1)
                    targets[b, :, [1, 3]] = targets[b, :, [3, 1]]
                    targets[b, :, 8] *= -1
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            hidden = None
            if burn:
                with torch.no_grad():
                    _, hidden = model(frames[:, :burn], valid=valid[:, :burn])
                hidden = hidden.detach()
            output, _ = model(frames[:, burn:], hidden, valid[:, burn:])
            mask = valid[:, burn:]
            n = int(mask.sum())
            if not n:
                raise ValueError('Sequence batch contains no supervised frames')
            selected = {k: v[mask] for k, v in output.items()}
            loss, parts = behavior_cloning_loss(selected, targets[:, burn:][mask],
                binary_class_weights, mouse_scale, mouse_bins, label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite recurrent loss')
        if train:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if on_step:
                on_step(global_step, batch_index + 1)
        for k,v in {'total': loss, **parts}.items():
            totals[k] = totals.get(k, 0.) + float(v.detach()) * n
        count += n
    return {k: v / max(1, count) for k,v in totals.items()}, global_step

"""Opt-in UNLABELLED observation capture across assisted subprocess handoff.

Never reads input devices or sends game controls. Its own process group contains
only this observer and its capture/encoder children, so blocked I/O can be killed
without touching the policy, human recorder or game. No training ingestion here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


class ObservationContext:
    """Parent-side bounded lifecycle; lazy-imported by the opt-in session flag."""
    def __init__(self, directory, *, width, height, fps, output=None, requested_map=None):
        self.directory = Path(directory).resolve()
        self.width, self.height, self.fps = width, height, fps
        self.output, self.requested_map = output, requested_map
        self.proc = None
        self.closed = False

    def state(self):
        path = self.directory / 'metadata.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def mark(self, phase, **links):
        with (self.directory / 'links.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'phase': phase, 'orchestration_monotonic': time.monotonic(),
                'boundary_precision': 'orchestration_only_use_actual_attempt_and_human_timestamps', **links}) + '\n')

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
            '--directory', str(self.directory), '--width', str(self.width),
            '--height', str(self.height), '--fps', str(self.fps), '--parent-pid', str(os.getpid())]
        if self.output:
            command.extend(['--output', self.output])
        if self.requested_map:
            command.extend(['--requested-map', self.requested_map])
        try:
            self.proc = subprocess.Popen(command, start_new_session=True)
            deadline = time.monotonic() + 15.
            while time.monotonic() < deadline:
                state = self.state()
                if self.proc.poll() is not None or state.get('capture_status') == 'failed':
                    raise RuntimeError(f'Observation context startup failed: {state.get("error", "worker exited")}')
                if state.get('capture_status') == 'recording':
                    self.ensure_healthy()
                    print(f'Unlabelled observation context ready: {self.directory}', flush=True)
                    return self
                time.sleep(.05)
            raise RuntimeError('Observation context received no ready frame within15seconds')
        except BaseException:
            self.close()
            raise

    def ensure_healthy(self):
        state = self.state()
        if self.proc is None or self.proc.poll() is not None or state.get('capture_status') != 'recording':
            raise RuntimeError(f'Observation context is incomplete: {state.get("error", "worker stopped")}')
        if time.monotonic() - state.get('heartbeat_monotonic', -math.inf) > 5.:
            raise RuntimeError('Observation context heartbeat stalled; correction is not counted')

    def close(self):
        if self.closed:
            return
        self.closed = True
        error = None
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=8.)
            except subprocess.TimeoutExpired:
                error = 'Observer did not finalize within8seconds; owned capture group killed'
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=3.)
            # Reap any orphan capture child after abnormal observer exit. Never
            # signal the parent's group: Popen above creates a separate session.
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                pass
            else:
                error = error or 'Observer left capture children; owned group killed'
                os.killpg(self.proc.pid, signal.SIGKILL)
            state = self.state()
            if self.proc.returncode != 0 or state.get('capture_status') != 'complete':
                error = error or state.get('error') or 'Observer exited without complete capture metadata'
        if error:
            state = self.state()
            state.update(capture_status='failed', error=error, expert_targets=False,
                         labels='none', requires_review=True)
            atomic_json(self.directory / 'metadata.json', state)
            raise RuntimeError(error)

    def __exit__(self, error_type, error, traceback):
        try:
            self.close()
        except Exception:
            if error_type is None:
                raise
        return False


def record_observations(directory, camera, writer, *, fps, stopped, requested_map=None):
    """Worker loop. CFR media is indexed by explicit receive/sample timestamps."""
    directory = Path(directory)
    state = {'kind': 'shionn_unlabelled_observation_context_v1', 'expert_targets': False,
        'labels': 'none', 'requires_review': True, 'training_approved': False,
        'capture_status': 'starting', 'requested_map': requested_map,
        'scope': 'May include prior scene, loading, preflight, autonomous play, handoff and human play',
        'clock': 'time.monotonic', 'frame_timestamp': 'complete_raw_frame_received_in_python',
        'not_display_exposure_time': True, 'video': 'context.mp4', 'frame_index': 'frames.jsonl',
        'fps': fps, 'frames': 0, 'max_sample_gap_seconds': 0., 'max_frame_age_seconds': 0.,
        'reused_frames': 0, 'started_monotonic': time.monotonic()}
    atomic_json(directory / 'metadata.json', state)
    error = None
    camera_started = False
    try:
        camera.start()
        camera_started = True
        previous_sample = previous_id = None
        next_tick = time.monotonic()
        ready_deadline = next_tick + 12.
        with (directory / 'frames.jsonl').open('w', encoding='utf-8', buffering=1) as frames:
            while not stopped():
                sample = camera.get_latest_frame_sample()
                sampled = time.monotonic()
                if sample.frame is None or sample.received_monotonic is None:
                    if sampled > ready_deadline:
                        raise RuntimeError('Context capture received no frames: ' + camera.error_summary())
                    time.sleep(.01)
                    continue
                age = sampled - sample.received_monotonic
                if not 0 <= age <= 2.:
                    raise RuntimeError('Context capture is stale; no continuity claim can be made')
                reused = sample.frame_id == previous_id if previous_id is not None else None
                writer.write(sample.frame)
                frames.write(json.dumps({'frame_idx': state['frames'], 'frame_id': sample.frame_id,
                    'received_monotonic': sample.received_monotonic, 'sampled_monotonic': sampled,
                    'frame_age_seconds': age, 'reused': reused,
                    'skipped_capture_frames': max(0, sample.frame_id - previous_id - 1) if previous_id is not None else None,
                    'expert_targets': False}) + '\n')
                state['frames'] += 1
                state['reused_frames'] += int(reused is True)
                state['max_frame_age_seconds'] = max(state['max_frame_age_seconds'], age)
                if previous_sample is not None:
                    state['max_sample_gap_seconds'] = max(state['max_sample_gap_seconds'], sampled - previous_sample)
                state.update(capture_status='recording', heartbeat_monotonic=time.monotonic())
                atomic_json(directory / 'metadata.json', state)
                previous_sample, previous_id = sampled, sample.frame_id
                next_tick = max(next_tick + 1. / fps, time.monotonic())
                time.sleep(max(0., next_tick - time.monotonic()))
    except BaseException as caught:
        error = caught
    finally:
        if camera_started:
            try:
                camera.stop()
            except BaseException as caught:
                error = error or caught
        try:
            writer.release()
        except BaseException as caught:
            error = error or caught
        if not state['frames'] and error is None:
            error = RuntimeError('Context recording ended without frames')
        state.update(capture_status='complete' if error is None else 'failed',
            stopped_monotonic=time.monotonic(), error=f'{type(error).__name__}: {error}' if error else None)
        atomic_json(directory / 'metadata.json', state)
    if error:
        raise error
    if not state['frames']:
        raise RuntimeError('Context recording ended without frames')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output')
    parser.add_argument('--requested-map')
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=float, default=24)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps) <= 0 or not math.isfinite(args.fps):
        parser.error('Positive finite capture dimensions/FPS required')
    from recorder import WaylandCamera, FFmpegVideoWriter
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    camera = WaylandCamera(args.width, args.height, args.fps, args.output)
    writer = FFmpegVideoWriter(str(args.directory / 'context.mp4'), args.width, args.height, args.fps)
    record_observations(args.directory, camera, writer, fps=args.fps,
                        stopped=lambda: stop or os.getppid() != args.parent_pid, requested_map=args.requested_map)


if __name__ == '__main__':
    main()

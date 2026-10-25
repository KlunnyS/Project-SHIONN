"""Temporary Hyprland hover-focus suppression for a single model attempt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

FOCUS_GUARD_EXIT_CODE = 3


class FocusGuardViolation(RuntimeError):
    """Stop an attempt instead of fighting the compositor for focus."""


class HyprlandFocusGuard:
    """Restore hover-focus preferences on normal stop, error, or interrupt.

    Optional pointer confinement uses a uniquely tagged window and named rule.
    Neither setting proves Gamescope has acquired relative mouse input.
    """

    def __init__(self, state_dir=Path("/tmp"), *, confine_pointer=False):
        self.state_dir = state_dir
        self.instance = None
        self.window = None
        self.original_follow_mouse = None
        self.changed = False
        self.lock = None
        self.restore_result = None
        self.owns_state = False
        self.confine_pointer = confine_pointer
        self.confinement_rule = "shionn_cursor_" + uuid.uuid4().hex if confine_pointer else None
        self.rule_touched = False
        self.tag_touched = False
        self.rule_match_probe = None

    def _accepted(self, *args):
        reply = self._run(*args)
        if reply != "ok":
            raise RuntimeError(f"Hyprland rejected guard command: {reply}")

    def _run(self, *args):
        # hyprctl parses flags even after the command. A separate '-tag'
        # argument is interpreted as an option, so dispatch its payload as one
        # argument beginning with the dispatcher name instead.
        if args and args[0] == "dispatch":
            args = ("dispatch", " ".join(args[1:]))
        result = subprocess.run(
            ["hyprctl", "-i", self.instance, *args], check=True,
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()

    def _read(self):
        response = json.loads(self._run("getoption", "input:follow_mouse", "-j"))
        value = response.get("int")
        if type(value) is not int or value not in (0, 1, 2, 3):
            raise RuntimeError("Cannot read Hyprland input:follow_mouse; focus guard was not verified")
        return value

    def _set(self, value):
        reply = self._run("keyword", "input:follow_mouse", str(value))
        # hyprctl can exit zero even when the compositor rejects a request.
        if reply != "ok":
            raise RuntimeError(f"Hyprland rejected focus setting: {reply}")
        if self._read() != value:
            raise RuntimeError("Hyprland focus setting did not take effect")

    def activate(self, window):
        if not window or window.get("portal_focused") is not True:
            raise RuntimeError("Focus guard requires an observed Portal/Gamescope window")
        self.instance = window.get("hyprland_instance")
        address = window.get("address")
        if not self.instance or not re.fullmatch(r"0x[0-9a-fA-F]+", str(address)):
            raise RuntimeError("Focus guard requires Hyprland window identity and instance")
        workspace = (window.get("workspace") or {}).get("id")
        if type(workspace) is not int:
            raise RuntimeError("Focus guard requires a known game workspace")
        self.window = dict(window)
        token = hashlib.sha256(self.instance.encode()).hexdigest()[:16]
        path = self.state_dir / f"shionn-focus-guard-{os.getuid()}-{token}.json"
        self.lock = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            self.lock = None
            raise RuntimeError("Another SHIONN focus guard owns this Hyprland session") from error
        self.original_follow_mouse = self._read()
        # A killed process may leave a runtime setting behind. Preserve its
        # recovery note rather than silently treating the leftover zero as normal.
        self.lock.seek(0)
        previous = self.lock.read()
        if previous:
            try:
                recovery = json.loads(previous)
            except ValueError:
                recovery = {}
            if recovery.get("active") and (
                recovery.get("original_follow_mouse") != self.original_follow_mouse
                or recovery.get("confinement_rule")
            ):
                raise RuntimeError(f"Unrestored focus guard found; inspect {path} before starting another")
        self.lock.seek(0)
        self.owns_state = True
        self.lock.truncate()
        json.dump({"active": True, **self.describe()}, self.lock)
        self.lock.flush()
        if self.original_follow_mouse != 0:
            self.changed = True  # Also roll back a timed-out/ambiguous write.
            self._set(0)
        if self.confine_pointer:
            self._activate_confinement()
        print(f"Focus guard: hover focus disabled for this attempt; recovery state: {path}")

    def _rule_key(self, field):
        return f"windowrule[{self.confinement_rule}]:{field}"

    def _activate_confinement(self):
        # Configure a disabled named rule before attaching a unique static tag.
        # This targets the selected outer Gamescope address, never all games.
        self.rule_touched = True
        self._accepted("keyword", self._rule_key("enable"), "0")
        # match:tag is an exact tag name, unlike match:class/title regexes.
        self._accepted("keyword", self._rule_key("match:tag"), self.confinement_rule)
        self._accepted("keyword", self._rule_key("confine_pointer"), "on")
        # This Hyprland version cannot getprop(confine_pointer). Use a readable
        # effect on the SAME rule to prove it matched, not just that a tag exists.
        # Hover focus is already globally disabled throughout this guarded run.
        original = self._read_window_hover_property()
        expected = not original
        self.rule_match_probe = {"property": "no_follow_mouse", "original": original,
                                 "expected": expected, "observed": None}
        self._accepted("keyword", self._rule_key("no_follow_mouse"), "on" if expected else "off")
        self.tag_touched = True
        self._accepted("dispatch", "tagwindow", f"+{self.confinement_rule} address:{self.window['address']}")
        clients = json.loads(self._run("clients", "-j"))
        target = next((w for w in clients if w.get("address") == self.window["address"]), None)
        if not target or self.confinement_rule not in target.get("tags", []):
            raise RuntimeError("Could not verify the confinement tag on the selected game window")
        if any(self.confinement_rule in w.get("tags", []) and w.get("address") != self.window["address"] for w in clients):
            raise RuntimeError("Confinement tag appeared on an unexpected window")
        self._accepted("keyword", self._rule_key("enable"), "1")
        deadline = time.monotonic() + 0.5
        while True:
            observed = self._read_window_hover_property()
            self.rule_match_probe["observed"] = observed
            if observed == expected:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Confinement rule did not match the selected window; refusing to send policy input")
            time.sleep(0.05)
        print("Confinement rule match verified on the selected game window; preflight must verify camera motion.")

    def _read_window_hover_property(self):
        value = json.loads(self._run("getprop", f"address:{self.window['address']}", "no_follow_mouse", "-j"))
        prop = value.get("no_follow_mouse")
        if type(prop) is not bool:
            raise RuntimeError("Cannot verify confinement rule matching with Hyprland getprop")
        return prop

    def _remove_tag(self):
        clients = json.loads(self._run("clients", "-j"))
        for window in clients:
            if self.confinement_rule not in window.get("tags", []):
                continue
            address = window.get("address")
            if not re.fullmatch(r"0x[0-9a-fA-F]+", str(address)):
                raise RuntimeError("Cannot safely address a window carrying the confinement tag")
            self._accepted("dispatch", "tagwindow", f"-{self.confinement_rule} address:{address}")
        remaining = json.loads(self._run("clients", "-j"))
        if any(self.confinement_rule in w.get("tags", []) for w in remaining):
            raise RuntimeError("Confinement tag remained after cleanup")

    def describe(self):
        return {"hyprland_instance": self.instance,
                "window_address": (self.window or {}).get("address"),
                "workspace": (self.window or {}).get("workspace"),
                "original_follow_mouse": self.original_follow_mouse,
                "confinement_rule": self.confinement_rule,
                "confinement_requested": self.confine_pointer,
                "rule_match_probe": self.rule_match_probe,
                "confinement_cleanup_commands": [
                    ["hyprctl", "-i", self.instance, "keyword", self._rule_key("enable"), "0"],
                    ["hyprctl", "-i", self.instance, "dispatch",
                     f"tagwindow -{self.confinement_rule} address:{(self.window or {}).get('address')}"],
                ] if self.confine_pointer else [],
                "restore_command": ["hyprctl", "-i", self.instance,
                                    "keyword", "input:follow_mouse", str(self.original_follow_mouse)]}

    def check(self, snapshot, focus_losses=0, cursor_exits=0):
        if focus_losses:
            raise FocusGuardViolation("Focus loss observed during guarded attempt; refusing automatic refocus")
        if snapshot.get("portal_focused") is not True:
            raise FocusGuardViolation("Game focus is lost, unknown, or stale during guarded attempt")
        window = snapshot.get("window") or {}
        if window.get("address") != self.window.get("address"):
            raise FocusGuardViolation("Focused game window identity changed")
        if (window.get("workspace") or {}).get("id") != self.window["workspace"]["id"]:
            raise FocusGuardViolation("Game workspace changed during guarded attempt")
        if window.get("monitor") != self.window.get("monitor"):
            raise FocusGuardViolation("Game monitor changed during guarded attempt")
        if self.confine_pointer and (cursor_exits or snapshot.get("cursor_inside_game_window") is not True):
            raise FocusGuardViolation("Pointer containment was lost or unknown despite the confinement rule")

    def restore(self):
        if not self.changed and self.lock is None and not self.rule_touched and not self.tag_touched:
            return self.restore_result or {"status": "not_activated"}
        errors = []
        for field, command in (
            ("rule_touched", ("keyword", self._rule_key("enable"), "0")),
            ("tag_touched", ("dispatch", "tagwindow", f"-{self.confinement_rule} address:{(self.window or {}).get('address')}")),
        ):
            if getattr(self, field):
                try:
                    if field == "tag_touched":
                        self._remove_tag()
                    else:
                        self._accepted(*command)
                    setattr(self, field, False)
                except Exception as error:
                    errors.append(str(error))
        try:
            if self.changed:
                current = self._read()
                if current == 0:
                    self._set(self.original_follow_mouse)
                elif current != self.original_follow_mouse:
                    # Respect an explicit preference change made during the run.
                    self.restore_result = {"status": "external_change_preserved", "value": current}
            if errors:
                raise RuntimeError("; ".join(errors))
            if self.restore_result is None or self.restore_result.get("status") == "failed":
                self.restore_result = {"status": "restored"}
            self.changed = False
        except Exception as error:
            errors.append(str(error))
            self.restore_result = {"status": "failed", "error": str(error), **self.describe()}
            commands = self.describe()["confinement_cleanup_commands"] + [self.describe()["restore_command"]]
            print(f"Focus guard restoration failed: {error}. Cleanup commands: {commands}")
        finally:
            if self.lock is not None:
                # Keep a recovery note if restoration failed.
                if self.owns_state and not errors:
                    self.lock.seek(0)
                    self.lock.truncate()
                    self.lock.flush()
                self.lock.close()
                self.lock = None
        return self.restore_result

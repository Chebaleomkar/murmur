"""Global hotkeys and the paste keystroke, per OS.

Windows keeps the `keyboard` library, which is what the Windows build was
tested with. macOS and Linux use pynput: `keyboard` doesn't support macOS,
and on Linux it needs root.
"""
import sys

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"
MODIFIERS = ("ctrl", "alt", "shift", "windows")


class ComboTracker:
    """Turns key down/up events into dictate press/release and repaste calls.
    Works on key names only, so it can be tested without a keyboard hook."""

    def __init__(self, dictate_combos, release_keys, repaste_combo,
                 on_press, on_release, repaste):
        self._dictate = [frozenset(c.split("+")) for c in dictate_combos]
        self._release_keys = set(release_keys)
        self._repaste = frozenset(repaste_combo.split("+"))
        self._on_press, self._on_release, self._repaste_fn = on_press, on_release, repaste
        self._down = set()
        self._repaste_latched = False  # fire once per press, not per auto-repeat

    def key_down(self, name):
        self._down.add(name)
        if any(combo <= self._down for combo in self._dictate):
            self._on_press()  # the controller ignores repeats while held
        if self._repaste <= self._down and not self._repaste_latched:
            self._repaste_latched = True
            self._repaste_fn()

    def key_up(self, name):
        self._down.discard(name)
        if not self._repaste <= self._down:
            self._repaste_latched = False
        if name in self._release_keys:
            self._on_release()


def start(dictate_combos, release_keys, repaste_combo, on_press, on_release, repaste):
    if WINDOWS:
        import keyboard

        for combo in dictate_combos:
            keyboard.add_hotkey(combo, on_press)
        for key in release_keys:
            keyboard.on_release_key(key, lambda e: on_release())
        keyboard.add_hotkey(repaste_combo, repaste)
        return None

    from pynput import keyboard as pk

    tracker = ComboTracker(dictate_combos, release_keys, repaste_combo,
                           on_press, on_release, repaste)

    def name_of(key):
        if isinstance(key, pk.KeyCode):
            return key.char.lower() if key.char else None
        base = key.name.split("_")[0]  # ctrl_l -> ctrl, cmd_r -> cmd
        return "windows" if base == "cmd" else base

    def down(key):
        name = name_of(key)
        if name:
            tracker.key_down(name)

    def up(key):
        name = name_of(key)
        if name:
            tracker.key_up(name)

    listener = pk.Listener(on_press=down, on_release=up, daemon=True)
    listener.start()
    return listener


def send_paste():
    if WINDOWS:
        import keyboard

        # Win+Ctrl+V opens Sound output, so let go of the modifiers first.
        for name in MODIFIERS:
            keyboard.release(name)
        keyboard.send("ctrl+v")
        return

    from pynput import keyboard as pk

    controller = pk.Controller()
    for mod in (pk.Key.ctrl, pk.Key.alt, pk.Key.shift, pk.Key.cmd):
        controller.release(mod)
    with controller.pressed(pk.Key.cmd if MACOS else pk.Key.ctrl):
        controller.press("v")
        controller.release("v")

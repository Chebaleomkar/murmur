"""
Anti-aliased status pill for Murmur, drawn with Qt (PySide6).

The window is click-through and never takes focus: Murmur pastes into
whichever window is focused, so stealing focus would break dictation.

Qt must run on the main thread. Call Overlay.exec() from main(); the
show_*/hide methods are safe to call from any other thread.
"""

import math
import re
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from PySide6.QtCore import (
    QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen,
)
from PySide6.QtWidgets import QApplication, QWidget

PILL_W, PILL_H = 168, 44
PILL_MAX_W = 480
PAD_X = 20
FADE_MS = 120
FRAME_MS = 16  # ~60fps, only while visible

BAR_COUNT = 9
BAR_W, BAR_GAP = 4, 5
BAR_MIN_H = 4
BAR_ATTACK, BAR_DECAY = 0.55, 0.08  # per-frame easing: rise fast, fall slowly
LEVEL_FULL_SCALE = 0.06  # rms mapped to a full-height bar (speech is ~0.01-0.1)

BG = QColor(24, 24, 27, 235)
BORDER = QColor(255, 255, 255, 36)
TEXT = QColor(244, 244, 245)
TEAL = QColor("#4fd1c5")
DIM = QColor("#5a5a62")
RED = QColor("#ff6b6b")


class Mode(Enum):
    HIDDEN = "hidden"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    ERROR = "error"


@dataclass(frozen=True)
class OverlayState:
    mode: Mode = Mode.HIDDEN
    message: str = ""


@dataclass
class Frame:
    """Everything that changes per animation tick, so painting stays pure."""
    bars: list[float] = field(default_factory=lambda: [0.0] * BAR_COUNT)  # 0..1
    active: list[bool] = field(default_factory=lambda: [False] * BAR_COUNT)
    elapsed: float = 0.0  # seconds since listening began
    t: float = 0.0  # monotonic clock, drives the pulsing dots


def clean_message(message: str) -> str:
    """The red dot replaces any leading warning glyph or emoji."""
    return re.sub(r"^\W+", "", message)


def _text_width(fm: QFontMetrics, text: str) -> int:
    return fm.horizontalAdvance(text)


def _error_width(state: OverlayState, fm: QFontMetrics) -> int:
    # padding + dot + gap + text + padding
    fitted = PAD_X * 2 + 8 + 10 + _text_width(fm, clean_message(state.message))
    return min(max(PILL_W, fitted), PILL_MAX_W)


WIDTH_FOR: dict[Mode, Callable[[OverlayState, QFontMetrics], int]] = {
    Mode.LISTENING: lambda s, fm: PILL_W,
    Mode.TRANSCRIBING: lambda s, fm: PILL_W,
    Mode.ERROR: _error_width,
}


def _paint_listening(p: QPainter, state: OverlayState, rect: QRectF, frame: Frame):
    fm = p.fontMetrics()
    cy = rect.center().y()

    timer = f"{int(frame.elapsed) // 60}:{int(frame.elapsed) % 60:02d}"
    timer_w = _text_width(fm, "00:00")
    p.setPen(TEXT)
    p.drawText(
        QRectF(rect.right() - PAD_X - timer_w, rect.top(), timer_w, rect.height()),
        int(Qt.AlignVCenter | Qt.AlignRight), timer,
    )

    max_h = rect.height() - 16
    pen = QPen()
    pen.setWidthF(BAR_W)
    pen.setCapStyle(Qt.RoundCap)
    x = rect.left() + PAD_X + BAR_W / 2
    for level, active in zip(frame.bars, frame.active):
        pen.setColor(TEAL if active else DIM)
        p.setPen(pen)
        half = (BAR_MIN_H + level * (max_h - BAR_MIN_H) - BAR_W) / 2  # caps add BAR_W/2 each end
        p.drawLine(QPointF(x, cy - half), QPointF(x, cy + half))
        x += BAR_W + BAR_GAP


def _paint_transcribing(p: QPainter, state: OverlayState, rect: QRectF, frame: Frame):
    fm = p.fontMetrics()
    label = "Transcribing"
    dot_r, dot_gap, label_gap = 2.5, 4.5, 8
    dots_w = 3 * dot_r * 2 + 2 * dot_gap
    total = _text_width(fm, label) + label_gap + dots_w
    left = rect.center().x() - total / 2

    p.setPen(TEXT)
    p.drawText(
        QRectF(left, rect.top(), _text_width(fm, label) + 2, rect.height()),
        int(Qt.AlignVCenter | Qt.AlignLeft), label,
    )

    p.setPen(Qt.NoPen)
    x = left + _text_width(fm, label) + label_gap + dot_r
    for i in range(3):
        # each dot peaks a third of a cycle after the previous one
        pulse = 0.5 + 0.5 * math.cos(frame.t * 2 * math.pi * 1.2 - i * 2 * math.pi / 3)
        color = QColor(TEXT)
        color.setAlphaF(0.3 + 0.7 * pulse)
        p.setBrush(color)
        r = dot_r * (0.8 + 0.3 * pulse)
        p.drawEllipse(QPointF(x, rect.center().y()), r, r)
        x += dot_r * 2 + dot_gap


def _paint_error(p: QPainter, state: OverlayState, rect: QRectF, frame: Frame):
    fm = p.fontMetrics()
    dot_d, gap = 8, 10
    text_x = rect.left() + PAD_X + dot_d + gap

    p.setPen(Qt.NoPen)
    p.setBrush(RED)
    p.drawEllipse(QPointF(rect.left() + PAD_X + dot_d / 2, rect.center().y()),
                  dot_d / 2, dot_d / 2)

    text_w = rect.right() - PAD_X - text_x
    text = fm.elidedText(clean_message(state.message), Qt.ElideRight, int(text_w))
    p.setPen(TEXT)
    p.drawText(QRectF(text_x, rect.top(), text_w, rect.height()),
               int(Qt.AlignVCenter | Qt.AlignLeft), text)


PAINTERS: dict[Mode, Callable[[QPainter, OverlayState, QRectF, Frame], None]] = {
    Mode.LISTENING: _paint_listening,
    Mode.TRANSCRIBING: _paint_transcribing,
    Mode.ERROR: _paint_error,
}


def paint(p: QPainter, state: OverlayState, size: tuple[int, int], frame: Frame):
    """Draw one frame of `state` into a width x height pill. No side effects
    beyond the painter."""
    painter_fn = PAINTERS.get(state.mode)
    if painter_fn is None:
        return
    w, h = size
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
    # inset half a pixel so the 1px border is crisp rather than split across pixels
    rect = QRectF(0.5, 0.5, w - 1, h - 1)
    p.setPen(QPen(BORDER, 1))
    p.setBrush(BG)
    p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
    painter_fn(p, state, QRectF(0, 0, w, h), frame)


def _make_font() -> QFont:
    font = QFont()
    font.setFamilies(["Segoe UI Variable", "Segoe UI"])
    font.setPointSize(10)
    return font


class _Pill(QWidget):
    request = Signal(object, str)  # (Mode, message); emitted from any thread

    def __init__(self, level_source: Callable[[], list], quiet_threshold: float):
        super().__init__()
        self._level_source = level_source
        self._quiet_threshold = quiet_threshold
        self._state = OverlayState()
        self._painted = OverlayState()  # last visible state, kept for the fade-out
        self._frame = Frame()
        self._listening_since = 0.0
        self._font = _make_font()

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFont(self._font)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.InOutQuad)
        self._fade.finished.connect(self._on_fade_finished)

        self._ticker = QTimer(self)
        self._ticker.setInterval(FRAME_MS)
        self._ticker.timeout.connect(self._tick)

        # Queued so calls from the hotkey/worker threads run on the Qt thread.
        self.request.connect(self._apply, Qt.QueuedConnection)

    def _apply(self, mode: Mode, message: str):
        previous = self._state.mode
        self._state = OverlayState(mode, message)
        if mode is Mode.HIDDEN:
            if self.isVisible():
                self._fade_to(0.0)
            return

        self._painted = self._state
        if mode is Mode.LISTENING and previous is not Mode.LISTENING:
            self._listening_since = time.monotonic()
            self._frame = Frame()
        self._place()
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
        self._ticker.start()
        self._tick()
        self._fade_to(1.0)

    def _fade_to(self, target: float):
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(target)
        self._fade.start()

    def _on_fade_finished(self):
        if self._state.mode is Mode.HIDDEN:
            self._ticker.stop()
            self.hide()

    def _place(self):
        """Re-measure and re-centre on every show: the primary monitor or its
        DPI may have changed since last time."""
        fm = QFontMetrics(self._font)
        width = WIDTH_FOR[self._painted.mode](self._painted, fm)
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.setGeometry(
            area.x() + (area.width() - width) // 2,
            area.y() + area.height() - PILL_H - 24,
            width, PILL_H,
        )

    def _tick(self):
        frame = self._frame
        frame.t = time.monotonic()
        if self._painted.mode is Mode.LISTENING:
            frame.elapsed = frame.t - self._listening_since
            self._update_bars()
        self.update()

    def _update_bars(self):
        try:
            recent = list(self._level_source())[-BAR_COUNT:]
        except RuntimeError:  # deque mutated by the audio thread mid-copy
            return
        recent = [0.0] * (BAR_COUNT - len(recent)) + recent
        frame = self._frame
        for i, rms in enumerate(recent):
            target = min(1.0, math.sqrt(rms / LEVEL_FULL_SCALE))
            rate = BAR_ATTACK if target > frame.bars[i] else BAR_DECAY
            frame.bars[i] += (target - frame.bars[i]) * rate
            frame.active[i] = rms > self._quiet_threshold

    def paintEvent(self, event):
        p = QPainter(self)
        p.setFont(self._font)
        paint(p, self._painted, (self.width(), self.height()), self._frame)
        p.end()


class Overlay:
    def __init__(self, level_source: Callable[[], list], quiet_threshold: float):
        self._app = QApplication.instance() or QApplication([])
        self._app.setQuitOnLastWindowClosed(False)
        self._pill = _Pill(level_source, quiet_threshold)

    def show_listening(self):
        self._pill.request.emit(Mode.LISTENING, "")

    def show_transcribing(self):
        self._pill.request.emit(Mode.TRANSCRIBING, "")

    def show_error(self, message: str):
        self._pill.request.emit(Mode.ERROR, message)

    def hide(self):
        self._pill.request.emit(Mode.HIDDEN, "")

    def exec(self):
        """Run the Qt event loop on the calling (main) thread. Ctrl+C quits it."""
        signal.signal(signal.SIGINT, lambda *_: self._app.quit())
        # Python only runs signal handlers when the interpreter gets control,
        # and a pure Qt loop never gives it any; wake it periodically.
        wake = QTimer()
        wake.start(200)
        wake.timeout.connect(lambda: None)
        return self._app.exec()

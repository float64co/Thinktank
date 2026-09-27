#!/usr/bin/env python3
"""
thinktank/tui.py — IRC-style multi-agent Claude chat TUI.

See thinktank.txt for the full design spec.
"""

from __future__ import annotations

import _curses
import argparse
import asyncio
import datetime
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic

try:
    from .window import (
        ALIGN_LEFT, EXPAND,
        Pane, Window,
        display_width, palette, truncate_to_display_width,
    )
except ImportError:
    from window import (
        ALIGN_LEFT, EXPAND,
        Pane, Window,
        display_width, palette, truncate_to_display_width,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Themes
# ─────────────────────────────────────────────────────────────────────────────

THEMES: Dict[str, dict] = {
    "dark": {
        "header_fg":         "white",
        "header_bg":         "blue",
        "status_fg":         "yellow",
        "input_fg":          "white",
        "timestamp_fg":      "cyan",
        "system_fg":         "yellow",
        "user_fg":           "green",
        "participant_colors": ["cyan", "magenta", "yellow", "red", "green", "white"],
    },
    "light": {
        "header_fg":         "black",
        "header_bg":         "white",
        "status_fg":         "black",
        "input_fg":          "black",
        "timestamp_fg":      "blue",
        "system_fg":         "red",
        "user_fg":           "black",
        "participant_colors": ["blue", "red", "magenta", "cyan", "green", "black"],
    },
    "matrix": {
        "header_fg":         "black",
        "header_bg":         "green",
        "status_fg":         "green",
        "input_fg":          "green",
        "timestamp_fg":      "green",
        "system_fg":         "white",
        "user_fg":           "green",
        "participant_colors": ["green", "white", "cyan", "yellow"],
    },
}

DEFAULT_MODEL    = "claude-opus-4-6"
DEFAULT_MAX_TOKENS = 4096
IRC_MAX_CHARS    = 490   # RFC 1459 / RFC 2812 PRIVMSG payload headroom
IRC_MAX_TOKENS   = 150   # generous token budget for one IRC line
USER_SENDER      = "you"

OLLAMA_PREFIX    = "ollama:"   # model ids with this prefix are served by local Ollama

# Shown when the Anthropic models endpoint can't be reached.
FALLBACK_ANTHROPIC_MODELS = [
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
]

PARTICIPANT_PREAMBLE = (
    "You are on IRC. You pitch in on whatever comes up — debugging, life advice, "
    "recipes, philosophy, plumbing, grief, whatever. Specialization is for insects. "
    "Be genuinely helpful first; let your persona flavor how you help, not whether you do."
)

CONSTRAIN_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: You are speaking live on IRC. "
    "Reply with exactly ONE sentence — no line breaks, no lists, no preamble. "
    "Your entire message must fit within 490 characters."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    sender:         str
    text:           str
    timestamp:      str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S")
    )
    full_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def log_line(self) -> str:
        return f"[{self.full_timestamp}] <{self.sender}> {self.text}"


@dataclass
class Participant:
    name:            str
    system_prompt:   str
    model:           str  = DEFAULT_MODEL
    muted:           bool = False
    paused:          bool = False
    paused_at_index: int  = 0    # len(channel_history) when paused
    prefill_buffer:  str  = ""   # partial text saved on cancel for prefill

    @property
    def is_opus(self) -> bool:
        return "opus-4-6" in self.model

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "system_prompt": self.system_prompt}
        if self.model != DEFAULT_MODEL:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Participant":
        return cls(
            name=d["name"],
            system_prompt=d["system_prompt"],
            model=d.get("model", DEFAULT_MODEL),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Panes
# ─────────────────────────────────────────────────────────────────────────────

class HeaderBar(Pane):
    """Single-row title bar showing channel name and status."""
    geometry = [EXPAND, 1]

    def __init__(self, name: str, channel: str = "#thinktank", theme: dict = None):
        super().__init__(name)
        self.channel = channel
        self.theme   = theme or THEMES["dark"]
        self.right   = ""   # right-aligned annotation (participant count, etc.)

    def update(self):
        attr  = palette(self.theme["header_fg"], self.theme["header_bg"])
        left  = f" {self.channel} "
        right = self.right
        pad   = max(0, (self.width or 0) - display_width(left) - display_width(right))
        text  = left + " " * pad + right
        self.content = [[text, ALIGN_LEFT, attr]]


class ScrollBuffer(Pane):
    """Scrolling IRC-style message buffer.

    Displays messages as:  <HH:MM:SS> <name> text
    Word-wraps long lines.  PgUp / PgDn to scroll; auto-scrolls on new messages.
    """
    geometry = [EXPAND, EXPAND]

    def __init__(self, name: str, theme: dict = None):
        super().__init__(name)
        self.theme    = theme or THEMES["dark"]
        self.messages: List[Message]    = []
        self.scroll_offset: int         = 0   # lines from bottom; 0 = pinned
        self.auto_scroll: bool          = True
        self._colors:    Dict[str, int] = {}  # sender → curses attr
        self._streaming: Dict[str, str] = {}  # name → partial text

    # ── Public API ───────────────────────────────────────────

    def add_message(self, msg: Message):
        self.messages.append(msg)
        if self.auto_scroll:
            self.scroll_offset = 0

    def set_streaming(self, name: str, text: str):
        self._streaming[name] = text

    def clear_streaming(self, name: str):
        self._streaming.pop(name, None)

    def assign_color(self, name: str, attr: int):
        self._colors[name] = attr

    def clear_messages(self):
        self.messages.clear()
        self._streaming.clear()
        self.scroll_offset = 0
        self.auto_scroll   = True

    # ── Rendering ────────────────────────────────────────────

    @staticmethod
    def _word_wrap(text: str, width: int) -> List[str]:
        if width <= 0:
            return [text]
        words  = text.split(" ")
        lines: List[str] = []
        cur    = ""
        for word in words:
            candidate = (cur + " " + word).lstrip() if cur else word
            if display_width(candidate) <= width:
                cur = candidate
            else:
                if cur:
                    lines.append(cur)
                if display_width(word) > width:
                    lines.append(truncate_to_display_width(word, width))
                    cur = ""
                else:
                    cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    def update(self):
        if not self.height or not self.width:
            return

        w        = self.width
        sys_attr = palette(self.theme["system_fg"])
        usr_attr = palette(self.theme["user_fg"])

        # Build a flat list of (rendered_line_text, attr) for every display row.
        all_lines: List[Tuple[str, int]] = []

        # Real messages + ephemeral streaming partials
        display: List[Message] = list(self.messages)
        for sname, partial in self._streaming.items():
            display.append(Message(sender=sname, text=partial + "▋"))

        for msg in display:
            prefix     = f"<{msg.timestamp}> <{msg.sender}> "
            attr       = self._colors.get(msg.sender, usr_attr)
            if msg.sender in ("*", "system"):
                attr   = sys_attr
            text_width = max(1, w - display_width(prefix))
            wrapped    = self._word_wrap(msg.text, text_width)
            indent     = " " * display_width(prefix)
            for i, segment in enumerate(wrapped):
                line = (prefix if i == 0 else indent) + segment
                all_lines.append((line, attr))

        # Clamp scroll offset
        max_offset         = max(0, len(all_lines) - self.height)
        self.scroll_offset = min(self.scroll_offset, max_offset)

        total   = len(all_lines)
        end     = total - self.scroll_offset
        start   = max(0, end - self.height)
        visible = all_lines[start:end]

        # Each content item must end with "\n" so the draw loop advances y.
        self.content = [[line + "\n", ALIGN_LEFT, attr] for line, attr in visible]

    def process_input(self, character: int):
        step = self.height or 10
        if character == 339:                        # PgUp
            self.scroll_offset += step
            self.auto_scroll    = False
        elif character == 338:                      # PgDn
            self.scroll_offset  = max(0, self.scroll_offset - step)
            self.auto_scroll    = self.scroll_offset == 0
        elif character == 262:                      # Home → scroll to top
            self.scroll_offset  = max(0, len(self.messages) - 1)
            self.auto_scroll    = False
        elif character == 360:                      # End → scroll to bottom
            self.scroll_offset  = 0
            self.auto_scroll    = True


class StatusBar(Pane):
    """Small status area for /exec output that isn't injected into the channel."""
    geometry = [EXPAND, 3]

    def __init__(self, name: str, theme: dict = None):
        super().__init__(name)
        self.theme  = theme or THEMES["dark"]
        self._lines: List[str] = []

    def set_text(self, text: str):
        self._lines = text.splitlines()
        self.hidden = not bool(self._lines)

    def update(self):
        attr  = palette(self.theme["status_fg"])
        shown = self._lines[: self.height or 3]
        text  = "\n".join(shown)
        self.content = [[text, ALIGN_LEFT, attr]]


class InputBar(Pane):
    """Single-line input with readline-style editing and command history."""
    geometry = [EXPAND, 1]
    wrap     = None

    def __init__(self, name: str, theme: dict = None):
        super().__init__(name)
        self.theme          = theme or THEMES["dark"]
        self.buffer         = ""
        self.cursor         = 0
        self.history:  List[str] = []
        self.history_pos     = -1
        self.history_draft   = ""   # draft saved while navigating history
        self.submit_callback = None  # callable(text) — sync, may schedule async work
        self.back_callback   = None  # callable() — Ctrl-B returns to the model picker

    def update(self):
        attr   = palette(self.theme["input_fg"])
        prompt = "> "
        text   = prompt + self.buffer
        self.content = [[text, ALIGN_LEFT, attr]]
        if self.window and self.coords:
            top, left          = self.coords[0][0]
            before             = self.buffer[: self.cursor]
            self.window.cursor_pos = (
                top,
                left + display_width(prompt) + display_width(before),
            )

    def process_input(self, character: int):
        if self.window:
            self.window.window.clear()

        if character in (10, 13):               # Enter
            text = self.buffer.strip()
            if text:
                self.history.append(text)
            self.history_pos  = -1
            self.history_draft = ""
            self.buffer = ""
            self.cursor = 0
            if text and self.submit_callback:
                self.submit_callback(text)

        elif character == 260:                  # ← Left
            self.cursor = max(0, self.cursor - 1)
        elif character == 261:                  # → Right
            self.cursor = min(len(self.buffer), self.cursor + 1)

        elif character in (263, 127, 8):        # Backspace
            if self.cursor > 0:
                self.buffer = (
                    self.buffer[: self.cursor - 1] + self.buffer[self.cursor :]
                )
                self.cursor -= 1
        elif character == 330:                  # Delete
            if self.cursor < len(self.buffer):
                self.buffer = (
                    self.buffer[: self.cursor] + self.buffer[self.cursor + 1 :]
                )

        elif character == 1:                    # Ctrl-A  beginning of line
            self.cursor = 0
        elif character == 5:                    # Ctrl-E  end of line
            self.cursor = len(self.buffer)
        elif character == 2:                    # Ctrl-B  back to model picker
            if self.back_callback:
                self.back_callback()
        elif character == 6:                    # Ctrl-F  forward char
            self.cursor = min(len(self.buffer), self.cursor + 1)

        elif character == 23:                   # Ctrl-W  kill word
            before     = self.buffer[: self.cursor].rstrip()
            parts      = before.rsplit(" ", 1)
            new_before = parts[0] + " " if len(parts) > 1 else ""
            self.buffer = new_before + self.buffer[self.cursor :]
            self.cursor = len(new_before)
        elif character == 21:                   # Ctrl-U  kill to start
            self.buffer = self.buffer[self.cursor :]
            self.cursor = 0
        elif character == 11:                   # Ctrl-K  kill to end
            self.buffer = self.buffer[: self.cursor]

        elif character == 259:                  # ↑  history prev
            if not self.history:
                return
            if self.history_pos == -1:
                self.history_draft = self.buffer
                self.history_pos   = len(self.history) - 1
            elif self.history_pos > 0:
                self.history_pos  -= 1
            self.buffer = self.history[self.history_pos]
            self.cursor = len(self.buffer)

        elif character == 258:                  # ↓  history next
            if self.history_pos == -1:
                return
            self.history_pos += 1
            if self.history_pos >= len(self.history):
                self.history_pos = -1
                self.buffer      = self.history_draft
            else:
                self.buffer = self.history[self.history_pos]
            self.cursor = len(self.buffer)

        elif 32 <= character < 256:
            try:
                ch          = chr(character)
                self.buffer = (
                    self.buffer[: self.cursor] + ch + self.buffer[self.cursor :]
                )
                self.cursor += 1
            except (ValueError, OverflowError):
                pass


class ModelPicker(Pane):
    """Full-screen model list shown at startup and on Ctrl-B.

    Rows are either section headings (not selectable) or model ids.
    ↑/↓ PgUp/PgDn Home/End to move, Enter to choose, Esc to go back to chat.
    """
    geometry = [EXPAND, EXPAND]

    def __init__(self, name: str, theme: dict = None):
        super().__init__(name)
        self.theme           = theme or THEMES["dark"]
        self.rows: List[Tuple[str, Optional[str]]] = []  # (label, model_id | None)
        self.selected        = 0
        self.notice          = "Loading models…"
        self.choose_callback = None  # callable(model_id)
        self.cancel_callback = None  # callable()

    def set_sections(self, sections: List[Tuple[str, List[str]]],
                     current: Optional[str] = None):
        """sections = [(heading, [model_id, …]), …]; empty sections are skipped."""
        self.rows = []
        for heading, models in sections:
            if not models:
                continue
            self.rows.append((heading, None))
            for m in models:
                label = m[len(OLLAMA_PREFIX):] if m.startswith(OLLAMA_PREFIX) else m
                self.rows.append((label, m))
        self.notice   = "" if self.rows else "No models found."
        selectable    = [i for i, (_, m) in enumerate(self.rows) if m]
        self.selected = next(
            (i for i in selectable if self.rows[i][1] == current),
            selectable[0] if selectable else 0,
        )

    def _move(self, delta: int):
        selectable = [i for i, (_, m) in enumerate(self.rows) if m]
        if not selectable:
            return
        pos = selectable.index(self.selected) if self.selected in selectable else 0
        pos = max(0, min(len(selectable) - 1, pos + delta))
        self.selected = selectable[pos]

    def update(self):
        if not self.height or not self.width:
            return
        w        = self.width
        head     = palette(self.theme["system_fg"])
        normal   = palette(self.theme["input_fg"])
        selected = palette(self.theme["header_fg"], self.theme["header_bg"])

        lines: List[Tuple[str, int]] = [
            (" Select a model  (↑/↓, Enter; Esc returns to chat)", head),
            ("", normal),
        ]
        if self.notice:
            lines.append(("  " + self.notice, head))
        sel_line = 0
        for i, (label, model) in enumerate(self.rows):
            if model is None:
                if i:
                    lines.append(("", normal))
                lines.append((f" {label}", head))
            else:
                if i == self.selected:
                    sel_line = len(lines)
                attr = selected if i == self.selected else normal
                text = f"   {label}"
                lines.append((text + " " * max(0, w - display_width(text)), attr))

        # Keep the selected row on screen.
        start = max(0, sel_line - self.height + 1)
        self.content = [
            [truncate_to_display_width(t, w) + "\n", ALIGN_LEFT, a]
            for t, a in lines[start : start + self.height]
        ]

    def process_input(self, character: int):
        if character == 259:                    # ↑
            self._move(-1)
        elif character == 258:                  # ↓
            self._move(1)
        elif character == 339:                  # PgUp
            self._move(-(self.height or 10))
        elif character == 338:                  # PgDn
            self._move(self.height or 10)
        elif character == 262:                  # Home
            self._move(-len(self.rows))
        elif character == 360:                  # End
            self._move(len(self.rows))
        elif character in (10, 13):             # Enter
            if self.rows and self.rows[self.selected][1] and self.choose_callback:
                self.choose_callback(self.rows[self.selected][1])
        elif character == 27:                   # Esc
            if self.cancel_callback:
                self.cancel_callback()


# ─────────────────────────────────────────────────────────────────────────────
# Model discovery
# ─────────────────────────────────────────────────────────────────────────────

def ollama_base_url() -> str:
    """Base URL of the local Ollama server, honouring $OLLAMA_HOST."""
    host = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/").replace("://0.0.0.0", "://127.0.0.1")


async def list_anthropic_models() -> List[str]:
    try:
        client = anthropic.AsyncAnthropic()
        return [m.id async for m in client.models.list(limit=100)]
    except Exception:
        return list(FALLBACK_ANTHROPIC_MODELS)


async def list_ollama_models() -> List[str]:
    """Models installed in local Ollama, or [] if the server isn't reachable."""
    def fetch() -> List[str]:
        with urllib.request.urlopen(ollama_base_url() + "/api/tags", timeout=2) as r:
            data = json.load(r)
        return sorted(m["name"] for m in data.get("models", []))
    try:
        return [OLLAMA_PREFIX + name for name in await asyncio.to_thread(fetch)]
    except Exception:
        return []


def client_for(model: str) -> Tuple[anthropic.AsyncAnthropic, str]:
    """Return (client, api_model_id).  Ollama speaks the Anthropic Messages API."""
    if model.startswith(OLLAMA_PREFIX):
        client = anthropic.AsyncAnthropic(base_url=ollama_base_url(), api_key="ollama")
        return client, model[len(OLLAMA_PREFIX):]
    return anthropic.AsyncAnthropic(), model


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

class ThinktankApp:
    """Wires together the TUI, participants, and async API calls."""

    def __init__(self, args: argparse.Namespace):
        self.args          = args
        self.channel       = getattr(args, "channel", None) or "#thinktank"
        self.theme_name    = getattr(args, "theme",   None) or "matrix"
        self.theme         = THEMES.get(self.theme_name, THEMES["matrix"])
        self.default_model = getattr(args, "model",   None) or DEFAULT_MODEL

        self.constrained   = bool(getattr(args, "constrain", True))

        self.participants:    Dict[str, Participant] = {}
        self.channel_history: List[Message]         = []
        self.solo_name:       Optional[str]         = None
        self._logfile                               = None
        self._responding                            = False
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._loop:         Optional[asyncio.AbstractEventLoop] = None

        # Build window
        self.window   = Window(blocking=False)
        self.window.delay = 0.020

        self.header   = HeaderBar("header",   channel=self.channel, theme=self.theme)
        self.scroll   = ScrollBuffer("scroll", theme=self.theme)
        self.status   = StatusBar("status",   theme=self.theme)
        self.inputbar = InputBar("input",     theme=self.theme)
        self.picker   = ModelPicker("picker", theme=self.theme)

        self.status.hidden           = True
        self.inputbar.submit_callback = self._on_submit
        self.inputbar.back_callback   = self._show_picker
        self.picker.choose_callback   = self._choose_model
        self.picker.cancel_callback   = self._hide_picker

        for pane in (self.header, self.picker, self.scroll, self.status, self.inputbar):
            self.window.add(pane)

        self._picking        = False
        self._status_visible = False
        self._model_chosen   = False

    # ── Model picker ─────────────────────────────────────────────────────────

    def _set_picking(self, picking: bool):
        if picking and not self._picking:
            self._status_visible = not self.status.hidden
        self._picking       = picking
        self.picker.hidden  = not picking
        self.picker.active  = picking
        for pane in (self.scroll, self.inputbar):
            pane.hidden = picking
            pane.active = not picking
        self.status.hidden = picking or not self._status_visible
        if self.window.window:
            self.window.window.clear()
        if picking:
            self.header.right = " [select model] "
        else:
            self._update_header_right()

    def _show_picker(self):
        self._set_picking(True)
        if self._loop:
            self._loop.create_task(self._refresh_models())

    def _hide_picker(self):
        if self._model_chosen:      # nothing to go back to before the first pick
            self._set_picking(False)

    async def _refresh_models(self):
        self.picker.notice = "Loading models…"
        claude, local = await asyncio.gather(
            list_anthropic_models(), list_ollama_models()
        )
        self.picker.set_sections(
            [("Anthropic", claude), (f"Ollama ({ollama_base_url()})", local)],
            current=self.default_model,
        )

    def _choose_model(self, model: str):
        """Make *model* the model for every participant and return to chat."""
        changed             = model != self.default_model or not self._model_chosen
        self.default_model  = model
        self._model_chosen  = True
        for p in self.participants.values():
            p.model = model
        self._set_picking(False)
        if changed:
            self._system_message(f"Model: {model}")

    # ── Logging ──────────────────────────────────────────────────────────────

    def _open_logfile(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._logfile = open(path, "a", encoding="utf-8")

    def _log(self, msg: Message):
        if self._logfile:
            self._logfile.write(msg.log_line() + "\n")
            self._logfile.flush()

    @staticmethod
    def _default_log_path() -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        return os.path.join("logs", f"{ts}.log")

    # ── Messages ─────────────────────────────────────────────────────────────

    def _add_message(self, sender: str, text: str, log: bool = True) -> Message:
        msg = Message(sender=sender, text=text)
        self.channel_history.append(msg)
        self._ensure_color(sender)
        self.scroll.add_message(msg)
        if log:
            self._log(msg)
        return msg

    def _system_message(self, text: str):
        """Display a system notice (not added to channel_history or log)."""
        msg = Message(sender="*", text=text)
        self.scroll.add_message(msg)

    def _ensure_color(self, name: str):
        if name in self.scroll._colors:
            return
        colors = self.theme["participant_colors"]
        if name == USER_SENDER:
            attr = palette(self.theme["user_fg"])
        elif name in ("*", "system", "exec"):
            attr = palette(self.theme["system_fg"])
        else:
            idx  = sum(
                1 for k in self.scroll._colors
                if k not in (USER_SENDER, "*", "system", "exec")
            ) % len(colors)
            attr = palette(colors[idx])
        self.scroll._colors[name] = attr

    def _update_header_right(self):
        n    = len(self.participants)
        mode = "C" if self.constrained else "F"
        pstr = f"{n}p " if n else ""
        self.header.right = f" [{self.default_model}] [{pstr}{mode}] "

    # ── API context building ──────────────────────────────────────────────────

    def _build_messages_for(self, participant: Participant) -> List[dict]:
        """Build an alternating user/assistant message list for the Anthropic API."""
        history = self.channel_history
        if participant.paused:
            history = history[: participant.paused_at_index]

        raw: List[dict]       = []
        pending_other: List[str] = []

        for msg in history:
            if msg.sender == participant.name:
                if pending_other:
                    raw.append({"role": "user",
                                "content": "\n".join(pending_other)})
                    pending_other = []
                raw.append({"role": "assistant", "content": msg.text})
            else:
                pending_other.append(f"<{msg.sender}> {msg.text}")

        if pending_other:
            raw.append({"role": "user",
                        "content": "\n".join(pending_other)})

        if not raw:
            return []

        # Merge consecutive same-role turns (safety net)
        merged: List[dict] = [raw[0]]
        for m in raw[1:]:
            if m["role"] == merged[-1]["role"]:
                merged[-1] = {
                    "role":    m["role"],
                    "content": merged[-1]["content"] + "\n" + m["content"],
                }
            else:
                merged.append(m)

        # API requires first turn = user
        if merged[0]["role"] == "assistant":
            merged.insert(0, {"role": "user", "content": "…"})

        # Must end with user (we want an assistant completion)
        if merged[-1]["role"] != "user":
            return []

        return merged

    # ── API streaming ─────────────────────────────────────────────────────────

    async def _stream_participant(
        self, participant: Participant, prefill: str = ""
    ) -> Tuple[str, str]:
        """Fetch one participant's response.  Returns (name, full_text).

        Constrain mode: single non-streaming call; first line, truncated to
        IRC_MAX_CHARS.  Full mode: streaming with live preview in the buffer.
        """
        messages = self._build_messages_for(participant)
        if not messages:
            return participant.name, ""

        system = PARTICIPANT_PREAMBLE + "\n\n" + participant.system_prompt
        if self.constrained:
            system = system + CONSTRAIN_SYSTEM_SUFFIX

        client, api_model = client_for(participant.model)
        full_text         = ""

        try:
            if self.constrained:
                # Local thinking models can spend a small budget entirely on
                # reasoning, so give Ollama room; the reply is truncated anyway.
                local = participant.model.startswith(OLLAMA_PREFIX)
                resp = await client.messages.create(
                    model=api_model,
                    max_tokens=DEFAULT_MAX_TOKENS if local else IRC_MAX_TOKENS,
                    system=system,
                    messages=messages,
                )
                # Skip thinking blocks some local models emit ahead of the text.
                raw       = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                lines     = raw.strip().splitlines()
                full_text = lines[0].strip()[:IRC_MAX_CHARS] if lines else ""
                return participant.name, full_text

            else:
                full_text    = prefill
                api_messages = (
                    messages + [{"role": "assistant", "content": prefill}]
                    if prefill else messages
                )
                async with client.messages.stream(
                    model=api_model,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    system=system,
                    messages=api_messages,
                ) as stream:
                    async for chunk in stream.text_stream:
                        full_text += chunk
                        self.scroll.set_streaming(participant.name, full_text)

                self.scroll.clear_streaming(participant.name)
                return participant.name, full_text

        except asyncio.CancelledError:
            self.scroll.clear_streaming(participant.name)
            # Save partial for prefill on non-Opus models (Opus 4.6 returns 400)
            if not self.constrained and not participant.is_opus and full_text:
                participant.prefill_buffer = full_text
            raise

        except Exception as exc:
            self.scroll.clear_streaming(participant.name)
            self._system_message(f"Error ({participant.name}): {exc}")
            return participant.name, ""

    async def _query_participants(self):
        """
        Cancel-and-restart parallel query strategy:
        1. Start all active participants in parallel.
        2. When the first stream finishes, cancel the rest.
        3. Re-query the cancelled participants with updated context (first reply included).
        """
        if self._responding:
            return
        self._responding = True

        try:
            targets = [
                p for p in self.participants.values()
                if not p.paused and not p.muted
            ]
            if self.solo_name:
                targets = [p for p in targets if p.name == self.solo_name]
            if not targets:
                return

            # Collect any saved prefills
            prefills: Dict[str, str] = {}
            for p in targets:
                if p.prefill_buffer:
                    prefills[p.name]   = p.prefill_buffer
                    p.prefill_buffer   = ""

            self.header.right = " [thinking…] "

            # Phase 1: launch all
            task_map: Dict[str, asyncio.Task] = {
                p.name: asyncio.create_task(
                    self._stream_participant(p, prefill=prefills.get(p.name, ""))
                )
                for p in targets
            }
            self._active_tasks = task_map

            # Wait for the first to finish
            done, _ = await asyncio.wait(
                list(task_map.values()), return_when=asyncio.FIRST_COMPLETED
            )

            first_task               = next(iter(done))
            name_by_task             = {v: k for k, v in task_map.items()}
            first_name               = name_by_task[first_task]
            try:
                _, first_text = first_task.result()
            except Exception:
                first_text = ""

            if first_text:
                self._add_message(first_name, first_text)

            # Cancel the rest and let them save their prefills
            remaining_names = [n for n in task_map if n != first_name]
            for name in remaining_names:
                task_map[name].cancel()
            if remaining_names:
                await asyncio.gather(
                    *[task_map[n] for n in remaining_names],
                    return_exceptions=True,
                )

            # Phase 2: re-query remaining with updated context
            remaining = [p for p in targets if p.name in remaining_names]
            if remaining:
                task_map2: Dict[str, asyncio.Task] = {
                    p.name: asyncio.create_task(
                        self._stream_participant(
                            p, prefill=p.prefill_buffer or ""
                        )
                    )
                    for p in remaining
                }
                for p in remaining:
                    p.prefill_buffer = ""
                self._active_tasks = task_map2

                results = await asyncio.gather(
                    *task_map2.values(), return_exceptions=True
                )
                for res in results:
                    if isinstance(res, Exception):
                        continue
                    name, text = res
                    if text:
                        self._add_message(name, text)

        finally:
            self._responding = False
            self._active_tasks = {}
            self._update_header_right()

    # ── Input dispatch ────────────────────────────────────────────────────────

    def _on_submit(self, text: str):
        """Sync callback wired to InputBar.submit_callback."""
        if self._loop:
            self._loop.create_task(self._handle_input(text))

    async def _handle_input(self, text: str):
        if text.startswith("/"):
            await self._dispatch_command(text)
        else:
            self._add_message(USER_SENDER, text)
            await self._query_participants()

    # ── Slash commands ────────────────────────────────────────────────────────

    _COMMANDS = {
        "/add", "/remove", "/redefine",
        "/list", "/names", "/n",
        "/mute", "/unmute",
        "/pause", "/unpause",
        "/solo",
        "/save", "/load",
        "/reply",
        "/exec",
        "/clear",
        "/log",
        "/theme",
        "/constrain",
        "/help",
        "/quit", "/q", "/exit",
    }

    async def _dispatch_command(self, text: str):
        parts = text.split()
        cmd   = parts[0].lower()
        args  = parts[1:]

        handlers = {
            "/add":      self._cmd_add,
            "/remove":   self._cmd_remove,
            "/redefine": self._cmd_redefine,
            "/list":     self._cmd_list,
            "/names":    self._cmd_list,
            "/n":        self._cmd_list,
            "/mute":     self._cmd_mute,
            "/unmute":   self._cmd_unmute,
            "/pause":    self._cmd_pause,
            "/unpause":  self._cmd_unpause,
            "/solo":     self._cmd_solo,
            "/save":     self._cmd_save,
            "/load":     self._cmd_load,
            "/reply":    self._cmd_reply,
            "/exec":     self._cmd_exec,
            "/clear":    self._cmd_clear,
            "/log":      self._cmd_log,
            "/theme":     self._cmd_theme,
            "/constrain": self._cmd_constrain,
            "/help":      self._cmd_help,
            "/quit":     self._cmd_quit,
            "/q":        self._cmd_quit,
            "/exit":     self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            await handler(args)
        else:
            self._system_message(f"Unknown command: {cmd}  (try /help)")

    # Participants ─────────────────────────────────────────────────────────────

    async def _cmd_add(self, args: List[str]):
        if not args:
            self._system_message("Usage: /add <name> [system_prompt]")
            return
        name = args[0]
        if len(args) > 1:
            prompt = " ".join(args[1:])
        else:
            prompt = await self._open_editor(f"# System prompt for {name}\n")
            if prompt is None:
                return
        p = Participant(name=name, system_prompt=prompt, model=self.default_model)
        self.participants[name] = p
        self._ensure_color(name)
        self._update_header_right()
        self._system_message(f"Added: {name}")

    async def _cmd_remove(self, args: List[str]):
        if not args:
            self._system_message("Usage: /remove <name>")
            return
        name = args[0]
        if name not in self.participants:
            self._system_message(f"No such participant: {name}")
            return
        del self.participants[name]
        self._update_header_right()
        self._system_message(f"Removed: {name}")

    async def _cmd_redefine(self, args: List[str]):
        if not args:
            self._system_message("Usage: /redefine <name>")
            return
        name = args[0]
        p    = self.participants.get(name)
        if not p:
            self._system_message(f"No such participant: {name}")
            return
        prompt = await self._open_editor(p.system_prompt)
        if prompt is not None:
            p.system_prompt = prompt
            self._system_message(f"Redefined: {name}")

    async def _cmd_list(self, args: List[str]):
        if not self.participants:
            self._system_message("No participants.")
            return
        for p in self.participants.values():
            flags   = ""
            if p.muted:  flags += " [muted]"
            if p.paused: flags += " [paused]"
            snippet = p.system_prompt[:60].replace("\n", " ")
            if len(p.system_prompt) > 60:
                snippet += "…"
            self._system_message(f"  {p.name} ({p.model}){flags}: {snippet}")

    async def _cmd_mute(self, args: List[str]):
        if not args:
            self._system_message("Usage: /mute <name>")
            return
        p = self.participants.get(args[0])
        if not p:
            self._system_message(f"No such participant: {args[0]}")
            return
        p.muted = True
        self._system_message(f"Muted: {args[0]}")

    async def _cmd_unmute(self, args: List[str]):
        if not args:
            self._system_message("Usage: /unmute <name>")
            return
        p = self.participants.get(args[0])
        if not p:
            self._system_message(f"No such participant: {args[0]}")
            return
        p.muted = False
        self._system_message(f"Unmuted: {args[0]}")

    async def _cmd_pause(self, args: List[str]):
        if not args:
            self._system_message("Usage: /pause <name>")
            return
        p = self.participants.get(args[0])
        if not p:
            self._system_message(f"No such participant: {args[0]}")
            return
        p.paused          = True
        p.paused_at_index = len(self.channel_history)
        self._system_message(f"Paused: {args[0]}")

    async def _cmd_unpause(self, args: List[str]):
        if not args:
            self._system_message("Usage: /unpause <name>")
            return
        p = self.participants.get(args[0])
        if not p:
            self._system_message(f"No such participant: {args[0]}")
            return
        p.paused = False
        self._system_message(f"Unpaused: {args[0]}")

    async def _cmd_solo(self, args: List[str]):
        if not args:
            self.solo_name = None
            self._system_message("Solo cleared")
        else:
            name = args[0]
            if name not in self.participants:
                self._system_message(f"No such participant: {name}")
                return
            self.solo_name = name
            self._system_message(f"Solo: {name}")

    async def _cmd_save(self, args: List[str]):
        path = args[0] if args else "participants.json"
        data = [p.to_dict() for p in self.participants.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._system_message(f"Saved {len(data)} participants → {path}")

    async def _cmd_load(self, args: List[str]):
        if not args:
            self._system_message("Usage: /load <file.json>")
            return
        path = args[0]
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            added = 0
            for item in data:
                p = Participant.from_dict(item)
                self.participants[p.name] = p
                self._ensure_color(p.name)
                added += 1
            self._update_header_right()
            self._system_message(f"Loaded {added} participants from {path}")
        except Exception as exc:
            self._system_message(f"Load error: {exc}")

    # Messaging ────────────────────────────────────────────────────────────────

    async def _cmd_reply(self, args: List[str]):
        if len(args) < 2:
            self._system_message("Usage: /reply <name> <message>")
            return
        name = args[0]
        if name not in self.participants:
            self._system_message(f"No such participant: {name}")
            return
        text     = " ".join(args[1:])
        self._add_message(USER_SENDER, text)
        old_solo, self.solo_name = self.solo_name, name
        await self._query_participants()
        self.solo_name = old_solo

    async def _cmd_exec(self, args: List[str]):
        inject = False
        if args and args[0] == "-e":
            inject = True
            args   = args[1:]
        cmd = " ".join(args)
        if not cmd:
            self._system_message("Usage: /exec [-e] <command>")
            return
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            output = (result.stdout + result.stderr).strip()
        except subprocess.TimeoutExpired:
            output = "Error: command timed out"
        except Exception as exc:
            output = f"Error: {exc}"

        if inject:
            self._add_message("exec", output)
            await self._query_participants()
        else:
            self.status.set_text(output)
            self.status.hidden = False

    # Session ──────────────────────────────────────────────────────────────────

    async def _cmd_clear(self, args: List[str]):
        self.scroll.clear_messages()
        if self.window.window:
            self.window.window.clear()

    async def _cmd_log(self, args: List[str]):
        if not args:
            if self._logfile:
                self._system_message(f"Logging → {self._logfile.name}")
            else:
                self._system_message("Logging off")
            return
        if args[0].lower() == "off":
            if self._logfile:
                self._logfile.close()
                self._logfile = None
            self._system_message("Logging paused")
        else:
            if self._logfile:
                self._logfile.close()
            self._open_logfile(args[0])
            self._system_message(f"Logging → {args[0]}")

    async def _cmd_theme(self, args: List[str]):
        if not args:
            self._system_message("Themes: " + ", ".join(THEMES.keys()))
            return
        name = args[0]
        if name not in THEMES:
            self._system_message(
                f"Unknown theme '{name}'.  Available: {', '.join(THEMES.keys())}"
            )
            return
        self.theme_name = name
        self.theme      = THEMES[name]
        for pane in (self.header, self.picker, self.scroll, self.status, self.inputbar):
            pane.theme = self.theme
        self.scroll._colors.clear()
        self._ensure_color(USER_SENDER)
        for p in self.participants.values():
            self._ensure_color(p.name)
        if self.window.window:
            self.window.window.clear()
        self._system_message(f"Theme: {name}")

    async def _cmd_constrain(self, args: List[str]):
        if not args:
            state = "on" if self.constrained else "off"
            self._system_message(f"Constrain mode: {state}  (/constrain on|off)")
            return
        val = args[0].lower()
        if val in ("on", "1", "true", "yes"):
            self.constrained = True
        elif val in ("off", "0", "false", "no"):
            self.constrained = False
        else:
            self._system_message("Usage: /constrain [on|off]")
            return
        self._update_header_right()
        state = "on" if self.constrained else "off"
        self._system_message(f"Constrain mode: {state}")

    async def _cmd_help(self, args: List[str]):
        lines = [
            "Participants:",
            "  /add <name> [prompt]   /remove <name>   /redefine <name>",
            "  /list  /names  /n",
            "  /mute <name>   /unmute <name>",
            "  /pause <name>  /unpause <name>",
            "  /solo [name]           (no arg = clear solo)",
            "  /save [file.json]      /load <file.json>",
            "Messaging:",
            "  /reply <name> <msg>",
            "  /exec [-e] <cmd>       (-e injects output as channel message)",
            "  /clear",
            "Session:",
            "  /log [path|off]        /theme [name]",
            "  /constrain [on|off]    one-line IRC-safe mode (default: on)",
            "  /help                  /quit",
            "Keys:",
            "  ↑/↓ in input box → command history",
            "  Ctrl-B               → back to the model list",
            "  PgUp/PgDn            → scroll buffer",
        ]
        for line in lines:
            self._system_message(line)

    async def _cmd_quit(self, args: List[str]):
        self.window.running = False

    # ── $EDITOR helper ────────────────────────────────────────────────────────

    async def _open_editor(self, initial: str = "") -> Optional[str]:
        """Suspend curses, open $EDITOR with initial text, return result."""
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        fd, fname = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(initial)

            # Suspend curses
            _curses.nocbreak()
            self.window.window.keypad(0)
            _curses.echo()
            _curses.resetty()
            _curses.endwin()

            subprocess.run([editor, fname])

            # Restore curses
            new_win = _curses.initscr()
            _curses.savetty()
            _curses.start_color()
            _curses.use_default_colors()
            new_win.leaveok(1)
            _curses.raw()
            new_win.keypad(1)
            _curses.noecho()
            _curses.cbreak()
            _curses.nonl()
            new_win.nodelay(1)
            new_win.clear()
            self.window.window = new_win

            with open(fname, encoding="utf-8") as f:
                text = f.read().strip()
            return text or None

        except Exception as exc:
            self._system_message(f"Editor error: {exc}")
            return None
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass

    # ── Session resume ────────────────────────────────────────────────────────

    def _resume_log(self, path: str):
        """Replay an IRC-format log into the scroll buffer (no re-responses)."""
        try:
            count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or not line.startswith("["):
                        continue
                    try:
                        bracket_end = line.index("]")
                        full_ts     = line[1:bracket_end]
                        rest        = line[bracket_end + 2:].strip()
                        name_end    = rest.index("> ", 1)
                        name        = rest[1:name_end]
                        text        = rest[name_end + 2:]
                        ts          = full_ts[11:19] if len(full_ts) >= 19 else full_ts
                        msg = Message(
                            sender=name, text=text,
                            timestamp=ts, full_timestamp=full_ts,
                        )
                        self.channel_history.append(msg)
                        self._ensure_color(name)
                        self.scroll.add_message(msg)
                        count += 1
                    except (ValueError, IndexError):
                        continue
            self._system_message(f"Resumed {count} messages from {path}")
        except Exception as exc:
            self._system_message(f"Resume error: {exc}")

    # ── Startup helpers ───────────────────────────────────────────────────────

    def _load_participants_file(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            p = Participant.from_dict(item)
            if not p.model:
                p.model = self.default_model
            self.participants[p.name] = p
            self._ensure_color(p.name)

    # ── Async main ────────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # Initialise curses manually so we can drive the event loop ourselves.
        win = _curses.initscr()
        _curses.savetty()
        _curses.start_color()
        _curses.use_default_colors()
        win.leaveok(1)
        _curses.raw()
        win.keypad(1)
        _curses.noecho()
        _curses.cbreak()
        _curses.nonl()
        win.nodelay(1)
        try:
            _curses.set_escdelay(25)    # make Esc in the model picker responsive
        except Exception:
            pass
        self.window.window  = win
        self.window.running = True

        try:
            self._ensure_color(USER_SENDER)

            if getattr(self.args, "participants", None):
                try:
                    self._load_participants_file(self.args.participants)
                    self._update_header_right()
                except Exception as exc:
                    self._system_message(f"Participants load error: {exc}")

            if getattr(self.args, "resume", None):
                self._resume_log(self.args.resume)

            if not getattr(self.args, "no_log", False):
                log_path = getattr(self.args, "logfile", None) or self._default_log_path()
                try:
                    self._open_logfile(log_path)
                except Exception as exc:
                    self._system_message(f"Log error: {exc}")

            self._system_message(
                f"Welcome to {self.channel}  —  /help for commands, "
                "Ctrl-B for models, /quit to exit"
            )
            self._show_picker()

            while self.window.running:
                self.window.cycle()
                await asyncio.sleep(0.020)

        finally:
            for task in list(self._active_tasks.values()):
                task.cancel()
            if self._active_tasks:
                await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
            if self._logfile:
                self._logfile.close()
            try:
                _curses.nocbreak()
                win.keypad(0)
                _curses.echo()
                _curses.resetty()
                _curses.endwin()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="thinktank",
        description="IRC-style multi-agent Claude chat",
    )
    p.add_argument(
        "-p", "--participants", metavar="FILE",
        help="load participant definitions on startup",
    )
    p.add_argument(
        "-l", "--logfile", metavar="PATH",
        help="log destination (default: ./logs/YYYY-MM-DD_HH-MM.log)",
    )
    p.add_argument(
        "--no-log", action="store_true",
        help="disable logging",
    )
    p.add_argument(
        "--model", metavar="MODEL-ID",
        help=f"model highlighted in the startup model list (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--channel", metavar="NAME",
        help="channel name shown in header bar (default: #thinktank)",
    )
    p.add_argument(
        "--resume", metavar="LOGFILE",
        help="replay previous session into scroll buffer on load (no re-responses)",
    )
    p.add_argument(
        "--theme", metavar="NAME",
        choices=list(THEMES.keys()),
        help="color theme (default: matrix)",
    )
    p.add_argument(
        "--constrain", action=argparse.BooleanOptionalAction, default=True,
        help="constrain mode: one-line IRC-safe responses (default: on); disable with --no-constrain",
    )
    return p.parse_args()


def main():
    args = parse_args()
    app  = ThinktankApp(args)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

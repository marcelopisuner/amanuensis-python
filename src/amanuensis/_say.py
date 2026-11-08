"""What the package tells the user while it works.

Messages go to standard error, one line per sentence, each opening with an
ASCII bullet: ``i`` information, ``v`` done, ``!`` worth a look, ``x`` a
problem, ``*`` an item of a list, ``>`` the next step. Warnings (something to
check; the call went ahead) are ``warnings.warn(..., AmanuensisWarning)``,
errors are exceptions (``_errors``). ``options.quiet`` or a function's
``quiet=True`` silences the messages, never warnings or errors.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import warnings

from ._errors import AmanuensisWarning, format_lines
from ._options import options

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_quiet_depth = 0


@contextlib.contextmanager
def quietly(on: bool = True):
    """Silence messages inside the block (warnings still show)."""
    global _quiet_depth
    if on:
        _quiet_depth += 1
    try:
        yield
    finally:
        if on:
            _quiet_depth -= 1


def is_quiet() -> bool:
    return bool(options.quiet) or _quiet_depth > 0


def _line(bullet: str, text: str) -> str:
    text = str(text)
    lines = text.split("\n")
    first = f"{bullet} {lines[0]}" if bullet.strip() else f"  {lines[0]}"
    return "\n".join([first] + ["  " + l for l in lines[1:]])


def say(*lines) -> None:
    """Print message lines: each a string (bullet ``i``) or a ``(bullet, text)`` pair."""
    if is_quiet():
        return
    out = []
    for l in lines:
        if l is None:
            continue
        bullet, text = l if isinstance(l, tuple) else ("i", l)
        if text is None or text == "":
            continue
        out.append(_line(bullet, text))
    if out:
        print("\n".join(out), file=sys.stderr, flush=True)


def info(text: str) -> None:
    say(("i", text))


def done(text: str) -> None:
    say(("v", text))


def attention(text: str) -> None:
    say(("!", text))


def warn(head: str, *details) -> None:
    """Warn (``AmanuensisWarning``), pointing at the user's own line of code."""
    msg = format_lines(head, details)
    if sys.version_info >= (3, 12):
        warnings.warn(msg, AmanuensisWarning, skip_file_prefixes=(_PACKAGE_DIR,))
    else:  # pragma: no cover - older Pythons point at the package's own line
        warnings.warn(msg, AmanuensisWarning, stacklevel=3)


def plural(n, one: str, many: str | None = None) -> str:
    """``1 page``, ``3 pages``, with thousands separators."""
    many = many if many is not None else one + "s"
    return f"{int(n):,} {one if n == 1 else many}"


def listing(items, n: int = 5) -> str:
    items = [str(x) for x in items]
    if len(items) > n:
        return ", ".join(items[:n]) + f" and {len(items) - n} more"
    return ", ".join(items)


class Progress:
    """A one-line progress counter on standard error.

    In a terminal the line is redrawn in place; elsewhere (a notebook, a log
    file) a line is printed every tenth of the way, so a long run neither
    floods the output nor stays silent.
    """

    def __init__(self, label: str, total: int, status=None):
        self.label = label
        self.total = max(int(total), 0)
        self.n = 0
        self.status = status
        self.tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self.step = max(1, self.total // 10)
        self.started = time.monotonic()
        self.shown = False

    def update(self, inc: int = 1) -> None:
        self.n += inc
        if is_quiet() or not self.total:
            return
        extra = f" | {self.status()}" if self.status else ""
        text = f"{self.label} {self.n}/{self.total}{extra}"
        if self.tty:
            sys.stderr.write("\r" + text.ljust(60))
            sys.stderr.flush()
            self.shown = True
        elif self.n % self.step == 0 or self.n == self.total:
            print(text, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.tty and self.shown and not is_quiet():
            sys.stderr.write("\n")
            sys.stderr.flush()

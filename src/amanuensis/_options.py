"""Package-wide options: ``amanuensis.options``.

The R package reads ``options(amanuensis.*)``; here they are attributes of one
object, set directly (``amanuensis.options.ask = False``) or for a block of
code with ``with amanuensis.options.override(ask=False): ...``. Unknown names
raise, so a misspelt option is never silently ignored.

    ask           ask questions (menus, yes/no) in interactive sessions
    interactive   None (detect), True or False
    quiet         True to print only warnings and errors
    sleep         function used for every wait (tests pass a no-op)
    poll_seconds  seconds between checks of a batch job (None: 30; 0 for practice)
    session       keep pages read without a results folder in memory
    long_run      pages from which a run without a folder suggests one
    batch_schema  None (by codebook), "json" or "openapi": the batch schema channel
    batch_max_bytes  largest batch file written (Google accepts 2 GB)
    base_url      another address for Google's API (a proxy)
    next_steps    print the "Next:" hints after a run
"""

from __future__ import annotations

import builtins
import contextlib
import sys
import time

_DEFAULTS = {
    "ask": True,
    "interactive": None,
    "quiet": False,
    "sleep": time.sleep,
    "poll_seconds": None,
    "session": True,
    "long_run": 50,
    "batch_schema": None,
    "batch_max_bytes": 1.9e9,
    "base_url": None,
    "next_steps": True,
}


class Options:
    """The package's options (see the module docstring)."""

    def __init__(self):
        object.__setattr__(self, "_values", dict(_DEFAULTS))

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(f"amanuensis has no option {name!r}; options: {', '.join(sorted(values))}")

    def __setattr__(self, name, value):
        if name not in self._values:
            raise AttributeError(f"amanuensis has no option {name!r}; options: {', '.join(sorted(self._values))}")
        self._values[name] = value

    @contextlib.contextmanager
    def override(self, **kwargs):
        """Set options for a block of code; the old values come back after it."""
        old = {}
        for k, v in kwargs.items():
            old[k] = getattr(self, k)
            setattr(self, k, v)
        try:
            yield self
        finally:
            for k, v in old.items():
                setattr(self, k, v)

    def reset(self):
        """Back to the defaults."""
        self._values.clear()
        self._values.update(_DEFAULTS)

    def __repr__(self):
        rows = [f"  {k:<16}{v!r}" for k, v in sorted(self._values.items())]
        return "amanuensis options:\n" + "\n".join(rows)


options = Options()


def is_interactive() -> bool:
    """A person at the keyboard: the Python prompt, IPython or Jupyter."""
    if options.interactive is not None:
        return bool(options.interactive)
    return bool(hasattr(sys, "ps1") or sys.flags.interactive or getattr(builtins, "__IPYTHON__", False))


def can_ask() -> bool:
    return is_interactive() and bool(options.ask)


def sleep(seconds: float) -> None:
    options.sleep(seconds)


def ask_yes_no(question: str, default: bool = True) -> bool:
    """A yes/no question; the default on an empty answer. False when no one can answer."""
    if not can_ask():
        return default
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {hint} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "s", "si", "sí"):
            return True
        if answer in ("n", "no"):
            return False


def menu(choices: list[str], title: str) -> int:
    """A numbered menu; returns the 1-based choice, 0 for none."""
    print(title, file=sys.stderr)
    for i, c in enumerate(choices, 1):
        print(f"  {i}: {c}", file=sys.stderr)
    answer = input("Selection: ").strip()
    try:
        k = int(answer)
    except ValueError:
        return 0
    return k if 1 <= k <= len(choices) else 0

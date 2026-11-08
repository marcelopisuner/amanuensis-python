"""Exceptions and warnings.

Every error the package raises on purpose is an ``AmanuensisError``, and its
message is written for the person running the code: what went wrong, then
lines starting with ``i`` saying what to do. Errors about the user's own
arguments or files are also ``ValueError`` (``InputError``), so code that
already catches ``ValueError`` keeps working.

A message is built from lines: the first is the headline, the others are
``(bullet, text)`` pairs, bullet one of ``i`` (what to do), ``x`` (what was
found), ``*`` (an item) or ``!`` (worth knowing). These are the ASCII bullets
the R package's console library falls back to; console output has no glyphs.
"""

from __future__ import annotations


def format_lines(head: str, details=()) -> str:
    """The text of a message: the headline, then one indented line per detail."""
    out = [str(head)]
    for d in details:
        if isinstance(d, tuple):
            bullet, text = d
        else:
            bullet, text = "i", d
        if text is None or text == "":
            continue
        for k, line in enumerate(str(text).split("\n")):
            prefix = f"  {bullet} " if k == 0 else "    "
            out.append(prefix + line)
    return "\n".join(out)


class AmanuensisError(Exception):
    """Base class of the errors raised by amanuensis."""

    def __init__(self, head: str, *details, **fields):
        self.head = str(head)
        self.details = list(details)
        for k, v in fields.items():
            setattr(self, k, v)
        super().__init__(format_lines(self.head, self.details))


class InputError(AmanuensisError, ValueError):
    """An argument or a file the user gave cannot be used as it is."""


class NotFoundError(InputError):
    """A file or folder the user named does not exist."""


class MissingKeyError(AmanuensisError):
    """GEMINI_API_KEY is not set (or does not look like a key)."""


class ContractError(AmanuensisError):
    """Results made with other instructions, codebook or settings are in the way."""


class GoldEmptyError(InputError):
    """A benchmark file has no page marked done (or no values) yet."""


class ApiError(AmanuensisError):
    """Google refused a request. ``status`` is the HTTP status (or None)."""

    status: int | None = None


class NetworkError(AmanuensisError, ConnectionError):
    """Google could not be reached, after the retries."""


class InternalError(AmanuensisError):
    """A problem in the package itself; please report it."""


class Interrupted(KeyboardInterrupt):
    """Ctrl-C during a run. ``result`` holds the pages read before it.

    It is a KeyboardInterrupt, so a script stops as usual; in a notebook,
    ``sys.last_value.result`` (or ``except KeyboardInterrupt as e: e.result``)
    gets the partial result. Every page read is saved before this is raised.
    """

    def __init__(self, message: str, result=None):
        super().__init__(message)
        self.result = result


class AmanuensisWarning(UserWarning):
    """Something worth checking; the call went ahead."""

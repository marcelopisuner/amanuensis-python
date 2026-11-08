"""The package's table of Gemini models: thinking levels, output cap, timeout
and price.

Model names change every few months, so the table carries the date it was
checked and every function treats an unknown model as allowed but unchecked
(no price, no setting checks) rather than an error. Prices are US dollars per
million tokens from https://ai.google.dev/gemini-api/docs/pricing, read
2026-10-23 (the R package's table, unchanged). Thinking tokens are billed at
the output price. Batch and flex cost half.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ._errors import InputError
from ._say import say

MODELS_CHECKED = "2026-10-23"
PRACTICE_MODELS = ("mock", "example")
DEFAULT_MODEL = "gemini-3.8-flash"
THINKING_LEVELS = ("minimal", "low", "medium", "high")

_COLUMNS = ["model", "status", "routes_to", "thinking", "max_output", "ultra_high", "timeout",
            "price_input", "price_output", "price_cached", "next_from", "next_input", "next_output",
            "next_cached"]

# model, status, routes_to, thinking levels, max_out, ultra, timeout (s),
# in, out, cached, next_from, next_in, next_out, next_cached
_ROWS = [
    ("gemini-3.8-flash", "stable", None, "low,medium,high", 65536, True, 600, 0.75, 3.75, 0.075, "2027-01-01", 1.50, 7.50, 0.15),
    ("gemini-4-argon", "stable", None, "low,medium,high", 32768, True, 1200, 2.00, 10.00, 0.10, "2026-12-01", 4.00, 20.00, 0.20),
    ("gemini-3.6-flash", "stable", None, "minimal,low,medium,high", 65536, True, 600, 0.75, 3.75, 0.075, "2027-01-01", 1.50, 7.50, 0.15),
    ("gemini-3.5-flash-lite", "stable", None, "minimal,low,medium,high", 65536, True, 600, 0.30, 2.50, 0.03, None, None, None, None),
    ("gemini-3.1-flash-lite", "stable, shuts down 2027-05-07", None, "minimal,low,medium,high", 65536, True, 600, 0.25, 1.50, 0.025, None, None, None, None),
    ("gemini-3.1-pro-preview", "preview", None, "low,medium,high", 65536, True, 900, 2.00, 12.00, 0.20, None, None, None, None),
    ("gemini-3-flash-preview", "preview", None, "minimal,low,medium,high", 65536, True, 600, 0.50, 3.00, 0.05, None, None, None, None),
    ("gemini-3.7-flash", "alias", "gemini-3.8-flash", "low,medium,high", 65536, True, 600, 0.75, 3.75, 0.075, "2027-01-01", 1.50, 7.50, 0.15),
    ("gemini-3.5-flash", "alias", "gemini-3.6-flash", "minimal,low,medium,high", 65536, True, 600, 0.75, 3.75, 0.075, "2027-01-01", 1.50, 7.50, 0.15),
]

_TABLE = {r[0]: dict(zip(_COLUMNS, r)) for r in _ROWS}


class Models(pd.DataFrame):
    """The model table; prints with the date it was checked."""

    _metadata = ["checked"]

    @property
    def _constructor(self):
        return Models

    def __repr__(self) -> str:
        head = (f"Gemini models known to amanuensis (checked {getattr(self, 'checked', MODELS_CHECKED)})\n"
                "Prices: US$ per million tokens, standard rate; batch and flex cost half.\n")
        tail = ("\nDefault: gemini-3.8-flash. Harder handwriting: gemini-4-argon. "
                "Cheapest clean print: gemini-3.5-flash-lite.")
        return head + pd.DataFrame(self).to_string(index=False) + tail


def gemini_models() -> Models:
    """The Gemini models this version of the package knows, with prices.

    Other model names still work in :func:`transcribe`; the package just
    cannot check their settings or estimate their cost. Prices are US dollars
    per million tokens at the standard (one page at a time) rate, checked on
    ``Models.checked``; batch and flex runs cost half. The ``next_*`` columns
    give announced price changes and the date they take effect. When prices
    change, pass your own to ``transcribe(prices={...})``.
    """
    df = Models([dict(r) for r in _TABLE.values()], columns=_COLUMNS)
    df["max_output"] = df["max_output"].astype("Int64")
    df["timeout"] = df["timeout"].astype("Int64")
    df.checked = MODELS_CHECKED
    return df


def model_info(model: str) -> dict | None:
    """One row of the table, or None (unknown and practice models)."""
    if model in PRACTICE_MODELS:
        return None
    return _TABLE.get(model)


def canonical_model(model) -> str:
    """The model that actually answers a model name (an alias routes to another)."""
    m = str(model or "")
    if m.startswith("models/"):
        m = m[len("models/"):]
    row = _TABLE.get(m)
    if row and row["routes_to"]:
        return row["routes_to"]
    return m


def thinking_for(model: str, thinking: str) -> str:
    """Check the thinking level against what the model accepts; the API spelling ("LOW")."""
    if not isinstance(thinking, str) or thinking not in THINKING_LEVELS:
        raise InputError(f"`thinking` must be one of {', '.join(repr(t) for t in THINKING_LEVELS)}.",
                         ("x", f"You gave {thinking!r}."))
    info = model_info(model)
    if info is not None:
        allowed = info["thinking"].split(",")
        if thinking not in allowed:
            raise InputError(f"{model!r} does not accept thinking={thinking!r}.",
                             f"It accepts {', '.join(repr(a) for a in allowed)}. The default, 'low', works with every current model.")
    return thinking.upper()


def price_for(model: str, when: date | None = None, discount: bool = False, prices: dict | None = None) -> dict | None:
    """Price per million tokens on a date: announced changes, the batch/flex
    discount and a user override ``{"input":, "output":, "cached":}``."""
    if prices is not None:
        if not isinstance(prices, dict) or "input" not in prices or "output" not in prices:
            return None
        p = {"input": float(prices["input"]), "output": float(prices["output"]),
             "cached": float(prices.get("cached", prices["input"]))}
    else:
        info = model_info(model)
        if info is None:
            return None
        p = {"input": info["price_input"], "output": info["price_output"], "cached": info["price_cached"]}
        when = when or date.today()
        if info["next_from"] and when >= date.fromisoformat(info["next_from"]):
            p = {"input": info["next_input"], "output": info["next_output"], "cached": info["next_cached"]}
    if discount:
        p = {k: v / 2 for k, v in p.items()}
    return p


def cost(tokens: dict, price: dict | None) -> float | None:
    """Dollars for one reply's tokens: cached input at the cached price, thinking as output."""
    if price is None:
        return None

    def num(k):
        v = tokens.get(k)
        return 0.0 if v is None else float(v)

    inp, cached = num("input"), num("cached")
    out = num("output") + num("thinking")
    return ((inp - cached) * price["input"] + cached * price["cached"] + out * price["output"]) / 1e6


def check_model(model) -> str:
    """Say what is worth knowing about a model name; returns it without "models/"."""
    if not isinstance(model, str) or not model:
        raise InputError("`model` must be a model name, e.g. 'gemini-3.8-flash'.")
    if model in PRACTICE_MODELS:
        return model
    if model.startswith("models/"):
        model = model[len("models/"):]
    info = model_info(model)
    if info is None:
        say(("!", f"{model!r} is not in this package's model table (checked {MODELS_CHECKED})."),
            ("i", "The run will go ahead, but costs cannot be estimated. See gemini_models()."))
    elif info["status"] == "alias":
        say(("!", f"{model!r} is an old name: Google sends these requests to {info['routes_to']!r}."),
            ("i", f"Use model={info['routes_to']!r} so your records name the model that actually ran; "
                  "pages already done are not redone for this change."))
    elif "preview" in info["status"]:
        say(("!", f"{model!r} is a preview model: Google may change or withdraw it without notice."))
    return model

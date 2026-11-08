"""Local validation of each reply against the schema.

Google enforces the schema while generating, but it silently ignores keywords
it does not support (pattern, minLength, ...), and real runs gave slugs with
accents that only a local check caught, so every reply is checked again here.

A small validator for the draft-07 subset that codebook schemas use: type
(including ["string", "null"] unions), enum, const, properties, required,
additionalProperties, items, minItems/maxItems, minimum/maximum and their
exclusive forms, minLength/maxLength, pattern, anyOf/oneOf/allOf/not and local
$ref. Other keywords (format, $id, title, ...) are accepted and not checked.
It is the R package's validator, ported rather than replaced by jsonschema,
so that the problems a record stores read the same: plain words, at most 20,
and list positions counted from 1 like the ``row`` column of the results.

Two departures, both about running on a reply that was paid for:

- ``validate()`` never raises. A schema shape it cannot handle (a draft-03
  ``"required": true`` inside a property, say) becomes a problem of the reply
  (the page is "to check", and kept), never an exception that loses the reply.
- ``pattern`` is JSON Schema's, that is ECMA-262's: ``$`` matches only at the
  very end of the text (Python's also before a final line break), and a
  check that backtracks for more than PATTERN_SECONDS is stopped and reported
  instead of hanging the run; the stop needs a timer signal, so it works on
  macOS and Linux, in the main thread.
"""

from __future__ import annotations

import functools
import math
import re
import signal
import threading

PATTERN_SECONDS = 1.0

CHECKED_KEYWORDS = {
    "type", "enum", "const", "properties", "required", "additionalProperties",
    "items", "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "minLength", "maxLength", "pattern", "anyOf", "oneOf",
    "allOf", "not", "$ref", "definitions", "$defs", "description", "title",
    "$schema", "$id", "format", "propertyOrdering", "default", "examples",
    "nullable", "$comment",
}

TYPE_WORDS = {"null": "empty (null)", "boolean": "yes/no (true or false)", "integer": "a whole number",
              "number": "a number", "string": "text", "array": "a list", "object": "a set of fields"}


def json_type_of(x) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "boolean"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return "null"
        if isinstance(x, int) or (math.isfinite(x) and x == round(x)):
            return "integer"
        return "number"
    if isinstance(x, str):
        return "string"
    if isinstance(x, list):
        return "array"
    if isinstance(x, dict):
        return "object"
    return "unknown"


def _type_matches(actual: str, wanted: str) -> bool:
    return actual == wanted or (wanted == "number" and actual == "integer")


def resolve_ref(ref, root):
    """What a local JSON pointer ("#/$defs/person") points to, or None."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node = root
    for p in ref[2:].split("/"):
        p = p.replace("~0", "~").replace("~1", "/")
        if isinstance(node, dict) and p in node:
            node = node[p]
        elif isinstance(node, list) and p.isdigit() and int(p) < len(node):
            node = node[int(p)]
        else:
            return None
    return node


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return a == b and type(a) is type(b)
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=1.5e-8, abs_tol=0.0) or float(a) == float(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return type(a) is type(b) and a == b


def show_value(x) -> str:
    if x is None:
        return "empty (null)"
    if isinstance(x, list):
        return "a list"
    if isinstance(x, dict):
        return "a set of fields"
    if isinstance(x, str):
        s = x[:37] + "..." if len(x) > 40 else x
        return f'"{s}"'
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    return str(x)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


class _Collector:
    def __init__(self, maximum: int):
        self.errors: list[str] = []
        self.max = maximum

    def full(self) -> bool:
        return len(self.errors) >= self.max

    def add(self, path: str, msg: str) -> None:
        if len(self.errors) < self.max:
            self.errors.append(f"{path if path else '(whole reply)'}: {msg}")


def _with_root(sub, root):
    if not isinstance(sub, dict):
        return sub
    sub = dict(sub)
    if isinstance(root, dict):
        if "definitions" in root and "definitions" not in sub:
            sub["definitions"] = root["definitions"]
        if "$defs" in root and "$defs" not in sub:
            sub["$defs"] = root["$defs"]
    return sub


def validate(x, schema, maximum: int = 20) -> list[str]:
    """Problems of ``x`` against ``schema`` (empty when valid), at most
    ``maximum``. Never raises: what the check cannot handle is a problem."""
    c = _Collector(maximum)
    try:
        _node(x, schema, "", schema, c)
    except RecursionError:
        c.add("", "could not be checked against the schema here (it nests too deeply)")
    except (TypeError, ValueError, AttributeError, KeyError, IndexError) as e:
        c.add("", f"could not be checked against the schema here ({type(e).__name__}: {e})")
    return c.errors


# ------- pattern -------------------------------------------------------------------------

def ecma_pattern(p: str) -> str:
    """A JSON Schema pattern for Python's re: every ``$`` outside a character
    class becomes ``\\Z`` (the end of the text only, as in ECMA-262)."""
    out, i, n = [], 0, len(p)
    in_class = False
    while i < n:
        ch = p[i]
        if ch == "\\" and i + 1 < n:
            out.append(p[i:i + 2])
            i += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
            out.append(ch)
        elif ch == "[":
            in_class = True
            out.append(ch)
            if i + 1 < n and p[i + 1] == "^":
                out.append("^")
                i += 1
            if i + 1 < n and p[i + 1] == "]":  # a ] first in a class is a literal one
                out.append("]")
                i += 1
        elif ch == "$":
            out.append(r"\Z")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=256)
def _compiled(pattern: str):
    try:
        return re.compile(ecma_pattern(pattern))
    except re.error:
        return None


class _PatternTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _PatternTimeout()


def _can_time() -> bool:
    return (hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGALRM) == signal.SIG_DFL)


def pattern_matches(pattern: str, value: str):
    """True or False; True for a pattern Python cannot compile (not held
    against the reply); None when the check did not finish within
    PATTERN_SECONDS (a pattern that backtracks, such as (\\w+ ?)+$ on a long
    value ending in "?")."""
    rx = _compiled(pattern)
    if rx is None:
        return True
    if not _can_time():  # pragma: no cover - Windows, or not the main thread
        return rx.search(value) is not None
    old = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, PATTERN_SECONDS)
        try:
            return rx.search(value) is not None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except _PatternTimeout:
        return None
    finally:
        signal.signal(signal.SIGALRM, old)


def _node(x, s, path: str, root, c: _Collector) -> None:
    if c.full():
        return
    if s is True:
        return
    if s is False:
        c.add(path, "no value is allowed here")
        return
    if not isinstance(s, dict):
        return
    if "$ref" in s:
        target = resolve_ref(s["$ref"], root)
        if target is not None:
            _node(x, target, path, root, c)
    actual = json_type_of(x)

    if "type" in s and s["type"] is not None:
        wanted = s["type"] if isinstance(s["type"], list) else [s["type"]]
        wanted = [w for w in wanted if isinstance(w, str)]
        if s.get("nullable") is True and "null" not in wanted:
            wanted = wanted + ["null"]
        if wanted and not any(_type_matches(actual, w) for w in wanted):
            words = " or ".join(TYPE_WORDS[w] for w in wanted if w in TYPE_WORDS)
            c.add(path, f"expected {words}, got {show_value(x)}")
            return
    if "enum" in s and isinstance(s["enum"], list):
        if not any(values_equal(x, v) for v in s["enum"]):
            allowed = ", ".join("empty" if v is None else show_value(v).strip('"') for v in s["enum"])
            c.add(path, f"{show_value(x)} is not one of the allowed values ({allowed})")
    if "const" in s and not values_equal(x, s["const"]):
        c.add(path, f"must be exactly {show_value(s['const'])}")

    if actual == "object":
        props = s.get("properties") or {}
        req = s.get("required")
        # draft-07 "required" is a list of names; a draft-03 "required": true
        # inside a property says nothing about the fields of this object
        req = [req] if isinstance(req, str) else [r for r in req if isinstance(r, str)] if isinstance(req, list) else []
        for r in req:
            if r not in x:
                c.add(_join(path, r), "is missing")
        for nm, v in x.items():
            sub = props.get(nm) if isinstance(props, dict) else None
            if sub is not None:
                _node(v, sub, _join(path, nm), root, c)
            elif s.get("additionalProperties") is False:
                c.add(_join(path, nm), "is not a field in the codebook")
            elif isinstance(s.get("additionalProperties"), dict):
                _node(v, s["additionalProperties"], _join(path, nm), root, c)

    if actual == "array":
        n = len(x)
        if _is_number(s.get("minItems")) and n < s["minItems"]:
            c.add(path, f"has {n} item(s), needs at least {show_value(s['minItems'])}")
        if _is_number(s.get("maxItems")) and n > s["maxItems"]:
            c.add(path, f"has {n} item(s), at most {show_value(s['maxItems'])} allowed")
        items = s.get("items")
        if isinstance(items, list) and items:
            for i in range(min(n, len(items))):  # tuple form: one schema per position
                _node(x[i], items[i], f"{path}[{i + 1}]", root, c)
        elif items is not None:
            for i in range(n):
                _node(x[i], items, f"{path}[{i + 1}]", root, c)

    if actual in ("integer", "number"):
        if _is_number(s.get("minimum")) and x < s["minimum"]:
            c.add(path, f"{show_value(x)} is below the minimum {show_value(s['minimum'])}")
        if _is_number(s.get("maximum")) and x > s["maximum"]:
            c.add(path, f"{show_value(x)} is above the maximum {show_value(s['maximum'])}")
        if _is_number(s.get("exclusiveMinimum")) and x <= s["exclusiveMinimum"]:
            c.add(path, f"{show_value(x)} must be above {show_value(s['exclusiveMinimum'])}")
        if _is_number(s.get("exclusiveMaximum")) and x >= s["exclusiveMaximum"]:
            c.add(path, f"{show_value(x)} must be below {show_value(s['exclusiveMaximum'])}")

    if actual == "string":
        n = len(x)
        if _is_number(s.get("minLength")) and n < s["minLength"]:
            c.add(path, f"is shorter than {show_value(s['minLength'])} characters")
        if _is_number(s.get("maxLength")) and n > s["maxLength"]:
            c.add(path, f"is longer than {show_value(s['maxLength'])} characters")
        if isinstance(s.get("pattern"), str):
            ok = pattern_matches(s["pattern"], x)
            if ok is None:
                c.add(path, f"{show_value(x)} could not be checked against the pattern {s['pattern']} in time "
                            "(the pattern backtracks too much for the local check)")
            elif not ok:
                c.add(path, f"{show_value(x)} does not match the pattern {s['pattern']}")

    if isinstance(s.get("allOf"), list):
        for sub in s["allOf"]:
            _node(x, sub, path, root, c)
    for key in ("anyOf", "oneOf"):
        alts = s.get(key)
        if not isinstance(alts, list):
            continue
        hits = [len(validate(x, _with_root(sub, root), maximum=1)) == 0 for sub in alts]
        if not any(hits):
            c.add(path, f"{show_value(x)} matches none of the allowed forms")
        if key == "oneOf" and sum(hits) > 1:
            c.add(path, "matches more than one of the allowed forms")
    if "not" in s and isinstance(s["not"], (dict, bool)):
        if len(validate(x, _with_root(s["not"], root), maximum=1)) == 0:
            c.add(path, "has a form that is not allowed")


def unchecked_keywords(schema) -> list[str]:
    """Keywords in a user's schema that the local check does not enforce."""
    found: set[str] = set()

    def walk(node, in_props=False):
        if isinstance(node, dict):
            if not in_props:
                found.update(node.keys())
            for k, v in node.items():
                walk(v, in_props=(k in ("properties", "definitions", "$defs")) and not in_props)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return sorted(k for k in found if k not in CHECKED_KEYWORDS and k)

"""Every call to Gemini, through google-genai, with the package's own retry
loop in charge.

The SDK builds and sends the requests (generateContent, the Files API, the
Batch API); the package decides when to retry, how long to wait, and what a
failure means for the run. The SDK's own retries stay off (its default), so a
page is never retried behind the package's back, and every wait is the
package's: Google's advice when it gives one (Retry-After, RetryInfo), else 5,
10, 20, 40 s, never more than two minutes. Server trouble (408, 429, 5xx) and
network failures are retried; anything else is classified once
(``classify_failure()``): stop the run (key, account, quota, a request Google
cannot decode) or fail this page with a reason.

``generate_content`` is asked for the raw HTTP reply
(``should_return_http_response``), so a reply is read here exactly as it
arrives, the same way for sync calls and for batch result lines, and a reply
that is not JSON is kept as a failure instead of raising.

The key comes only from the ``GEMINI_API_KEY`` environment variable, set in
the terminal (``export GEMINI_API_KEY=...``). It is passed to the SDK
explicitly, so a ``GOOGLE_API_KEY`` on the computer is never used instead,
and the package never asks for a key or stores one.

Tests and the practice models answer the SDK's real HTTP requests: the SDK
gets an ``httpx.Client`` whose transport is a local function
(``httpx.MockTransport``), so request building, reply reading, retries and
the whole batch cycle run offline exactly as with Google.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re

import httpx

from ._errors import ApiError, InputError, MissingKeyError, NetworkError
from ._options import options, sleep
from ._say import info
from ._text import fmt_int, merge_dicts

# Every httpx RequestError counts as the network failing: the transport errors
# (timeouts, dropped connections) and also a reply that could not be decoded
# or a redirect loop. Anything narrower let one garbled reply in a parallel
# round stop the run as a crash.
try:  # the SDK accepts httpx2 when it is installed; its errors count too
    import httpx2  # type: ignore

    _NETWORK_EXC = (httpx.RequestError, getattr(httpx2, "RequestError", httpx2.TransportError))
except ImportError:  # pragma: no cover
    _NETWORK_EXC = (httpx.RequestError,)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/"
RESOLUTIONS = ("low", "medium", "high", "ultra_high")
TRANSIENT = (408, 429, 500, 502, 503, 504)
# Image tokens Google counts for one page at each resolution (for estimates
# made before any page is measured).
IMAGE_TOKENS = {"low": 280, "medium": 560, "high": 1120, "ultra_high": 2240}

# Tests set this to False: any attempt to reach the real Google then fails.
ALLOW_NETWORK = True
_MOCKS: list = []


# ------- the key ---------------------------------------------------------------

def key_how_to() -> str:
    """How to set GEMINI_API_KEY, for every usual terminal (Windows included)."""
    return ("Set it in the terminal you start Python (or Jupyter) from, then start Python again from that terminal. "
            "macOS and Linux: export GEMINI_API_KEY=\"your-key\". Windows PowerShell: $env:GEMINI_API_KEY = \"your-key\". "
            "Windows cmd: set GEMINI_API_KEY=your-key (without quotes). To keep it on Windows for every new terminal, and for "
            "Jupyter started from Anaconda Navigator or the Start menu: setx GEMINI_API_KEY \"your-key\", then open a new one. "
            "Get a key at https://aistudio.google.com/apikey.")


def gemini_key() -> str:
    """The key in GEMINI_API_KEY, checked for shape; the only place a key comes from."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingKeyError(
            "No Gemini API key: GEMINI_API_KEY is not set.",
            key_how_to(),
            "Never write the key in a script or a notebook.",
            "To practise without a key, use model=\"mock\" (or \"example\" with example_project()).")
    return check_key_shape(key)


_QUOTES = "\"'`"


def check_key_shape(key: str) -> str:
    """The key with surrounding spaces and quotes removed, or MissingKeyError.
    Windows cmd keeps the quotes of ``set GEMINI_API_KEY="AIza..."`` as part
    of the value; a real key never contains a quote, so a pair around it is
    taken off and a quote anywhere else is refused."""
    key = key.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in _QUOTES:
        key = key[1:-1].strip()
    if not key:
        raise MissingKeyError("GEMINI_API_KEY is empty.")
    if any(c in _QUOTES for c in key):
        raise MissingKeyError("GEMINI_API_KEY does not look like an API key: it contains a quote mark.",
                              "In Windows cmd write set GEMINI_API_KEY=your-key without quotes; elsewhere copy only the key "
                              "from https://aistudio.google.com/apikey and set it again.")
    if any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise MissingKeyError("GEMINI_API_KEY does not look like an API key: it contains a space, a line break or an unusual character.",
                              "Copy only the key from https://aistudio.google.com/apikey and export it again.")
    if len(key) < 20:
        raise MissingKeyError(f"GEMINI_API_KEY does not look like an API key: it has only {len(key)} character{'s' if len(key) != 1 else ''}.",
                              "Gemini keys usually start with AIza and are about 39 characters long; newer keys start with AQ. instead.")
    return key


def mask_key(key: str) -> str:
    return f"{key[:4]}...{key[-3:]} ({len(key)} characters)"


# ------- the client ---------------------------------------------------------------

@contextlib.contextmanager
def mocked(handler):
    """Answer every Gemini request inside the block with ``handler(httpx.Request)``.

    The practice models use their own handler; tests use this for real model
    names. Nothing inside reaches the network.
    """
    _MOCKS.append(handler)
    try:
        yield handler
    finally:
        _MOCKS.remove(handler)


def make_client(key: str, handler=None):
    """A google-genai client for the Gemini Developer API (never Vertex), with
    the SDK's retries off and, for tests and practice, a local transport."""
    from google import genai
    from google.genai import types

    handler = handler or (_MOCKS[-1] if _MOCKS else None)
    http = None
    if handler is not None:
        http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    elif not ALLOW_NETWORK:
        raise AssertionError("a test tried to reach Google")
    opts = types.HttpOptions(api_version="v1beta", base_url=options.base_url or None, httpx_client=http)
    return genai.Client(api_key=key, vertexai=False, http_options=opts)


def http_options(timeout: float | None = None, extra_body: dict | None = None, headers: dict | None = None):
    from google.genai import types

    kw = {}
    if timeout:
        kw["timeout"] = int(timeout * 1000)
    if extra_body:
        kw["extra_body"] = extra_body
    if headers:
        kw["headers"] = headers
    return types.HttpOptions(**kw) if kw else None


# ------- retries -------------------------------------------------------------------

def is_api_error(e) -> bool:
    from google.genai import errors

    return isinstance(e, errors.APIError)


def is_network_error(e) -> bool:
    return isinstance(e, _NETWORK_EXC)


def retry_wait(attempt: int, e=None) -> float:
    """Seconds before retry number ``attempt`` (from 1)."""
    advised = None
    if e is not None and is_api_error(e):
        resp = getattr(e, "response", None)
        ra = resp.headers.get("retry-after") if resp is not None and hasattr(resp, "headers") else None
        if ra and re.fullmatch(r"[0-9.]+", ra.strip()):
            advised = float(ra)
        if advised is None:
            m = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', json.dumps(getattr(e, "details", None), default=str))
            if m:
                advised = float(m.group(1))
    base = 5 * 2 ** (attempt - 1)
    wait = max(advised, 1) if advised is not None else base + random.uniform(0, 1)
    return min(wait, 120)


def call(fn, retries: int = 4, quiet: bool = False, what: str = "Google"):
    """Run one SDK call, retrying transient failures. Returns what it returns;
    raises the SDK's APIError for an HTTP error that is not transient (or
    outlived the retries), or ``NetworkError`` when the network failed every try."""
    attempt = 0
    while True:
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if is_api_error(e):
                status = getattr(e, "code", None)
                if status not in TRANSIENT or attempt >= retries:
                    raise
                attempt += 1
                wait = retry_wait(attempt, e)
                if not quiet:
                    info(f"{what} is busy or rate-limited (HTTP {status}); waiting {round(wait)} s before try {attempt + 1} of {retries + 1}.")
            elif is_network_error(e):
                if attempt >= retries:
                    raise NetworkError(f"Could not reach Google: {e}",
                                       "Check your internet connection and run the same command again; finished pages are kept.") from e
                attempt += 1
                wait = retry_wait(attempt)
                if not quiet:
                    info(f"Network problem ({e}); waiting {round(wait)} s before try {attempt + 1} of {retries + 1}.")
            else:
                raise
        sleep(wait)


# ------- what a failure means ----------------------------------------------------------

def short_message(x, n: int = 500):
    """Google's messages can quote the request back, base64 image included:
    long base64-looking runs become "[...]" and the message is cut to ``n``."""
    if x is None:
        return None
    x = " ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
    x = re.sub(r"[A-Za-z0-9+/=_-]{80,}", "[...]", x)
    if len(x) > n:
        x = x[: n - 3] + "..."
    return x


def error_message(e) -> str:
    msg = getattr(e, "message", None)
    if not msg:
        details = getattr(e, "details", None)
        if isinstance(details, dict):
            msg = details.get("status")
    return short_message(msg or "unknown error")


def error_details(e) -> dict:
    """Google's error status ("FAILED_PRECONDITION") and the quota ids of a 429."""
    out = {"reason": None, "quota": []}
    d = getattr(e, "details", None)
    err = d.get("error") if isinstance(d, dict) else None
    if not isinstance(err, dict):
        return out
    for item in err.get("details") or []:
        if not isinstance(item, dict):
            continue
        for v in item.get("violations") or []:
            if isinstance(v, dict):
                q = v.get("quotaId") or v.get("quota_id")
                if isinstance(q, str):
                    out["quota"].append(q)
    if isinstance(err.get("status"), str):
        out["reason"] = err["status"]
    return out


def classify_failure(status: int, msg: str, model=None, engine=None, details=None) -> dict:
    """What to do about a failed request: ``{"action": "stop", "message"}`` when
    the whole run cannot succeed, or ``{"action": "skip", "problem", "message"}``
    when only this page failed ("image" only when Google said so about the
    image, "server" for 5xx and 408, "request" for any other refusal)."""
    details = details or {"reason": None, "quota": []}
    low = (msg or "").lower()
    m = re.sub(r"[.\s]+$", "", msg or "")
    if status == 400 and re.search(r"api key|api_key", low):
        return {"action": "stop", "message": f"Google refused the API key ({msg}). Export GEMINI_API_KEY again with only the key, and restart Python from that terminal."}
    if status == 400 and (details.get("reason") == "FAILED_PRECONDITION" or
                          re.search(r"precondition|free tier|billing|user location|location is not supported|your country|your region", low)):
        return {"action": "stop", "message": f"Google refused the account, not the page (HTTP 400): {m}. Switch on billing for this Google project "
                                             "(or use a key from a project in a supported country) and run the same line again."}
    if status in (401, 403):
        extra = (" This feature needs a key from a project with billing switched on." if re.search(r"billing|batch", low)
                 else " Export a valid key in GEMINI_API_KEY and restart Python from that terminal.")
        return {"action": "stop", "message": f"Google refused the request (HTTP {status}): {m}.{extra}"}
    if status == 402:
        return {"action": "stop", "message": f"Google says the account has no credit left (HTTP 402): {m}. Top up the billing account and run again."}
    if status == 404:
        if model is None:
            return {"action": "stop", "message": f"Google could not find it (HTTP 404): {m}."}
        return {"action": "stop", "message": f"Google does not know the model '{model}' (HTTP 404): {m}. Check the name; gemini_models() lists current ones."}
    if status == 429:
        said = " ".join([msg or ""] + list(details.get("quota") or [])).lower()
        if re.search(r"perday|per day|per_day|daily", said):
            return {"action": "stop", "message": f"Google's daily quota for this key is used up (HTTP 429): {m}. Finished pages are kept: run the same "
                                                 "line again after midnight Pacific time (when daily quotas reset), or raise your usage tier."}
        if engine == "flex" or "flex" in low:
            return {"action": "stop", "message": f"Google's flex tier is at capacity right now (HTTP 429): {m}. Finished pages are kept: run the same "
                                                 "line again in a few minutes, or use flex=False (full price) to go on now."}
        if re.search(r"perminute|per minute|per_minute", said):
            return {"action": "stop", "message": f"Google's per-minute limit for this key was reached (HTTP 429): {m}. Finished pages are kept: wait a "
                                                 "minute or two and run the same line again (with fewer workers if you used several)."}
        return {"action": "stop", "message": f"Google's rate limit or quota for this key was reached (HTTP 429): {m}. Finished pages are kept: wait a "
                                             "minute or two and run the same line again (with fewer workers if you used several). If it happens again, "
                                             "the daily quota may be used up: daily quotas reset at midnight Pacific time, or raise your usage tier."}
    if status == 400 and "base64 decoding failed" in low:
        return {"action": "stop", "message": f"Google could not read the image data amanuensis sent ({msg}). This is a problem in the package, "
                                             "not in your pages: update amanuensis, or report it with this message."}
    if status == 400 and re.search(r"cannot find field|unknown name", low) and "schema" not in low:
        return {"action": "stop", "message": f"Google did not recognise a field in the request ({msg}). If you used api_args, check them; "
                                             "otherwise Google may have changed its API: update amanuensis."}
    if status == 413 or (status == 400 and re.search(r"image|inline_data|inlinedata|media", low)):
        return {"action": "skip", "problem": "image", "message": f"Google could not use this image (HTTP {status}): {msg}"}
    if status == 400 and "thinking" in low:
        return {"action": "stop", "message": f"Google rejected the thinking setting ({msg}). Use thinking=\"low\"."}
    if status == 400 and re.search(r"schema|response_json|responsejson", low):
        return {"action": "stop", "message": f"Google rejected the codebook ({msg}). Simplify the codebook, or check the JSON schema if you wrote one."}
    if status >= 500 or status == 408:
        return {"action": "skip", "problem": "server", "message": f"Google had a server problem (HTTP {status}): {m}. The page will be tried again next run."}
    return {"action": "skip", "problem": "request", "message": f"Google returned an error (HTTP {status}): {msg}"}


def api_error(what: str, e, model=None) -> ApiError:
    """An ApiError for a failed Files/Batch call, with the HTTP status."""
    status = getattr(e, "code", None)
    why = classify_failure(status or 0, error_message(e), model=model, details=error_details(e))
    return ApiError(f"{what}.", ("x", why["message"]), status=status)


# ------- request bodies --------------------------------------------------------------------

def media_level(resolution: str) -> str:
    return f"MEDIA_RESOLUTION_{resolution.upper()}"


def mime_of(path) -> str:
    e = os.path.splitext(str(path))[1].lower().lstrip(".")
    mimes = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
             "heic": "image/heic", "heif": "image/heif"}
    if e not in mimes:
        raise InputError(f"Cannot send {os.path.basename(str(path))} to Gemini: it accepts JPEG, PNG, WEBP and HEIC. Run images() first.")
    return mimes[e]


def schema_for_api(schema):
    """The schema as sent: the user's schema minus "$schema" (not a keyword Gemini supports)."""
    if schema is None:
        return None
    return {k: v for k, v in schema.items() if k != "$schema"}


def generate_request(prompt: str, image: bytes, mime: str, schema, settings: dict, headers: dict | None = None):
    """(contents, config) for ``client.models.generate_content``: the image part
    first, then the instructions (Google's advice), no system instruction,
    schema-constrained JSON. ultra_high exists only per image part; the other
    levels go in the config. api_args are merged into the REST body."""
    from google.genai import types

    res = settings["resolution"]
    part_kw = {"inline_data": types.Blob(data=image, mime_type=mime)}
    if res == "ultra_high":
        part_kw["media_resolution"] = types.PartMediaResolution(level=media_level(res))
    part = types.Part(**part_kw)
    kw = dict(response_mime_type="application/json",
              max_output_tokens=int(settings["max_tokens"]),
              thinking_config=types.ThinkingConfig(thinking_level=settings["thinking_level"], include_thoughts=False),
              should_return_http_response=True,
              http_options=http_options(settings.get("timeout"), settings.get("api_args") or None, headers))
    if schema is not None:
        if settings.get("schema_channel") == "openapi":
            kw["response_schema"] = to_response_schema(schema)
        else:
            kw["response_json_schema"] = schema
    if res != "ultra_high":
        kw["media_resolution"] = media_level(res)
    if settings.get("temperature") is not None:
        kw["temperature"] = float(settings["temperature"])
    if settings.get("flex"):
        kw["service_tier"] = "flex"
    contents = [types.Content(role="user", parts=[part, types.Part(text=prompt)])]
    return contents, types.GenerateContentConfig(**kw)


def batch_body(prompt: str, image_b64: str, mime: str, schema, settings: dict) -> dict:
    """The request of one batch JSONL line, in the REST field names (snake_case)
    the R package writes."""
    res = settings["resolution"]
    image_part = {"inline_data": {"mime_type": mime, "data": image_b64}}
    if res == "ultra_high":
        image_part["media_resolution"] = {"level": media_level(res)}
    config = {"response_mime_type": "application/json"}
    if schema is not None:
        if settings.get("schema_channel") == "openapi":
            config["response_schema"] = to_response_schema(schema)
        else:
            config["response_json_schema"] = schema
    config["max_output_tokens"] = int(settings["max_tokens"])
    config["thinking_config"] = {"thinking_level": settings["thinking_level"], "include_thoughts": False}
    if res != "ultra_high":
        config["media_resolution"] = media_level(res)
    if settings.get("temperature") is not None:
        config["temperature"] = settings["temperature"]
    body = {"contents": [{"role": "user", "parts": [image_part, {"text": prompt}]}], "generation_config": config}
    api_args = settings.get("api_args") or {}
    if api_args:
        body = merge_dicts(body, snake_keys(api_args))
    return body


_VERBATIM = {"response_json_schema", "response_schema", "parameters_json_schema", "properties", "definitions", "$defs"}


def snake_keys(x):
    """Field names in snake_case for batch lines; a schema inside api_args is
    the user's data, not request fields, so the walk stops at it."""
    if isinstance(x, list):
        return [snake_keys(v) for v in x]
    if not isinstance(x, dict):
        return x
    out = {}
    for k, v in x.items():
        nk = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(k)).lower()
        out[nk] = v if nk in _VERBATIM else snake_keys(v)
    return out


# ------- the OpenAPI schema channel --------------------------------------------------------------

_OPENAPI_FORMATS = {"STRING": ("enum", "date-time"), "INTEGER": ("int32", "int64"), "NUMBER": ("float", "double")}


def _types(node) -> list:
    t = node.get("type")
    return t if isinstance(t, list) else ([] if t is None else [t])


def deref(node, root):
    from ._validate import resolve_ref

    for _ in range(20):
        if not isinstance(node, dict) or "$ref" not in node:
            break
        target = resolve_ref(node["$ref"], root)
        if target is None:
            break
        node = target
    return node


def is_null_schema(node) -> bool:
    if not isinstance(node, dict):
        return False
    if _types(node) == ["null"]:
        return True
    enum = node.get("enum")
    return isinstance(enum, list) and len(enum) > 0 and all(v is None for v in enum)


def merge_all_of(node, root):
    """allOf branches merged into the node: properties and required joined,
    any other keyword taken from the last branch that has it."""
    out = {k: v for k, v in node.items() if k != "allOf"}
    for b in node.get("allOf") or []:
        b = deref(b, root)
        if not isinstance(b, dict):
            continue
        for nm, val in b.items():
            if nm == "required":
                req = list(out.get("required") or [])
                for r in val or []:
                    if r not in req:
                        req.append(r)
                out["required"] = req
            elif nm == "properties":
                props = dict(out.get("properties") or {})
                props.update(val or {})
                out["properties"] = props
            elif val is not None:
                out[nm] = val
    return out


def _openapi_type_of(vals) -> str:
    if vals and all(isinstance(v, bool) for v in vals):
        return "BOOLEAN"
    if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return "INTEGER" if all(float(v) == int(v) for v in vals) else "NUMBER"
    return "STRING"


def _openapi_node(node, root):
    from ._validate import resolve_ref

    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target = resolve_ref(node["$ref"], root)
        if target is not None:
            return _openapi_node(target, root)
    if "allOf" in node:
        return _openapi_node(merge_all_of(node, root), root)
    alts = node.get("anyOf") if node.get("anyOf") is not None else node.get("oneOf")
    if alts is not None:
        alts = [deref(a, root) for a in alts]
        nulls = [is_null_schema(a) for a in alts]
        rest = [a for a, n in zip(alts, nulls) if not n]
        parent = {k: v for k, v in node.items() if k not in ("anyOf", "oneOf")}
        if len(rest) == 1:
            lone = dict(parent)
            for k, v in rest[0].items():
                if k not in parent:
                    lone[k] = v
            out = _openapi_node(lone, root)
        else:
            out = _openapi_node(parent, root)
            if rest:
                out["anyOf"] = [to_response_schema(a, root) for a in rest]
        if any(nulls):
            out["nullable"] = True
        return out
    out = {}
    if node.get("description") is not None:
        out["description"] = node["description"]
    enum = node.get("enum")
    if enum is None and node.get("const") is not None:
        enum = [node["const"]]
    ty = _types(node)
    if ty:
        base = [t for t in ty if t != "null"]
        out["type"] = (base[0] if base else "string").upper()
        if "null" in ty:
            out["nullable"] = True
    if enum is not None:
        vals = [v for v in enum if v is not None]
        if len(vals) < len(enum):
            out["nullable"] = True
        if "type" not in out:
            out["type"] = _openapi_type_of(vals)
        if out["type"] == "STRING":
            out["enum"] = [str(v) for v in vals]
    fmt = node.get("format")
    if isinstance(fmt, str) and out.get("type") in _OPENAPI_FORMATS and fmt in _OPENAPI_FORMATS[out["type"]]:
        out["format"] = fmt
    if node.get("properties") is not None:
        props = {k: to_response_schema(v, root) for k, v in node["properties"].items()}
        out["properties"] = props
        out["propertyOrdering"] = list(node["properties"].keys())
    if node.get("required") is not None:
        req = node["required"]
        out["required"] = list(req) if isinstance(req, list) else [req]
    if node.get("items") is not None:
        out["items"] = to_response_schema(node["items"], root)
    for kw in ("minItems", "maxItems", "minimum", "maximum"):
        if node.get(kw) is not None:
            out[kw] = node[kw]
    return out


def to_response_schema(node, root=None):
    """The older OpenAPI-style schema ("responseSchema"): one upper-case type
    plus "nullable" instead of type arrays; enum on strings only (minus null);
    propertyOrdering added (OpenAPI schemas otherwise come back in
    alphabetical order); oneOf read as anyOf, allOf merged, const as a
    one-value enum. Every node gets a type: a node with nothing OpenAPI can
    say became [], which Google cannot parse. pattern, additionalProperties
    and $ref are dropped; the local check still enforces them."""
    root = node if root is None else root
    out = _openapi_node(node, root)
    if isinstance(out, dict) and "type" not in out and "anyOf" not in out:
        out["type"] = "OBJECT" if "properties" in out else "ARRAY" if "items" in out else "STRING"
    return out


# ------- replies ---------------------------------------------------------------------------

def _pick(d: dict, camel: str, snake: str):
    if not isinstance(d, dict):
        return None
    v = d.get(camel)
    return d.get(snake) if v is None else v


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def read_reply(reply: dict, raw_body: str | None = None) -> dict:
    """What is kept of a generateContent reply (camelCase or snake_case): the
    text (non-thought parts joined, as the SDKs do), finish and block
    reasons, model version and token counts."""
    usage = _pick(reply, "usageMetadata", "usage_metadata") or {}
    thinking_raw = _pick(usage, "thoughtsTokenCount", "thoughts_token_count")
    cached = _pick(usage, "cachedContentTokenCount", "cached_content_token_count")
    tokens = {
        "input": _num(_pick(usage, "promptTokenCount", "prompt_token_count")),
        "cached": _num(cached) if cached is not None else 0.0,
        "thinking": _num(thinking_raw) if thinking_raw is not None else 0.0,
        "thinking_raw": _num(thinking_raw),
        "output": _num(_pick(usage, "candidatesTokenCount", "candidates_token_count")),
        "total": _num(_pick(usage, "totalTokenCount", "total_token_count")),
    }
    image_tokens = None
    for d in _pick(usage, "promptTokensDetails", "prompt_tokens_details") or []:
        if isinstance(d, dict) and d.get("modality") == "IMAGE":
            image_tokens = _num(_pick(d, "tokenCount", "token_count"))
    out = {"text": None, "finish_reason": None, "block_reason": None,
           "model_version": _pick(reply, "modelVersion", "model_version"), "tokens": tokens,
           "image_tokens": image_tokens, "nul": raw_body is not None and "\\u0000" in raw_body}
    cands = reply.get("candidates") or []
    if not cands:
        fb = _pick(reply, "promptFeedback", "prompt_feedback") or {}
        out["block_reason"] = _pick(fb, "blockReason", "block_reason") or "no reply"
        return out
    c1 = cands[0] or {}
    out["finish_reason"] = _pick(c1, "finishReason", "finish_reason")
    parts = ((c1.get("content") or {}).get("parts")) or []
    out["text"] = "".join("" if (p.get("thought") is True or p.get("text") is None) else str(p.get("text")) for p in parts if isinstance(p, dict))
    return out


def finish_message(reason, max_tokens: int) -> str:
    if reason == "MAX_TOKENS":
        return (f"The reply was cut off at max_tokens = {fmt_int(max_tokens)}. The page is very dense, or the model got stuck "
                f"repeating a character. Run again with a larger max_tokens (for example {fmt_int(min(2 * max_tokens, 65536))}).")
    if reason == "CONTINUATION":
        return f"The reply was cut off at max_tokens = {fmt_int(max_tokens)}. Run again with a larger max_tokens."
    if reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "IMAGE_SAFETY", "SPII"):
        return f"Google's content filter blocked this page ({reason}). Try again later; if it repeats, transcribe this page by hand."
    if reason == "RECITATION":
        return ("Google blocked the reply because it resembles published text it was trained on (RECITATION). This can happen "
                "with famous printed works; try again, or transcribe this page by hand.")
    if reason is None:
        return "Google sent an incomplete reply (no finish reason)."
    return f"The reply ended unexpectedly ({reason})."


# ------- Files and Batch ------------------------------------------------------------------------

def upload(client, path, display_name: str, retries: int = 4) -> str:
    from google.genai import types

    cfg = types.UploadFileConfig(display_name=display_name, mime_type="application/jsonl",
                                 http_options=types.HttpOptions(timeout=3600 * 1000))
    try:
        f = call(lambda: client.files.upload(file=str(path), config=cfg), retries=retries)
    except Exception as e:
        if is_api_error(e):
            raise api_error("Uploading the batch file failed", e) from None
        raise
    return f.name


def batch_create(client, model: str, file_name: str, display_name: str):
    """Create a job. Never retried: creating a job is not idempotent."""
    from google.genai import types

    cfg = types.CreateBatchJobConfig(display_name=display_name, http_options=types.HttpOptions(timeout=300 * 1000))
    try:
        return call(lambda: client.batches.create(model=model, src=file_name, config=cfg), retries=0)
    except Exception as e:
        if is_api_error(e):
            raise api_error("Creating the batch job failed", e, model=model) from None
        raise


def batch_get(client, name: str, retries: int = 4):
    try:
        return call(lambda: client.batches.get(name=name), retries=retries, quiet=True)
    except Exception as e:
        if is_api_error(e):
            raise api_error("Checking the batch job failed", e) from None
        raise


def batch_state(job) -> str:
    state = getattr(job, "state", None)
    s = getattr(state, "value", state) or "UNSPECIFIED"
    return re.sub(r"^(BATCH|JOB)_STATE_", "", str(s))


def batch_results_file(job):
    dest = getattr(job, "dest", None)
    return getattr(dest, "file_name", None) if dest is not None else None


def find_job_by_display(client, display: str, limit: int = 2000):
    """The job name for a display name; None only when the complete list of
    the account's jobs does not hold it. Raises when the list could not be
    read in full: "could not ask" must never be taken for "not there"."""
    from google.genai import types

    try:
        pager = call(lambda: client.batches.list(config=types.ListBatchJobsConfig(page_size=100)), retries=2, quiet=True)
        n = 0
        for job in pager:
            n += 1
            if getattr(job, "display_name", None) == display:
                return job.name
            if n >= limit:
                raise ApiError(f"Google lists more than {limit:,} batch jobs for this key; amanuensis did not look through them all.")
    except Exception as e:
        if is_api_error(e):
            raise api_error("Listing the batch jobs failed", e) from None
        if is_network_error(e):
            raise NetworkError(f"Could not reach Google to list the batch jobs: {e}") from e
        raise
    return None


def batch_cancel(client, name: str):
    try:
        call(lambda: client.batches.cancel(name=name), retries=2)
        return True, None
    except Exception as e:
        if is_api_error(e):
            return False, error_message(e)
        raise


def download(client, file_name: str, retries: int = 4) -> str:
    """The results file, read as UTF-8 whatever the content type says."""
    from google.genai import types

    try:
        data = call(lambda: client.files.download(file=file_name, config=types.DownloadFileConfig(
            http_options=types.HttpOptions(timeout=3600 * 1000))), retries=retries)
    except Exception as e:
        if is_api_error(e):
            raise api_error("Downloading the batch results failed", e) from None
        raise
    return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)


def file_delete(client, name: str) -> bool:
    try:
        call(lambda: client.files.delete(name=name), retries=1, quiet=True)
        return True
    except Exception as e:
        if is_api_error(e) or isinstance(e, NetworkError) or is_network_error(e):
            return False
        raise


def check_key_online(key: str):
    """True (accepted), False (refused) or None (could not check). Free: it
    asks for the list of models."""
    client = make_client(key)
    try:
        call(lambda: next(iter(client.models.list(config={"page_size": 1})), None), retries=1, quiet=True)
        return True, None
    except Exception as e:
        if is_api_error(e):
            return False, f"HTTP {getattr(e, 'code', '?')}: {error_message(e)}"
        if isinstance(e, NetworkError) or is_network_error(e):
            return None, str(e)
        raise

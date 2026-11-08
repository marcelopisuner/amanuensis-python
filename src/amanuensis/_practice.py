"""The practice models: ``model="mock"`` invents schema-shaped replies for any
page; ``model="example"`` replays prepared replies for the example project's
pages. Neither needs a key or costs anything.

They are not a separate code path: they answer the SDK's real HTTP requests
through a local transport (``_gemini.make_client(handler=...)``), so request
building, reply reading, validation, records, resuming and the whole batch
cycle (upload, create, poll, download) run exactly as with Google. The tests
use the same mechanism with handlers of their own.
"""

from __future__ import annotations

import json
import re
from importlib import resources

import httpx

from ._errors import InputError
from ._gemini import deref, is_null_schema, merge_all_of
from ._text import to_json
from ._validate import resolve_ref

_STATE: dict = {"upload": 0, "job": 0, "files": {}, "jobs": {}, "example": None}


def reset() -> None:
    """Forget the practice server's files and jobs (for tests)."""
    _STATE.update({"upload": 0, "job": 0, "files": {}, "jobs": {}})


def handler(model: str):
    """The transport function answering a practice model's requests."""

    def respond(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        path = re.sub(r"^https?://[^/]+/", "", url.split("?")[0])
        method = req.method
        if path.endswith(":generateContent"):
            body = json.loads(req.content or b"{}")
            file = req.headers.get("x-amanuensis-file", "page")
            return httpx.Response(200, json=practice_reply(model, body, file))
        if path.endswith("upload/v1beta/files") and req.headers.get("x-goog-upload-command", "").startswith("start"):
            _STATE["upload"] += 1
            n = _STATE["upload"]
            return httpx.Response(200, headers={"x-goog-upload-url": f"https://mock.amanuensis/upload/{n}"}, json={})
        if url.startswith("https://mock.amanuensis/upload/"):
            n = url.rsplit("/", 1)[-1]
            name = f"files/mock-{n}"
            _STATE["files"][name] = _STATE["files"].get(name, "") + req.content.decode("utf-8")
            return httpx.Response(200, headers={"x-goog-upload-status": "final"},
                                  json={"file": {"name": name, "mimeType": "application/jsonl"}})
        if path.endswith(":batchGenerateContent"):
            body = json.loads(req.content or b"{}")
            _STATE["job"] += 1
            name = f"batches/mock-{_STATE['job']}"
            batch = body.get("batch") or {}
            _STATE["jobs"][name] = {"file": (batch.get("inputConfig") or {}).get("fileName"), "polls": 0,
                                    "display": batch.get("displayName"), "state": "PENDING"}
            return httpx.Response(200, json={"name": name, "metadata": {"state": "BATCH_STATE_PENDING",
                                                                       "displayName": batch.get("displayName")}})
        m = re.search(r"(batches/mock-[0-9]+):cancel$", path)
        if m:
            job = _STATE["jobs"].get(m.group(1))
            if job is not None:
                job["state"] = "CANCELLED"
            return httpx.Response(200, json={})
        m = re.search(r"(batches/mock-[0-9]+)$", path)
        if m and method == "GET":
            name = m.group(1)
            job = _STATE["jobs"].get(name)
            if job is None:
                return httpx.Response(404, json={"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
            if job["state"] in ("PENDING", "RUNNING"):
                job["polls"] += 1
                job["state"] = "SUCCEEDED" if job["polls"] >= 2 else "RUNNING"
            meta = {"state": f"BATCH_STATE_{job['state']}", "displayName": job["display"]}
            if job["state"] == "SUCCEEDED":
                meta["output"] = {"responsesFile": f"{job['file']}-out"}
            return httpx.Response(200, json={"name": name, "metadata": meta})
        if path.endswith("v1beta/batches") and method == "GET":
            ops = [{"name": nm, "metadata": {"displayName": j["display"], "state": f"BATCH_STATE_{j['state']}"}}
                   for nm, j in _STATE["jobs"].items()]
            return httpx.Response(200, json={"operations": ops})
        if path.endswith(":download"):
            m = re.search(r"(files/mock-[0-9]+)-out:download$", path)
            text = _STATE["files"].get(m.group(1), "") if m else ""
            out = []
            for line in text.split("\n"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                out.append(to_json({"key": obj.get("key"), "response": practice_reply(model, obj.get("request") or {}, obj.get("key"))}))
            return httpx.Response(200, headers={"content-type": "application/octet-stream"},
                                  content="\n".join(out).encode("utf-8"))
        if method == "DELETE":
            return httpx.Response(200, json={})
        if path.endswith("v1beta/models"):
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404, json={"error": {"code": 404, "message": f"mock: unknown endpoint {path}", "status": "NOT_FOUND"}})

    return respond


def _schema_of(body: dict):
    gc = body.get("generationConfig") or body.get("generation_config") or {}
    for k in ("responseJsonSchema", "response_json_schema", "responseSchema", "response_schema"):
        if gc.get(k) is not None:
            return gc[k]
    return None


def practice_reply(model: str, body: dict, file) -> dict:
    """A generateContent reply for a request body (camelCase or snake_case)."""
    schema = _schema_of(body)
    prompt = ""
    contents = body.get("contents") or [{}]
    for p in (contents[0] or {}).get("parts") or []:
        if isinstance(p, dict) and p.get("text") is not None:
            prompt = p["text"]
    if model == "example":
        ex = example_reply(file)
        text = to_json(ex["data"])
        tokens = {"prompt": int(ex["tokens"]["prompt"]), "output": int(ex["tokens"]["output"])}
    else:
        text = to_json(mock_value(schema if schema is not None else {"type": "object"}, "value", 1, file, schema))
        tokens = {"prompt": 1300 + int(len(prompt) / 4), "output": int(len(text) / 4) + 1}
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": tokens["prompt"], "candidatesTokenCount": tokens["output"],
                          "totalTokenCount": tokens["prompt"] + tokens["output"],
                          "promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 2240},
                                                  {"modality": "TEXT", "tokenCount": tokens["prompt"] - 2240}]},
        "modelVersion": model,
    }


def mock_value(s, name: str, i: int, file, root):
    """Made-up data that fits a schema (JSON-schema or OpenAPI style)."""
    if not isinstance(s, dict):
        return f"{name} {i}"
    if "$ref" in s:
        s = resolve_ref(s["$ref"], root) or s
    if "allOf" in s:
        s = merge_all_of(s, root)
    if is_null_schema(s):
        return None
    alts = s.get("anyOf") if s.get("anyOf") is not None else s.get("oneOf")
    if alts is not None:
        alts = [deref(a, root) for a in alts]
        keep = [a for a in alts if not is_null_schema(a)]
        return mock_value(keep[0] if keep else alts[0], name, i, file, root)
    if s.get("const") is not None:
        return s["const"]
    if isinstance(s.get("enum"), list):
        vals = [v for v in s["enum"] if v is not None]
        return vals[(i - 1) % len(vals)] if vals else None
    t = s.get("type")
    types = [x.lower() for x in (t if isinstance(t, list) else [t] if t else []) if isinstance(x, str) and x.lower() != "null"]
    if not types:
        types = ["object"] if s.get("properties") is not None else ["array"] if s.get("items") is not None else ["string"]
    ty = types[0]
    if ty == "object":
        return {nm: mock_value(sub, nm, i, file, root) for nm, sub in (s.get("properties") or {}).items()}
    if ty == "array":
        it = s.get("items") or {"type": "string"}
        n = 3 if isinstance(it, dict) and it.get("properties") is not None else 2
        if isinstance(s.get("minItems"), int):
            n = max(n, s["minItems"])
        return [mock_value(it, name, j, file, root) for j in range(1, n + 1)]
    if ty == "integer":
        return int(i)
    if ty == "number":
        return i + 0.5
    if ty == "boolean":
        return i % 2 == 1
    if name == "text":
        return (f"Practice transcription of {file}.\n"
                "The mock model invents this text; use a real model to read the page.")
    return f"{name} {i}"


# ------- the example model ----------------------------------------------------------

def example_replies() -> dict:
    if _STATE["example"] is None:
        ref = resources.files("amanuensis").joinpath("data/example/replies.json")
        _STATE["example"] = json.loads(ref.read_text(encoding="utf-8"))
    return _STATE["example"]


def example_reply(file) -> dict:
    rep = example_replies().get(file)
    if rep is None:
        raise InputError("The 'example' model only knows the pages of the example project.",
                         "Use a real model for your own pages, or 'mock' to practise.")
    return rep


def check_example_pages(pages) -> None:
    known = set(example_replies())
    bad = [f for f in pages["file"] if f not in known]
    if bad:
        raise InputError(f"The 'example' model only knows the pages of the example project; it cannot read {', '.join(bad[:3])}.",
                         "Use a real model for your own pages, or model='mock' to practise.")

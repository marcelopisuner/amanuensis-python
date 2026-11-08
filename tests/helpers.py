"""Shared set-up for the offline tests.

Every test runs in a temporary folder of its own, with options that make the
run non-interactive and instant (no questions, no sleeping, no polling
delay), a fake key in GEMINI_API_KEY (GOOGLE_API_KEY removed, so a key on
the test machine can never be used), and a guard that fails any attempt to
reach Google. Gemini is faked at the HTTP level under google-genai
(``httpx.MockTransport``), so the SDK builds and sends its real requests.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

import httpx
import pandas as pd

import amanuensis as am
from amanuensis import _gemini, _practice, _store

FAKE_KEY = "AIzaSyFAKE-test-key-0123456789abcdefgh"
_EXAMPLE_DIR = []


def example_file(name: str) -> str:
    """A copy of one of the example project's files, found through the package
    (so the tests run against an installed copy or a wheel as well)."""
    if not _EXAMPLE_DIR:
        folder = tempfile.mkdtemp(prefix="amanuensis-example-")
        with contextlib.redirect_stderr(io.StringIO()):
            am.example_project(folder)
        _EXAMPLE_DIR.append(folder)
    return os.path.join(_EXAMPLE_DIR[0], name)
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


class Case(unittest.TestCase):
    """A test in its own temporary working folder, offline and silent."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.dir = tempfile.mkdtemp(prefix="amanuensis-test-")
        os.chdir(self.dir)
        self._env = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        os.environ["GEMINI_API_KEY"] = FAKE_KEY
        os.environ.pop("GOOGLE_API_KEY", None)
        self._opts = am.options.override(ask=False, interactive=False, sleep=lambda s: None, poll_seconds=0, quiet=False)
        self._opts.__enter__()
        _gemini.ALLOW_NETWORK = False
        _practice.reset()
        _store.session_reset()
        self.stderr = io.StringIO()
        self._err = contextlib.redirect_stderr(self.stderr)
        self._err.__enter__()
        self.stdout = io.StringIO()
        self._out = contextlib.redirect_stdout(self.stdout)
        self._out.__enter__()

    def tearDown(self):
        self._out.__exit__(None, None, None)
        self._err.__exit__(None, None, None)
        self._opts.__exit__(None, None, None)
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.chdir(self._cwd)
        shutil.rmtree(self.dir, ignore_errors=True)

    def messages(self) -> str:
        return self.stderr.getvalue()

    def clear_messages(self):
        self.stderr.seek(0)
        self.stderr.truncate()

    def path(self, *parts) -> str:
        return os.path.join(self.dir, *parts)

    @contextlib.contextmanager
    def warns(self, pattern=None):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield w
        texts = [str(x.message) for x in w if issubclass(x.category, am.AmanuensisWarning)]
        if pattern is not None:
            self.assertTrue(any(pattern in t for t in texts), f"no warning with {pattern!r} in {texts}")

    @contextlib.contextmanager
    def no_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield w
        texts = [str(x.message) for x in w if issubclass(x.category, am.AmanuensisWarning)]
        self.assertEqual(texts, [])


# ------- images ----------------------------------------------------------------------

def make_jpegs(folder, n=3, prefix="p", width=60, height=40):
    """n small JPEGs of distinct colours (so their bytes differ), p001.jpg ..."""
    from PIL import Image

    os.makedirs(folder, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        img = Image.new("RGB", (width, height), ((37 * i) % 256, (91 * i) % 256, (151 * i) % 256))
        p = os.path.join(folder, f"{prefix}{i:03d}.jpg")
        img.save(p, "JPEG", quality=90)
        paths.append(p)
    return paths


def make_pdf(path, n=3):
    """A scanned-looking PDF of n pages (images, as a scanner makes them)."""
    from PIL import Image, ImageDraw

    pages = []
    for p in range(1, n + 1):
        img = Image.new("L", (300, 400), 255)
        d = ImageDraw.Draw(img)
        d.text((40, 40), f"Page {p}", fill=0)
        for i in range(1, 6):
            d.text((20, 60 + 40 * i), f"{i} Puig, Josep {20 + i}", fill=0)
        pages.append(img)
    pages[0].save(path, "PDF", resolution=100, save_all=True, append_images=pages[1:])
    return path


# ------- codebooks ------------------------------------------------------------------------

def roster_codebook():
    with contextlib.redirect_stderr(io.StringIO()):
        return am.codebook(pd.DataFrame({
            "variable": ["heading", "name", "age", "sex"],
            "type": ["text", "text", "integer", "category"],
            "level": ["page", "row", "row", "row"],
            "values": ["", "", "", "male; female"],
            "description": ["Page heading.", "Name as written.", "Age in years.", "Sex as recorded."],
        }))


def roster_data(file="page", n=2):
    return {"heading": f"Heading of {file}",
            "records": [{"name": f"Name {i}", "age": 20 + i, "sex": "male"} for i in range(1, n + 1)]}


def test_instructions(text="These are pages of a 1930 municipal register."):
    with contextlib.redirect_stderr(io.StringIO()):
        return am.instructions(text=text)


# ------- fake Gemini replies --------------------------------------------------------------------

def gemini_reply(data=None, text=None, finish="STOP", prompt=2500, output=200, thoughts=None, total=None,
                 image=2240, cached=None, model_version="gemini-3.8-flash", thought_part=False,
                 block_reason=None, no_candidates=False):
    """A generateContent reply body as Google sends it."""
    if text is None:
        text = json.dumps({"text": "Some text."} if data is None else data, ensure_ascii=False)
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": output}
    if cached is not None:
        usage["cachedContentTokenCount"] = cached
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    usage["totalTokenCount"] = total if total is not None else prompt + output + (thoughts or 0)
    if image is not None:
        usage["promptTokensDetails"] = [{"modality": "TEXT", "tokenCount": prompt - image}, {"modality": "IMAGE", "tokenCount": image}]
    body = {"usageMetadata": usage, "modelVersion": model_version, "responseId": "resp-1"}
    if no_candidates:
        body["promptFeedback"] = {"blockReason": block_reason or "OTHER"}
        return body
    parts = [{"text": text}]
    if thought_part:
        parts = [{"text": "Let me think about the page.", "thought": True}] + parts
    body["candidates"] = [{"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}]
    return body


STATUS_TEXT = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
               429: "RESOURCE_EXHAUSTED", 500: "INTERNAL", 503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED"}


def error_response(status, message="Something went wrong.", status_text=None, retry_delay=None, headers=None, quota=None):
    err = {"code": status, "message": message, "status": status_text or STATUS_TEXT.get(status, "UNKNOWN")}
    details = []
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    if quota is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota}]})
    if details:
        err["details"] = details
    return httpx.Response(status, json={"error": err}, headers=headers or {})


def decode_b64(s: str) -> bytes:
    s = s.strip()
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.replace("+", "-").replace("/", "_"))


class FakeGemini:
    """A fake generateContent endpoint that recognises the page of each request
    from its image bytes. ``handler(file, body, n, request)`` returns an
    httpx.Response, a reply body (sent with status 200), or None for a
    schema-shaped default reply; ``n`` counts the calls made for that file."""

    def __init__(self, images=None, handler=None):
        self.images = images
        self.handler = handler
        self.calls = []
        self.count = {}

    def files_now(self):
        imgs = self.images
        if imgs is None:
            return []
        if isinstance(imgs, (str, os.PathLike)) and os.path.isdir(imgs):
            return [os.path.join(imgs, f) for f in sorted(os.listdir(imgs)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        return list(imgs)

    def __call__(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if ":generateContent" not in url:
            return httpx.Response(404, json={"error": {"code": 404, "message": f"unexpected request {url}", "status": "NOT_FOUND"}})
        body = json.loads(req.content or b"{}")
        b64 = None
        for p in (body.get("contents") or [{}])[0].get("parts") or []:
            if p.get("inlineData"):
                b64 = p["inlineData"].get("data")
        sent = decode_b64(b64) if b64 else b""
        file = "unknown"
        for f in self.files_now():
            with open(f, "rb") as fh:
                if fh.read() == sent:
                    file = os.path.basename(f)
                    break
        n = self.count.get(file, 0) + 1
        self.count[file] = n
        self.calls.append({"file": file, "body": body, "url": url, "headers": dict(req.headers)})
        out = self.handler(file, body, n, req) if self.handler else None
        if isinstance(out, httpx.Response):
            return out
        if isinstance(out, BaseException):
            raise out
        if out is None:
            schema = (body.get("generationConfig") or {}).get("responseJsonSchema") or {"type": "object"}
            out = gemini_reply(data=_practice.mock_value(schema, "value", 1, file, schema))
        return httpx.Response(200, json=out)

    def files_called(self):
        return [c["file"] for c in self.calls]


class FakeBatch:
    """A fake Batch API for a real model name: upload, create, poll, list,
    download, cancel, delete. ``states``: the states returned by successive
    polls (the last repeats). ``reply(key, request)`` returns a results line
    (dict with key and response, or key and error); default: schema-shaped."""

    def __init__(self, states=("SUCCEEDED",), reply=None, shuffle=False, create_error=None, list_error=None):
        self.states = list(states)
        self.reply = reply
        self.shuffle = shuffle
        self.create_error = create_error
        self.list_error = list_error
        self.uploads = {}
        self.calls = []
        self.polls = 0
        self.deleted = []
        self.created = []
        self.cancelled = []
        self.display = None
        self.input = None
        self.chunks = {}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url).split("?")[0]
        self.calls.append(f"{req.method} {url}")
        if url.endswith("/upload/v1beta/files") and req.headers.get("x-goog-upload-command", "").startswith("start"):
            return httpx.Response(200, headers={"x-goog-upload-url": "https://upload.test/session-1"}, json={})
        if url.startswith("https://upload.test/"):
            self.chunks.setdefault("files/input", b"")
            name = f"files/input-{len(self.uploads) + 1}"
            self.uploads[name] = req.content.decode("utf-8")
            return httpx.Response(200, headers={"x-goog-upload-status": "final"}, json={"file": {"name": name, "sizeBytes": "100"}})
        if url.endswith(":batchGenerateContent"):
            if self.create_error is not None:
                return self.create_error() if callable(self.create_error) else self.create_error
            body = json.loads(req.content)
            self.created.append(body)
            self.input = body["batch"]["inputConfig"]["fileName"]
            self.display = body["batch"]["displayName"]
            return httpx.Response(200, json={"name": "batches/test-1", "metadata": {"state": "BATCH_STATE_PENDING", "displayName": self.display}})
        if url.endswith("/v1beta/batches"):
            if self.list_error is not None:
                return self.list_error
            ops = [] if self.display is None else [{"name": "batches/test-1", "metadata": {"displayName": self.display}}]
            return httpx.Response(200, json={"operations": ops})
        if url.endswith("/v1beta/batches/test-1:cancel"):
            self.cancelled.append("batches/test-1")
            return httpx.Response(200, json={})
        if url.endswith("/v1beta/batches/test-1"):
            self.polls += 1
            state = self.states[min(self.polls, len(self.states)) - 1]
            meta = {"state": f"BATCH_STATE_{state}", "displayName": self.display}
            out = {"name": "batches/test-1", "metadata": meta}
            if state == "SUCCEEDED":
                meta["output"] = {"responsesFile": "files/output-1"}
            if state == "FAILED":
                out["error"] = {"message": "internal batch failure"}
            return httpx.Response(200, json=out)
        if url.endswith("files/output-1:download"):
            lines = [l for l in self.uploads.get(self.input, "").split("\n") if l.strip()]
            out = []
            for line in lines:
                obj = json.loads(line)
                res = self.reply(obj["key"], obj["request"]) if self.reply else None
                if res is None:
                    gc = obj["request"].get("generation_config") or {}
                    schema = gc.get("response_json_schema") or gc.get("response_schema") or {"type": "object"}
                    res = {"key": obj["key"], "response": gemini_reply(data=_practice.mock_value(schema, "value", 1, obj["key"], schema))}
                out.append(json.dumps(res, ensure_ascii=False))
            if self.shuffle:
                out = out[::-1]
            return httpx.Response(200, headers={"content-type": "application/octet-stream"}, content="\n".join(out).encode("utf-8"))
        if req.method == "DELETE":
            self.deleted.append(url.split("/v1beta/")[-1])
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": {"code": 404, "message": f"unexpected request {url}", "status": "NOT_FOUND"}})


def quiet(fn, *a, **kw):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)

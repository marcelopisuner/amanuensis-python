"""The benchmark editor: a page in your browser with the page image beside a
form built from the codebook. It saves the benchmark file after every page,
so closing the browser or pressing Ctrl-C never loses more than the page in
hand. It edits the same file benchmark() writes as a template (an Excel
file, or a .txt/.md/.docx file for free text): what it saves opens in
Excel, and what was typed in Excel opens in the editor.

Where the R package runs a shiny app, this is a small web server from
Python's standard library, listening only on this computer (127.0.0.1),
with one self-contained HTML page. Everything except the server itself is
a plain function of a "state" (the benchmark's book plus the page in hand),
so the form, the reading of its inputs, the checks and the saving are tested
without a browser; ``handle()`` is the whole logic of the buttons.

Judgment calls:

- The form has a "finished" box: only finished pages are scored. Next page
  ticks it (moving on means the page is done); Save and Save and finish save
  the page as the box says, so stopping for the day never leaves a
  half-typed page marked done (the R package marks it done). A page is
  marked done only after its values pass the codebook's checks.
- Category and yes/no variables are drop-down menus with an empty first
  choice, so "not on the page" is always one click away. A value that is
  not one of the choices is shown as an extra choice, never emptied.
- Rows left entirely empty are dropped when saving.
- The file is the user's: a copy of it is kept before the editor first
  writes it; columns, sheets and notes the user added are written back; and
  if the file changed outside the editor since it was opened (saved from
  Excel), the editor never writes over it: what it holds goes to a file of
  its own.
- Every click says which page its form belongs to, and a click meant for
  another page (the second click of a double-click, after the editor moved
  on) is refused; the page also ignores a click that comes too soon after
  the previous one.
- The server answers only this page: every address carries a random token,
  the Host must be the server's own, a POST must come from it (Origin) and
  be JSON, so another web page in the browser, or another user of the same
  computer, cannot read or change the benchmark.
"""

from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import os
import secrets
import threading
import urllib.parse
import webbrowser

from ._errors import AmanuensisError, InputError
from ._images import as_images
from ._say import info, warn
from ._text import ext, page_key, parse_yes_no, show_path, stamp, stem

# Seconds to wait for the browser to load the page before giving up (a
# remote kernel, Colab, WSL: a browser that cannot reach this computer).
FIRST_REQUEST_SECONDS = 90


def file_stamp(path):
    """What tells whether a file changed: its size, time and content."""
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns, digest)


def editor_state(file, cb=None, pages=None) -> dict:
    """Everything the editor needs, read from the benchmark file."""
    from ._gold import TEXT_EXTS, gold_load, match_columns, parse_done, units_chr

    book = gold_load(file, fallback=cb)
    files = list(book["pages"]["file"])
    if not files:
        fc = match_columns(book["lines"].columns, ["file"])["file"]
        files = [] if fc is None else list(dict.fromkeys(str(v).strip() for v in book["lines"].iloc[:, fc] if v is not None))
    if not files:
        raise InputError(f"{os.path.basename(file)} lists no pages to transcribe.")
    units = units_chr(book["lines"], book["cb"], files)
    pk = [page_key(f) for f in book["pages"]["file"]]
    done = []
    for f in files:
        k = page_key(f)
        d = parse_done(book["pages"]["done"].iloc[pk.index(k)]) if k in pk else None
        if d is None:
            u = units[f]
            has_text = any(str(v).strip() for v in u["page"].values()) or any(len(c) for c in u["rows"].values())
            d = has_text if book["ext"] in TEXT_EXTS else False
        done.append(d)
    folder = book["about"].get("images")
    if pages is None and folder and os.path.isdir(folder):
        try:
            pages = as_images(folder)
        except Exception:
            pages = None
    images = {f: None for f in files}
    if pages is not None:
        pf = list(pages["file"])
        k1 = [page_key(f) for f in pf]
        k2 = [page_key(stem(f)) for f in pf]
        for f in files:
            if page_key(f) in k1:
                images[f] = pages["image"].iloc[k1.index(page_key(f))]
            elif page_key(stem(f)) in k2:
                images[f] = pages["image"].iloc[k2.index(page_key(stem(f)))]
    first = next((i for i, d in enumerate(done) if not d), 0)
    r = book["about"].get("rows")
    try:
        rows = int(r) if r not in (None, "") else None
    except ValueError:
        rows = None
    return {"file": file, "book": book, "files": files, "units": units, "done": done, "images": images,
            "current": first, "rows": rows, "stamp": file_stamp(file), "backup": None}


def editor_vars(state):
    cb = state["book"]["cb"]
    return [v for v in cb.vars if v.level == "page"], [v for v in cb.vars if v.level == "row"]


def shown_rows(state) -> int:
    """Rows shown for the page in hand: those typed so far, at least one."""
    pv, rv = editor_vars(state)
    if not rv:
        return 0
    u = state["units"][state["files"][state["current"]]]
    n = len(next(iter(u["rows"].values()), []))
    return max(1, n)


def form_values(inputs: dict, state) -> dict:
    """The page in hand as typed: input ids p_<j> (page variable j) and
    r_<i>_<j> (row i of row variable j), counted from 1."""
    pv, rv = editor_vars(state)

    def get(i):
        x = inputs.get(i)
        return "" if x is None else str(x)

    page = {v.variable: get(f"p_{j}") for j, v in enumerate(pv, 1)}
    n = shown_rows(state)
    rows = {v.variable: [get(f"r_{i}_{j}") for i in range(1, n + 1)] for j, v in enumerate(rv, 1)} if rv else {}
    return {"page": page, "rows": rows}


def check_values(values: dict, cb) -> list[str]:
    """Problems that stop the page being saved as done (empty is always fine)."""
    from ._gold import gold_convert

    out = []
    for v in cb.vars:
        if v.type not in ("number", "integer", "boolean"):
            continue
        if v.level == "page":
            x = [values["page"].get(v.variable, "")]
            where = [v.variable]
        else:
            x = values["rows"].get(v.variable, [])
            where = [f"row {i}, {v.variable}" for i in range(1, len(x) + 1)]
        conv = gold_convert([s.strip() for s in x], v.type, v.values)
        out += [f"{w}: \"{s}\" {p}" for w, s, p in zip(where, x, conv["problem"]) if p]
    return out


def store_page(state, values) -> dict:
    state["units"][state["files"][state["current"]]] = {"page": dict(values["page"]), "rows": {k: list(v) for k, v in values["rows"].items()}}
    return state


def add_row(state) -> dict:
    pv, rv = editor_vars(state)
    if not rv:
        return state
    u = state["units"][state["files"][state["current"]]]
    n = shown_rows(state)
    rows = {v.variable: list(u["rows"].get(v.variable, [])) for v in rv}
    for v in rv:
        rows[v.variable] += [""] * (n - len(rows[v.variable]))
        rows[v.variable].append("")
    u["rows"] = rows
    return state


def delete_row(state) -> dict:
    u = state["units"][state["files"][state["current"]]]
    if u["rows"] and len(next(iter(u["rows"].values()))):
        u["rows"] = {k: v[:-1] for k, v in u["rows"].items()}
    return state


def mark_done(state, done: bool = True) -> dict:
    import pandas as pd

    f = state["files"][state["current"]]
    state["done"][state["current"]] = bool(done)
    p = state["book"]["pages"]
    keys = [page_key(x) for x in p["file"]]
    word = "yes" if done else "no"
    if page_key(f) in keys:
        p.loc[keys.index(page_key(f)), "done"] = word
    else:
        p = pd.concat([p, pd.DataFrame({"file": [f], "done": [word], "notes": [None]})], ignore_index=True)
    state["book"]["pages"] = p
    return state


def state_book(state) -> dict:
    """The book to write: the editor's pages, with what the file held besides
    them kept (merge_lines: the columns the user added, the lines of pages
    the editor does not show)."""
    from ._gold import lines_chr, merge_lines

    book = dict(state["book"])
    book["lines"] = merge_lines(state["book"]["lines"], lines_chr(state["units"], book["cb"], state["files"]),
                                book["cb"], state["files"])
    return book


def save(state, file=None) -> str:
    """Write the benchmark file, or, when it changed outside the editor since
    the editor read or last wrote it, a file of its own beside it (raises
    InputError saying where)."""
    from ._gold import backup, book_write

    book = state_book(state)
    target = file or state["file"]
    book["file"] = target
    book["ext"] = ext(target)
    if file is None:
        now = file_stamp(target)
        if state.get("stamp") is not None and now != state["stamp"]:
            side = os.path.join(os.path.dirname(os.path.abspath(target)), f"{stem(target)}-from-editor-{stamp()}{os.path.splitext(target)[1]}")
            book["file"] = side
            book_write(book)
            raise InputError(f"{os.path.basename(target)} was changed outside the editor (saved from Excel or Word?) since the editor "
                             f"opened it, so the editor did not write over it: what the editor holds is in {os.path.basename(side)}.",
                             "Close the editor (Ctrl-C in Python), then run benchmark() again: it opens the file as it is now.")
        if state.get("backup") is None and os.path.exists(target):
            state["backup"] = backup(target)
    book_write(book)
    if file is None:
        state["stamp"] = file_stamp(target)
    return book["file"]


def page_title(state) -> str:
    f = state["files"][state["current"]]
    return f"Page {state['current'] + 1} of {len(state['files'])}: {f}" + (" (finished)" if state["done"][state["current"]] else "")


def handle(state, action: str, inputs: dict | None = None):
    """One click: returns (state, message, kind, finished). The page in hand is
    read from the form first, so nothing typed is lost by any button.

    ``inputs["_done"]`` is the "finished" box: Next page always finishes the
    page; the other buttons save it as the box says (without the box, as it
    was)."""
    inputs = inputs or {}
    name = os.path.basename(state["file"])

    def capture():
        return store_page(state, form_values(inputs, state))

    def write(s):
        try:
            save(s)
            return None
        except AmanuensisError as e:  # changed outside the editor, or open in Excel
            if "changed outside the editor" in e.head:
                return str(e)
            return (f"Could not save {name} ({e.head}). If it is open in Excel or Word, close it there, then click Save "
                    "again: what you typed is still here.")
        except Exception as e:  # pragma: no cover - anything else: keep the typing
            return f"Could not save {name} ({e}). What you typed is still here; click Save again."

    def save_page(done):
        s = capture()
        if done:
            probs = check_values(s["units"][s["files"][s["current"]]], s["book"]["cb"])
            if probs:
                return s, probs, "error"
        if done is not None:
            s = mark_done(s, done)
        err = write(s)
        if err:
            return s, [err], "error"
        what = "finished" if s["done"][s["current"]] else "not finished yet"
        extra = f" (a copy of the file as it was is in {os.path.basename(s['backup'])})" if s.get("backup") and not s.get("_said_backup") else ""
        if extra:
            s["_said_backup"] = True
        return s, [f"Saved {s['files'][s['current']]} in {name}, {what}{extra}."], "ok"

    box = inputs.get("_done")
    box = None if box is None else bool(box)
    if action == "add_row":
        return add_row(capture()), [], "ok", False
    if action == "del_row":
        return delete_row(capture()), [], "ok", False
    if action in ("prev", "goto"):
        s, msg, kind = save_page(box)
        if kind == "error":
            return s, msg, kind, False
        if action == "prev":
            s["current"] = max(0, s["current"] - 1)
        else:
            try:
                k = int(inputs.get("_page", s["current"] + 1)) - 1
            except ValueError:
                k = s["current"]
            s["current"] = min(max(0, k), len(s["files"]) - 1)
        return s, [], "ok", False
    if action in ("next", "save", "finish"):
        s, msg, kind = save_page(True if action == "next" else box)
        if kind == "error":
            return s, msg, kind, False
        if action == "next":
            if s["current"] < len(s["files"]) - 1:
                s["current"] += 1
            else:
                msg = ["That was the last page. Click Save and finish when you are done."]
        return s, msg, kind, action == "finish"
    raise InputError(f"Unknown editor action {action!r}.")


def view(state, message=None, kind="ok") -> dict:
    """What the page shows: the form's fields, the page's values, the image.
    Yes/no values come as "yes" or "no" when the benchmark reader would read
    them so (Sí, S, t, oui, ...), so the menu shows them; other text is sent
    as it is."""
    pv, rv = editor_vars(state)
    f = state["files"][state["current"]]
    u = state["units"][f]

    def var(v):
        return {"variable": v.variable, "type": v.type, "level": v.level, "description": v.description, "values": list(v.values)}

    def shown(v, x):
        if v.type != "boolean" or x is None or not str(x).strip():
            return x
        y = parse_yes_no(x)
        return x if y is None else ("yes" if y else "no")

    n = shown_rows(state)
    rows = []
    for i in range(n):
        rows.append([shown(v, (u["rows"].get(v.variable) or [""] * n)[i] if i < len(u["rows"].get(v.variable) or []) else "") for v in rv])
    return {"title": page_title(state), "file": f, "page_vars": [var(v) for v in pv], "row_vars": [var(v) for v in rv],
            "page": [shown(v, u["page"].get(v.variable, "")) for v in pv], "rows": rows, "rows_limit": state["rows"],
            "image": f"image/{state['current']}" if state["images"].get(f) else None,
            "message": message or [], "kind": kind, "done": state["done"][state["current"]],
            "n_pages": len(state["files"]), "current": state["current"] + 1}


class Editor:
    """The editor's routes, apart from the socket: ``route(method, path, body, headers)``.

    Every address is under ``/<token>/``; ``host`` (set once the server has
    a port) is the only Host accepted, and a POST must be JSON from the
    editor's own page (Origin, when the browser sends one)."""

    def __init__(self, state, token: str | None = None):
        self.state = state
        self.token = token or secrets.token_urlsafe(18)
        self.host = None
        self.finished = threading.Event()
        self.loaded = threading.Event()
        self.message, self.kind = [], "ok"
        self.lock = threading.Lock()

    def refuse(self, headers) -> tuple | None:
        if self.host is None:
            return None
        h = {k.lower(): v for k, v in (headers or {}).items()}
        if (h.get("host") or "").lower() != self.host.lower():
            return 403, "text/plain", b"forbidden"
        origin = h.get("origin")
        if origin is not None and origin.lower() != f"http://{self.host}".lower():
            return 403, "text/plain", b"forbidden"
        return None

    def route(self, method: str, path: str, body: bytes = b"", headers=None):
        bad = self.refuse(headers)
        if bad is not None:
            return bad
        path = urllib.parse.urlparse(path).path
        base = f"/{self.token}/"
        if not path.startswith(base):
            return 404, "text/plain", b"not found"
        sub = path[len(base):]
        if method == "GET" and sub == "":
            self.loaded.set()
            return 200, "text/html; charset=utf-8", PAGE.replace("@@TITLE@@", html.escape(f"Benchmark - {os.path.basename(self.state['file'])}")).encode("utf-8")
        if method == "GET" and sub == "state":
            self.loaded.set()
            return 200, "application/json", json.dumps(view(self.state, self.message, self.kind)).encode("utf-8")
        if method == "GET" and sub.startswith("image/"):
            try:
                k = int(sub.rsplit("/", 1)[-1])
                img = self.state["images"].get(self.state["files"][k])
            except (ValueError, IndexError):
                img = None
            if not img or not os.path.exists(img):
                return 404, "text/plain", b"not found"
            with open(img, "rb") as fh:
                return 200, mimetypes.guess_type(img)[0] or "image/jpeg", fh.read()
        if method == "POST" and sub == "action":
            ctype = {k.lower(): v for k, v in (headers or {}).items()}.get("content-type", "application/json")
            if not ctype.lower().startswith("application/json"):
                return 415, "text/plain", b"JSON only"
            try:
                req = json.loads(body or b"{}")
            except ValueError:
                return 400, "text/plain", b"bad request"
            if not isinstance(req, dict):
                return 400, "text/plain", b"bad request"
            with self.lock:
                page = req.get("page")
                if page != self.state["current"] + 1:
                    # a click for a form the editor has left: never applied to the page in hand
                    out = view(self.state, [f"That click was meant for page {page}, but the editor is on page "
                                            f"{self.state['current'] + 1}; nothing was saved. Check the page and click again."], "error")
                    out["finished"] = False
                    return 200, "application/json", json.dumps(out).encode("utf-8")
                self.state, self.message, self.kind, finished = handle(self.state, req.get("action", ""), req.get("inputs") or {})
                if finished:
                    self.finished.set()
                out = view(self.state, self.message, self.kind)
                out["finished"] = finished
            return 200, "application/json", json.dumps(out).encode("utf-8")
        return 404, "text/plain", b"not found"


def _server(handler):  # pragma: no cover - needs a socket; tests pass a fake server
    from http.server import ThreadingHTTPServer

    return ThreadingHTTPServer(("127.0.0.1", 0), handler)


def run_editor(file, cb=None, pages=None, *, server=None, opener=None, first_wait=None) -> bool:
    """Open the editor in the browser and serve it until Save and finish (True)
    or Ctrl-C (False). What was saved stays saved either way.

    When no browser could be opened, or the page is never loaded (a remote
    kernel, Colab or WSL, where the browser cannot reach this computer), it
    says so even with quiet=True, and stops waiting after
    FIRST_REQUEST_SECONDS instead of blocking until interrupted. ``server``
    and ``opener`` are for tests."""
    from http.server import BaseHTTPRequestHandler

    ed = Editor(editor_state(file, cb, pages))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, ctype, data):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(*ed.route("GET", self.path, b"", dict(self.headers.items())))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self._send(*ed.route("POST", self.path, self.rfile.read(n), dict(self.headers.items())))

        def log_message(self, *args):
            pass

    srv = (server or _server)(Handler)
    port = srv.server_address[1]
    ed.host = f"127.0.0.1:{port}"
    url = f"http://{ed.host}/{ed.token}/"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    info(f"Opening the benchmark editor in your browser. It saves {os.path.basename(str(file))} after every page.")
    info("Click Save and finish when you are done. To stop earlier, close the browser tab and press Ctrl-C here: what was saved stays saved.")
    try:
        opened = (opener or webbrowser.open)(url)
    except Exception:
        opened = False
    if not opened:
        warn("No browser could be opened from this Python session.",
             f"If this computer has one, open {url} in it. In a remote kernel (JupyterHub, Colab, a server) or WSL the browser "
             "cannot reach the editor: make the benchmark with app=False and fill in the file instead.")
    wait = FIRST_REQUEST_SECONDS if first_wait is None else first_wait
    try:
        if not ed.loaded.wait(wait):
            warn(f"The editor's page was not opened within {round(wait)} seconds, so the editor stopped.",
                 f"Nothing was changed in {os.path.basename(str(file))}. In a remote kernel, Colab or WSL use app=False and fill "
                 "in the file; otherwise run the line again and open the address it gives.")
            return False
        while not ed.finished.wait(0.5):
            pass
        return True
    except KeyboardInterrupt:
        return False
    finally:
        srv.shutdown()
        srv.server_close()


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>@@TITLE@@</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--card:#f6f8fa;--bad:#a40000;--ok:#1d6b1d}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--card:#151b23;--bad:#f85149;--ok:#3fb950}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{display:flex;gap:12px;padding:8px;height:100vh}.left{flex:7;display:flex;flex-direction:column;min-width:0}.right{flex:5;display:flex;flex-direction:column;min-width:0}
.img{flex:1;overflow:auto;border:1px solid var(--line);background:var(--card)}.img img{display:block;max-width:none}
.form{flex:1;overflow:auto;padding-right:6px}h2{font-size:16px;margin:4px 0}h3{font-size:14px;margin:12px 0 4px}
label{display:block;font-weight:600;margin-top:8px}.help{color:var(--muted);font-size:12px}
input,select,textarea{width:100%;font:inherit;padding:3px;background:var(--bg);color:var(--fg);border:1px solid var(--line)}
input[type=checkbox]{width:auto;margin-right:6px}
table{border-collapse:collapse}td,th{padding:2px 3px;vertical-align:top}th{white-space:nowrap;text-align:left}td input,td select,td textarea{min-width:90px}
.buttons button{margin:2px;padding:4px 10px}.msg.error{color:var(--bad)}.msg.ok{color:var(--ok)}
</style></head><body><div class="wrap">
<div class="left"><div><label style="display:inline">Zoom <input id="zoom" type="range" min="50" max="400" step="10" value="100" style="width:200px"></label>
<span id="title" style="font-weight:600;margin-left:12px"></span></div><div class="img" id="img"></div></div>
<div class="right"><div class="form" id="form"></div><div class="msg" id="msg"></div>
<div class="buttons"><button data-a="prev">Previous</button><button data-a="next">Next page</button><button data-a="save">Save</button>
<button data-a="finish"><b>Save and finish</b></button></div>
<p class="help">Next page marks this page finished. Save and Save and finish keep the box as you set it: untick it to keep a half-typed page out of the score.</p></div></div>
<script>
(function(){
var cur=null,busy=false,last=0;
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");}
function area(id,val,rows){return '<textarea id="'+id+'" rows="'+rows+'">\\n'+esc(val)+'</textarea>';}
function input(id,v,val){
 val=val==null?"":String(val);
 if(v.type==="category"){var o='<option value="">(empty)</option>';var vals=v.values.slice();
  if(val&&vals.indexOf(val)<0){vals.push(val);}
  vals.forEach(function(x){o+='<option value="'+esc(x)+'"'+(x===val?" selected":"")+'>'+esc(x)+(v.values.indexOf(x)<0?" (not in the codebook)":"")+'</option>';});
  return '<select id="'+id+'">'+o+'</select>';}
 if(v.type==="boolean"){var b='<option value="">(empty)</option><option value="yes"'+(val==="yes"?" selected":"")+'>yes</option><option value="no"'+(val==="no"?" selected":"")+'>no</option>';
  if(val&&val!=="yes"&&val!=="no"){b+='<option value="'+esc(val)+'" selected>'+esc(val)+' (not yes or no)</option>';}
  return '<select id="'+id+'">'+b+'</select>';}
 if(v.level==="page"&&v.variable==="text"&&v.type==="text"){return area(id,val,22);}
 if(val.indexOf("\\n")>=0){return area(id,val,Math.min(8,val.split("\\n").length));}
 return '<input id="'+id+'" value="'+esc(val)+'">';}
function render(s){cur=s;document.getElementById("title").textContent=s.title;
 var img=document.getElementById("img");img.innerHTML=s.image?'<img id="pic" src="'+esc(s.image)+'" alt="'+esc(s.file)+'">':'<p style="padding:20px">The image of '+esc(s.file)+' was not found. Give the page images to benchmark() (images=...) to see it here.</p>';
 zoom();var h='<label style="display:inline"><input type="checkbox" id="_done"'+(s.done?" checked":"")+'>This page is finished (only finished pages are scored)</label>';
 if(s.page_vars.length){h+="<h3>"+(s.row_vars.length?"Page":"Values")+"</h3>";
  s.page_vars.forEach(function(v,j){h+='<label for="p_'+(j+1)+'">'+esc(v.variable)+'</label>'+input("p_"+(j+1),v,s.page[j]);if(v.description)h+='<div class="help">'+esc(v.description.slice(0,160))+'</div>';});}
 if(s.row_vars.length){h+="<h3>Rows</h3>";if(s.rows_limit)h+="<p>Type only the first "+esc(s.rows_limit)+" rows of the page.</p>";
  h+='<p class="help">One line per row of the page\\'s table, in order. Hover over a column name for its description.</p><div style="overflow-x:auto"><table><thead><tr><th>row</th>';
  s.row_vars.forEach(function(v){h+='<th title="'+esc(v.description)+'">'+esc(v.variable)+'</th>';});h+="</tr></thead><tbody>";
  s.rows.forEach(function(r,i){h+="<tr><td>"+(i+1)+"</td>";s.row_vars.forEach(function(v,j){h+="<td>"+input("r_"+(i+1)+"_"+(j+1),v,r[j])+"</td>";});h+="</tr>";});
  h+='</tbody></table></div><div class="buttons"><button data-a="add_row">Add row</button><button data-a="del_row">Delete last row</button></div>';}
 document.getElementById("form").innerHTML=h;
 var m=document.getElementById("msg");m.className="msg "+s.kind;m.innerHTML=s.message.length>1?"<p>Not saved: correct these values first.</p><ul>"+s.message.map(function(x){return "<li>"+esc(x)+"</li>";}).join("")+"</ul>":esc(s.message[0]||"");
 bind();setBusy(false);}
function values(){var o={};document.querySelectorAll("#form input,#form select,#form textarea").forEach(function(e){if(e.id&&e.id!=="_done")o[e.id]=e.value;});var d=document.getElementById("_done");if(d)o._done=!!d.checked;return o;}
function setBusy(b){busy=b;document.querySelectorAll("button[data-a]").forEach(function(x){x.disabled=b;});}
function act(a){var t=Date.now();if(busy||t-last<700)return;last=t;setBusy(true);
 fetch("action",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:a,page:cur?cur.current:0,inputs:values()})})
 .then(function(r){return r.json();}).then(function(s){render(s);last=Date.now();if(s.finished){document.body.innerHTML="<p style='padding:20px'>Saved. You can close this tab.</p>";}})
 .catch(function(){setBusy(false);});}
function bind(){document.querySelectorAll("button[data-a]").forEach(function(b){b.onclick=function(){act(b.getAttribute("data-a"));};});}
function zoom(){var p=document.getElementById("pic");if(p)p.style.width=document.getElementById("zoom").value+"%";}
document.getElementById("zoom").oninput=zoom;
fetch("state").then(function(r){return r.json();}).then(render);
})();
</script></body></html>
"""

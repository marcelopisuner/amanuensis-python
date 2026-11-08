"""The object transcribe() returns: a pandas DataFrame of records (so the
usual pandas tools just work) that also carries the run - the pages table,
the page records with the model's data, the codebook, the instructions and
the settings - in ``res.info``. Selections of rows or columns
(``res[res.age > 30]``, ``res[["file", "age"]]``) keep the link, so
``output()`` can still write their provenance; ``merge`` and ``concat``
make plain tables, as in R.

A result holds only the page records made with the run's own settings. The
folder can hold others (made with an earlier codebook or other settings that
a partial redo did not reach yet, or invented practice replies under a real
model): those are never flattened into the table; the page is listed as
"not run", with the reason.
"""

from __future__ import annotations

import math
import os
import pickle

import pandas as pd

from ._codebook import records_path
from ._errors import ContractError, InputError, NotFoundError
from ._flatten import flatten_records, page_rows, page_unit
from ._models import MODELS_CHECKED, PRACTICE_MODELS, canonical_model, cost, price_for
from ._options import options
from ._say import plural, say
from ._store import pending_job_files, read_contracts, read_records, contract_objects
from ._text import chr_value, fmt_int, fmt_usd, is_missing, natural_order, page_key, sha12, show_path, to_json, utc_now

STATUS_LEVELS = ("ok", "check", "failed", "not run")
PAGES_COLUMNS = ["file", "source", "source_page", "status", "problem", "n_rows", "model", "model_version", "engine",
                 "finish_reason", "input_tokens", "cached_tokens", "image_tokens", "thinking_tokens", "output_tokens",
                 "cost_usd", "total_cost_usd", "attempts", "notes", "message", "date"]


# ------- fingerprints ------------------------------------------------------------

def sent_temperature(t):
    """A temperature as recorded: None when none was sent ("not sent" in records)."""
    if t is None or t == "not sent" or is_missing(t):
        return None
    return t


def _extra_settings(temperature, api_args) -> list[str]:
    """Settings that change what the model is asked, beyond the contract,
    model, resolution and thinking: a temperature that was sent, and
    api_args (which can change a request as much as the instructions can).
    Left out when unused, so the fingerprints of every run without them stay
    as they were."""
    out = []
    t = sent_temperature(temperature)
    if t is not None:
        try:
            out.append("t=" + chr_value(float(t)))
        except (TypeError, ValueError):
            out.append(f"t={t}")
    if api_args:
        out.append("a=" + sha12(to_json(api_args)))
    return out


def fingerprint(contract_id, model, resolution, thinking, temperature=None, api_args=None) -> str:
    """What makes a page done: the contract, the model that answers (an old
    name and the model it routes to are the same model), the resolution, the
    thinking level, and a temperature or api_args when they were sent.
    max_tokens is not part of it: it only caps a reply's length."""
    return "|".join([str(contract_id or ""), canonical_model(model or ""), str(resolution or ""), str(thinking or "")]
                    + _extra_settings(temperature, api_args))


def record_fingerprint(rec: dict) -> str:
    s = rec.get("settings") or {}
    return fingerprint(rec.get("contract"), rec.get("model"), s.get("resolution"), s.get("thinking"),
                       s.get("temperature"), s.get("api_args"))


def total_cost(rec: dict):
    """What a page's record says the page cost in all: its reply, plus the other
    paid attempts it carries (earlier tries, replaced results, failed redos)."""
    vals = [v for v in (rec.get("cost_usd"), rec.get("other_attempts_cost_usd"))
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not is_missing(v)]
    return sum(vals) if vals else None


def is_practice(rec: dict) -> bool:
    return (rec.get("model") or "") in PRACTICE_MODELS


def belongs_to(rec: dict, fp: str, model: str) -> bool:
    """Data (ok, to check) only when made with the same contract and settings;
    a failure holds no data and is reported whatever made it. Practice replies
    never count under a real model."""
    if is_practice(rec) and model not in PRACTICE_MODELS:
        return False
    if rec.get("status") == "failed":
        return True
    return record_fingerprint(rec) == fp


def why_left_out(rec: dict, settings: dict, contract_id: str) -> str:
    if is_practice(rec):
        return f"Has only a practice ({rec.get('model')}) result, which is never mixed with real ones."
    s = rec.get("settings") or {}
    what = []
    if (rec.get("contract") or "") != contract_id:
        what.append("other instructions or codebook")
    if canonical_model(rec.get("model") or "") != canonical_model(settings.get("model") or ""):
        what.append(f"model {rec.get('model') or 'unknown'}")
    if (s.get("resolution") or "") != (settings.get("resolution") or ""):
        what.append(f"resolution {s.get('resolution') or 'unknown'}")
    if (s.get("thinking") or "") != (settings.get("thinking") or ""):
        what.append(f"thinking {s.get('thinking') or 'unknown'}")
    if _extra_settings(s.get("temperature"), None) != _extra_settings(settings.get("temperature"), None):
        t = sent_temperature(s.get("temperature"))
        what.append(f"temperature {'not sent' if t is None else t}")
    if _extra_settings(None, s.get("api_args")) != _extra_settings(None, settings.get("api_args")):
        what.append("other api_args" if s.get("api_args") else "no api_args")
    return f"Has only a result made with other settings ({'; '.join(what) or 'settings'}); not part of this result."


# ------- the result object ----------------------------------------------------------

class ResultInfo:
    """Everything about a run besides its table of records."""

    def __init__(self, pages, records, codebook, instructions, contract, settings, folder, stopped=None):
        self.pages = pages
        self.records = records
        self.codebook = codebook
        self.instructions = instructions
        self.contract = contract
        self.settings = settings
        self.folder = folder
        self.date = utc_now()
        self.stopped = stopped
        self.n_rows = 0

    def __repr__(self) -> str:
        return (f"<amanuensis run: {len(self.pages)} pages, model {self.settings.get('model')}, "
                f"folder {self.folder or 'none (kept in memory)'}>")


class _BoundInfo:
    """What ``res.info`` gives: the run's ResultInfo (``res.info.pages``,
    ``res.info.settings``, ...), which can also be called. ``res.info`` hides
    pandas' ``DataFrame.info()`` method, and a pandas user's reflex
    ``res.info()`` then failed with "'ResultInfo' object is not callable":
    called, it is that method, on the table."""

    __slots__ = ("_info", "_frame")

    def __init__(self, info, frame):
        object.__setattr__(self, "_info", info)
        object.__setattr__(self, "_frame", frame)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_info"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_info"), name, value)

    def __call__(self, *args, **kwargs):
        return pd.DataFrame.info(pd.DataFrame(object.__getattribute__(self, "_frame")), *args, **kwargs)

    def __dir__(self):
        return dir(object.__getattribute__(self, "_info"))

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_info"))


class Result(pd.DataFrame):
    """The table of records of a run (one row per record), with the run in ``.info``.

    ``res.pages`` is the pages table (status, tokens, cost of every page).
    ``res.info()``, called, is pandas' ``DataFrame.info()``.
    """

    _metadata = ["_amanuensis"]

    @property
    def _constructor(self):
        return Result

    @property
    def info(self) -> ResultInfo:
        i = getattr(self, "_amanuensis", None)
        return None if i is None else _BoundInfo(i, self)

    @property
    def pages(self) -> pd.DataFrame:
        i = self.info
        return None if i is None else i.pages.copy()

    def _header(self) -> str:
        i = self.info
        if i is None:
            return ""
        if len(self) == i.n_rows:
            p = i.pages
            c = {s: int((p["status"] == s).sum()) for s in STATUS_LEVELS}
            head = f"amanuensis result: {fmt_int(len(self))} rows from {c['ok']} of {len(p)} pages ({i.settings.get('model')})"
            if c["check"] + c["failed"] + c["not run"]:
                head += f" | {c['check']} to check, {c['failed']} failed, {c['not run']} not run"
            return head
        return f"(part of an amanuensis result: {len(self)} of {i.n_rows} rows)"

    def __repr__(self) -> str:
        head = self._header()
        df = pd.DataFrame(self)
        if not len(df):
            body = "(no rows)"
        else:
            show = df.head(10).copy()
            width = 40
            for c in show.columns:
                if show[c].dtype == object:
                    show[c] = [v if not isinstance(v, str) else (v.replace("\n", " / ")[: width - 3] + "..." if len(v) > width else v.replace("\n", " / "))
                               for v in show[c]]
            body = show.to_string(index=False)
            if len(df) > 10:
                body += f"\n... and {fmt_int(len(df) - 10)} more rows"
        return (head + "\n" if head else "") + body

    def _repr_html_(self):  # pragma: no cover - notebooks
        head = self._header()
        html = pd.DataFrame(self).head(50)._repr_html_()
        return (f"<p>{head}</p>" if head else "") + html


def new_result(tab: pd.DataFrame, info: ResultInfo) -> Result:
    info.n_rows = len(tab)
    out = Result(tab)
    out._amanuensis = info
    return out


def result_info(x):
    return getattr(x, "_amanuensis", None) if isinstance(x, pd.DataFrame) else None


def pages_table(pages, page_recs: dict, cb, why: dict | None = None) -> pd.DataFrame:
    """One row per page image in scope: what happened to it."""
    why = why or {}
    path = records_path(cb)
    rows = []
    for _, pg in pages.iterrows():
        k = page_key(pg["file"])
        rec = page_recs.get(k) or {}
        tk = rec.get("tokens") or {}
        if rec.get("data") is not None:
            n_rows = 1 if path is None else len(page_rows(rec["data"], path))
        else:
            n_rows = None
        rows.append({
            "file": pg["file"], "source": pg.get("source") if not is_missing(pg.get("source")) else pg["file"],
            "source_page": pg.get("source_page"),
            "status": rec.get("status") or "not run", "problem": rec.get("problem"), "n_rows": n_rows,
            "model": rec.get("model"), "model_version": rec.get("model_version"),
            "engine": (rec.get("settings") or {}).get("engine"), "finish_reason": rec.get("finish_reason"),
            "input_tokens": tk.get("input"), "cached_tokens": tk.get("cached"), "image_tokens": tk.get("image"),
            "thinking_tokens": tk.get("thinking"), "output_tokens": tk.get("output"),
            "cost_usd": rec.get("cost_usd"), "total_cost_usd": total_cost(rec), "attempts": rec.get("attempts"),
            "notes": " ".join(str(n) for n in (rec.get("notes") or [])),
            "message": rec.get("message") if rec.get("message") is not None else why.get(k),
            "date": rec.get("date"),
        })
    df = pd.DataFrame(rows, columns=PAGES_COLUMNS)
    for c in ("source_page", "n_rows", "attempts"):
        df[c] = pd.array([None if is_missing(v) else int(v) for v in df[c]], dtype="Int64")
    for c in ("input_tokens", "cached_tokens", "image_tokens", "thinking_tokens", "output_tokens", "cost_usd", "total_cost_usd"):
        df[c] = pd.Series([math.nan if is_missing(v) else float(v) for v in df[c]], dtype="float64")
    for c in ("file", "source", "status", "problem", "model", "model_version", "engine", "finish_reason", "notes", "message", "date"):
        df[c] = pd.Series([None if is_missing(v) else str(v) for v in df[c]], dtype=object)
    return df


def new_result_info(run, pages, page_recs: dict, why: dict | None = None) -> ResultInfo:
    s = run.settings
    settings = {k: s.get(k) for k in ("model", "resolution", "thinking", "max_tokens", "engine")}
    settings["temperature"] = sent_temperature(s.get("temperature"))
    settings["api_args"] = s.get("api_args") or None
    return ResultInfo(pages=pages_table(pages, page_recs, run.contract["codebook"], why), records=page_recs,
                      codebook=run.contract["codebook"], instructions=run.contract["instructions"],
                      contract=run.contract["id"], settings=settings, folder=run.dir,
                      stopped=getattr(run, "stopped", None))


def finish(run, pages, plan=None) -> Result:
    """The result of a run, and the end-of-run summary."""
    recs = run.records
    keys = [page_key(f) for f in pages["file"]]
    page_recs = {k: recs[k] for k in keys if k in recs}
    from ._transcribe import run_fingerprint

    fp = run_fingerprint(run)
    mine = {k: belongs_to(r, fp, run.settings["model"]) for k, r in page_recs.items()}
    left_out = {k: why_left_out(r, run.settings, run.contract["id"]) for k, r in page_recs.items() if not mine[k]}
    page_recs = {k: r for k, r in page_recs.items() if mine[k]}
    waiting_files = pending_job_files(run.dir) if run.dir else []
    waiting = {page_key(f): "Waiting in a batch job at Google; run the same line later to collect it." for f in waiting_files}
    # an earlier failure of a page now waiting in a batch job is not its state
    for k in [k for k, r in page_recs.items() if k in waiting and r.get("status") == "failed"]:
        page_recs.pop(k)
    why = dict(waiting)
    for k, v in left_out.items():
        why.setdefault(k, v)
    info = new_result_info(run, pages, page_recs, why)
    tab = flatten_records([r for r in page_recs.values() if r.get("status") == "ok"], run.contract["codebook"])
    out = new_result(tab, info)
    report_run(out, run, plan, left_out={k: v for k, v in left_out.items() if k not in waiting}, waiting=list(waiting))
    return out


def standard_cost_per_page(recs, s: dict):
    """What one page would cost at the standard price (one page at a time)."""
    model = "gemini-3.8-flash" if s["model"] in PRACTICE_MODELS else s["model"]
    price = price_for(model, prices=s.get("prices"))
    if price is None:
        return None
    costs = []
    for r in recs.values():
        tk = r.get("tokens")
        if r.get("status") not in ("ok", "check") or not tk or is_missing(tk.get("input")):
            continue
        costs.append(cost({"input": tk.get("input"), "cached": tk.get("cached") or 0,
                           "output": tk.get("billed_output") or 0, "thinking": 0}, price))
    return sum(costs) / len(costs) if costs else None


def report_run(x: Result, run, plan, left_out=None, waiting=()) -> None:
    """The end-of-run summary: the headline says what this run did, with a
    success mark only when it went well; the lines below describe every page."""
    left_out = left_out or {}
    info = x.info
    p = info.pages
    s = run.settings
    n = len(p)
    counts = {st: int((p["status"] == st).sum()) for st in STATUS_LEVELS}
    wkeys = set(waiting)
    is_waiting = [(st == "not run") and page_key(f) in wkeys for st, f in zip(p["status"], p["file"])]
    n_wait = sum(is_waiting)
    n_left = counts["not run"] - n_wait
    # every paid attempt of a page counts, not only the one whose reply is kept
    total_cost = float(p["total_cost_usd"].sum(skipna=True)) if len(p) else 0.0
    other_cost = total_cost - (float(p["cost_usd"].sum(skipna=True)) if len(p) else 0.0)
    sent, collected = run.n_sent, run.collected
    got = run.statuses
    how = f" (resolution {s['resolution']}, thinking {s['thinking']}{', ' + s['engine'] if s['engine'] != 'sync' else ''})"
    if sent + collected > 0:
        parts = []
        if sent:
            parts.append(f"Sent {plural(sent, 'page')} to {s['model']}{how}")
        if collected:
            parts.append(f"{'collected' if sent else 'Collected'} {plural(collected, 'page')} from a batch job sent earlier")
        head = "; ".join(parts) + "."
        bullet = "x" if not any(g in ("ok", "check") for g in got) else ("v" if run.stopped is None and all(g == "ok" for g in got) else "!")
    elif n_wait:
        head = f"{plural(n_wait, 'page')} waiting in a batch job at Google; run the same line later to collect {'them' if n_wait != 1 else 'it'}."
        bullet = "i"
    elif run.stopped is not None:
        head, bullet = "No page was read in this run.", "x"
    elif plan is not None and plan.get("held"):
        k = len(plan["held"])
        head = f"Nothing to send: {plural(k, 'page')} that failed earlier {'are' if k != 1 else 'is'} not retried (see below)."
        bullet = "!"
    else:
        chose = " you chose" if plan is not None and len(plan["scope"]) < n else ""
        head = f"Nothing to send: every page{chose} is already done. (To extract pages again, use `redo`.)"
        bullet = "v"
    parts = [f"{counts['ok']} ok" if counts["ok"] else None, f"{counts['check']} to check" if counts["check"] else None,
             f"{counts['failed']} failed" if counts["failed"] else None,
             f"{n_wait} waiting in a batch job" if n_wait else None, f"{n_left} not run yet" if n_left else None]
    parts = [q for q in parts if q]
    if s["model"] in PRACTICE_MODELS:
        price_note = " (practice model: nothing was charged; shown at gemini-3.8-flash prices)"
    else:
        price_note = (f" at Google's paid prices of {MODELS_CHECKED}{' (half price)' if s['engine'] != 'sync' else ''}; "
                      "keys without billing are not charged")
    lines = [(bullet, head)]
    if run.stopped is not None:
        lines.append(("!", run.stopped))
    lines.append(("*", f"Of the {plural(n, 'page')}: {', '.join(parts)}."))
    others = f" (of which {fmt_usd(other_cost)} for other attempts of the same pages: retries and replaced results)" \
        if other_cost > 1e-9 else ""
    lines.append(("*", f"{plural(len(x), 'row')} in the table; estimated cost {fmt_usd(total_cost)} for all pages so far{others}{price_note}."))
    if run.spent > 0 and abs(run.spent - total_cost) > 1e-9:
        lines.append(("*", f"This run: {fmt_usd(run.spent)}."))
    if run.moved > 0:
        lines.append(("*", f"Moved {plural(run.moved, 'earlier result')} to {show_path(run.moved_to)} (kept, not deleted)."))
    redo_failed = getattr(run, "redo_failed", None) or {}
    if redo_failed:
        k = len(redo_failed)
        lines.append(("!", f"{plural(k, 'page')} to redo failed this time and {'keeps its' if k == 1 else 'keep their'} earlier "
                           f"result{'s' if k != 1 else ''}: {', '.join(list(redo_failed)[:3])}{' ...' if k > 3 else ''} "
                           f"({next(iter(redo_failed.values()))})."))
    if plan is not None and info.folder is not None:
        reached = set(run.written) | wkeys | {page_key(f) for f in redo_failed}
        not_reached = sum(1 for f in plan.get("supersede") or [] if page_key(f) not in reached)
        if not_reached:
            lines.append(("*", f"{plural(not_reached, 'page')} to redo {'were' if not_reached != 1 else 'was'} not reached and "
                               f"{'keep their earlier results' if not_reached != 1 else 'keeps its earlier result'}."))
    if info.folder is not None:
        lines.append(("*", f"Saved in {info.folder}."))
    say(*lines)
    if left_out:
        practice = [k for k, v in left_out.items() if "practice" in v]
        if practice:
            say(("!", f"{plural(len(practice), 'page')} {'has' if len(practice) == 1 else 'have'} only practice (mock/example) results, "
                      "never mixed with real ones: not in this table."),
                ("i", f"Run the same line again to extract {'it' if len(practice) == 1 else 'them'} for real."))
        other = len(left_out) - len(practice)
        if other:
            say(("!", f"{plural(other, 'page')} {'has' if other == 1 else 'have'} only results made with other settings (instructions, "
                      "codebook, model, resolution or thinking): not in this table."),
                ("i", f"Their old results stay in {info.folder or 'the results folder'}; the message column of res.pages says what differs."))
    chk = p[p["status"] == "check"]
    if len(chk):
        files = ", ".join(chk["file"].head(3)) + (" ..." if len(chk) > 3 else "")
        say(("!", f"{plural(len(chk), 'page')} to check (the reply did not match the codebook; kept apart from the table): {files}."),
            ("i", chk["message"].iloc[0]),
            ("i", "Include them with output(res, include=\"all\"), or extract them again with redo=\"check\"."))
    fl = p[p["status"] == "failed"]
    if len(fl):
        files = ", ".join(fl["file"].head(3)) + (" ..." if len(fl) > 3 else "")
        # the same rule the next run applies: a blocked, empty or unreadable
        # reply is tried (and paid) once more too
        from ._transcribe import should_retry
        retried = 0
        for f in fl["file"]:
            rec = info.records.get(page_key(f))
            if rec is None or should_retry(rec, s, None) is True:
                retried += 1
        msg = [("!", f"{plural(len(fl), 'page')} failed: {files}."), ("i", fl["message"].iloc[0])]
        if retried:
            msg.append(("i", f"{retried} of them will be tried again when you run the same line."))
        say(*msg)
    if plan is not None and plan.get("held"):
        k = len(plan["held"])
        say(("!", f"{plural(k, 'page')} that failed earlier {'was' if k == 1 else 'were'} not retried:"),
            *[("*", f"{f}: {why}") for f, why in plan["held"].items()])
    if n_left:
        per = standard_cost_per_page(info.records, s)
        if per:
            if s["engine"] == "sync":
                say(("i", f"The {plural(n_left, 'page')} not run yet would cost about {fmt_usd(per * n_left)} "
                          f"({fmt_usd(per * n_left / 2)} with batch=True)."))
            else:
                say(("i", f"The {plural(n_left, 'page')} not run yet would cost about {fmt_usd(per * n_left / 2)} with {s['engine']} "
                          f"({fmt_usd(per * n_left)} one page at a time)."))
    notes = int((p["notes"].fillna("") != "").sum())
    if notes:
        say(("i", f"{plural(notes, 'page')} carry notes worth a look (the notes column of res.pages)."))
    if options.next_steps:
        if len(x):
            say((">", "Next: output(res, \"results.xlsx\") saves the table; res.pages lists every page."))
        elif run.stopped is None and not n_wait:
            say((">", "Next: res.pages lists every page and what happened to it."))


# ------- rebuilding a result ------------------------------------------------------------

def result_table(x, include: str = "ok") -> pd.DataFrame:
    """The records table rebuilt from the page records: "ok", or "all" (pages
    to check too, with a page_status column after source_page)."""
    info = x.info if isinstance(x, Result) else x
    keep = ("ok", "check") if include == "all" else ("ok",)
    recs = [r for r in info.records.values() if r.get("status") in keep]
    tab = flatten_records(recs, info.codebook)
    if include == "all" and len(tab):
        status = {page_key(r["file"]): r["status"] for r in recs}
        ps = [status.get(page_key(f)) for f in tab["file"]]
        pos = list(tab.columns).index("source_page") + 1
        tab.insert(pos, "page_status", pd.Series(ps, dtype=object))
    return tab


def result_units(x) -> dict:
    """Per-page units for scoring, keyed by file: status, page values, rows.
    Pages to check are included with their data; failed pages have no data."""
    info = x.info if isinstance(x, Result) else x
    out = {}
    for r in info.records.values():
        u = page_unit(r.get("data") if r.get("status") in ("ok", "check") else None, info.codebook)
        u["status"] = r.get("status")
        out[r["file"]] = u
    return out


class _FolderRun:
    """What new_result_info() needs, for a result rebuilt from a folder."""

    def __init__(self, cid, cb, instr, settings, folder):
        self.contract = {"id": cid, "codebook": cb, "instructions": instr}
        self.settings = settings
        self.dir = folder
        self.stopped = None


def result_from_folder(folder, contract: str | None = None) -> Result:
    """Rebuild a result from a results folder alone: the page records found,
    flattened with the contract stored in contract.json. With more than one
    contract, ``contract`` picks one. As in transcribe(), a result never mixes
    settings: practice replies are left out when the folder also holds real
    results, and only the settings of the newest data record are used."""
    if not os.path.isdir(folder):
        raise NotFoundError(f"Cannot find the folder {show_path(folder)}.")
    folder = show_path(folder)
    recs = read_records(folder)
    if not recs:
        raise InputError(f"{folder} holds no results.", "Give the folder transcribe() saved into (its save_to).")
    contracts = read_contracts(folder)
    real = [r for r in recs.values() if not is_practice(r)]
    pool = real if real else list(recs.values())
    used = []
    for r in pool:
        c = r.get("contract") or ""
        if c and c not in used:
            used.append(c)
    if contract is None:
        if len(used) > 1:
            raise ContractError(f"{folder} holds results made with {len(used)} different instructions/codebooks.",
                                f"Choose one with contract= one of {', '.join(repr(u) for u in used)} (see contract.json).")
        contract = used[0] if used else ""
    entry = contracts.get(contract)
    if entry is None:
        raise ContractError(f"No contract {contract!r} in contract.json.")
    cb, instr = contract_objects(entry, contract)
    recs = [r for r in recs.values() if r.get("contract") == contract]
    if not recs:
        raise ContractError(f"No results in {folder} were made with contract {contract!r}.")
    recs = [recs[i] for i in natural_order([r["file"] for r in recs])]
    has_data = [r.get("status") in ("ok", "check") for r in recs]
    realr = [not is_practice(r) for r in recs]
    cand = [i for i in range(len(recs)) if has_data[i] and realr[i]] or [i for i in range(len(recs)) if has_data[i]] or list(range(len(recs)))
    pick = recs[max(cand, key=lambda i: (str(recs[i].get("date") or ""), -i))]
    ps = pick.get("settings") or {}
    settings = {"model": pick.get("model"), "resolution": ps.get("resolution"), "thinking": ps.get("thinking"),
                "max_tokens": ps.get("max_tokens"), "engine": ps.get("engine"),
                "temperature": sent_temperature(ps.get("temperature")), "api_args": ps.get("api_args") or None}
    fp = record_fingerprint(pick)
    mine = [belongs_to(r, fp, pick.get("model") or "") for r in recs]
    pages = pd.DataFrame({"image": [None] * len(recs), "file": [r["file"] for r in recs],
                          "source": [r.get("source") or r["file"] for r in recs],
                          "source_page": [r.get("source_page") for r in recs]})
    keys = [page_key(r["file"]) for r in recs]
    why = {keys[i]: why_left_out(recs[i], settings, contract) for i in range(len(recs)) if not mine[i]}
    run = _FolderRun(contract, cb, instr, settings, folder)
    info = new_result_info(run, pages, {keys[i]: recs[i] for i in range(len(recs)) if mine[i]}, why)
    tab = flatten_records([recs[i] for i in range(len(recs)) if mine[i] and recs[i].get("status") == "ok"], cb)
    if not all(mine):
        k = sum(1 for m in mine if not m)
        say(("!", f"{plural(k, 'page')} in {folder} {'has' if k == 1 else 'have'} only results made with other settings or by a "
                  "practice model: left out of the table (they are listed as not run)."),
            ("i", f"The table holds the results made with {pick.get('model')!r} (resolution {ps.get('resolution')}, thinking "
                  f"{ps.get('thinking')}); the message column of the pages table says what differs."))
    return new_result(tab, info)


def as_result(x) -> Result:
    """A result, a results folder, or a pickle written by output()."""
    if isinstance(x, Result) and x.info is not None:
        return x
    if isinstance(x, (str, os.PathLike)):
        p = str(os.path.expanduser(os.fspath(x)))
        if os.path.isdir(p):
            return result_from_folder(p)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (".pkl", ".pickle"):
            obj = load_pickle(p)
            if isinstance(obj, Result) and obj.info is not None:
                return obj
            raise InputError(f"{os.path.basename(p)} does not hold a result of transcribe().",
                             "Give the result itself, or its results folder.")
    raise InputError("Expected the result of transcribe(), or its results folder.",
                     "For example output(res, 'results.xlsx'), or output('results', 'results.xlsx') for the folder "
                     "transcribe() saved into.")


def load_pickle(path):
    """A result or benchmark saved with output(..., ".pkl"). Only open pickles
    you made yourself: a pickle can run code when it is opened."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception as e:
            raise InputError(f"{os.path.basename(str(path))} could not be read as a pickle file.", ("x", str(e))) from None

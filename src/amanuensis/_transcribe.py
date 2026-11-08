"""transcribe(): one call to the model per page image, each with a fresh
context, the instructions and the codebook's schema; every reply parsed and
checked against the codebook as soon as it arrives.

By default every page's record is saved as it arrives in a results folder
next to the page images, with the logs of the run, so a run that stops resumes
where it stopped, even after Python is closed, and nothing is paid twice.
``save_to=False`` keeps everything in memory instead (the session store in
_store.py still protects what was paid for until Python exits).

What decides whether a page is done (``plan()``): a record with status "ok"
or "check" made with the same contract (instructions + codebook), model,
resolution, thinking, and temperature and api_args when they were sent.
Failures are retried only when a retry can help: server and network trouble
always; a reply cut off at max_tokens only when max_tokens was raised; a
blocked or unreadable reply once; a page Google could not use as an image,
or a reply the package could not process ("internal", kept in its record),
never automatically.

The money rules, each argued where it is made: a paid reply is never lost
to an error (it is saved, or held in the session store, before the error is
raised); an earlier result is never written over without a copy, nor
replaced by a failed redo; a record that cannot be opened stops the run
instead of being paid again; many page images that only look changed are
not paid again without a yes; the budget holds from the first round; one
run at a time works in a results folder (``transcribe.lock``).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import platform
import re

from ._codebook import as_codebook, records_path
from ._errors import AmanuensisError, ContractError, InputError, Interrupted
from ._flatten import page_rows
from ._gemini import (IMAGE_TOKENS, RESOLUTIONS, call, classify_failure, error_details, error_message, gemini_key,
                      generate_request, is_api_error, is_network_error, make_client, media_level, mime_of, read_reply,
                      schema_for_api, finish_message, short_message)
from ._images import as_images, refuse_library_path
from ._instructions import as_instructions
from ._models import (DEFAULT_MODEL, PRACTICE_MODELS, canonical_model, check_model, cost, model_info, price_for,
                      thinking_for)
from ._options import ask_yes_no, can_ask, is_interactive, menu, options
from ._result import (Result, belongs_to, finish, fingerprint, is_practice, record_fingerprint, result_info)
from ._say import Progress, attention, info, listing, plural, quietly, say, warn
from ._store import (RECORD_VERSION, contract, contract_entry, default_results_dir, describe_change, move_aside,
                     read_contracts, read_jobs, read_records, release_lock, save_contract, session_keep, session_records,
                     superseded_dir, take_lock, write_record)
from ._text import (append_text, ext, fmt_int, fmt_usd, is_missing, now, page_key, remove_leftovers, safe_stem, show_path,
                    stamp, stem, to_json, today, user_path, utc_now)
from ._validate import validate
from ._version import __version__


def transcribe(images, instructions=None, codebook=None, *, model: str = DEFAULT_MODEL, batch: bool = False,
               save_to=None, continue_from=None, resolution: str = "ultra_high", thinking: str = "low",
               max_tokens: int = 32768, temperature: float | None = None, flex: bool = False, only=None,
               limit: int | None = None, redo=None, budget: float | None = None, confirm: bool | None = None,
               workers: int = 1, batch_size: int = 500, wait: bool = True, log=None, retries: int = 4,
               timeout: float | None = None, prices: dict | None = None, api_args: dict | None = None,
               quiet: bool = False) -> Result:
    """Transcribe page images into data.

    Sends each page image to Gemini with your instructions and codebook, and
    collects the replies into a table: one row per record, with the file,
    source and page each row comes from. Each page's result is saved as it
    arrives in a results folder (by default ``<images folder>/extraction``),
    so a run that stops - Ctrl-C, the budget, an error, a closed laptop -
    continues where it stopped when the same line runs again, and nothing is
    paid twice. When every page is done, running the line again costs nothing
    and returns the table.

    Pages that failed are tried again when that can help. Pages read with
    other instructions, codebook or settings, or by a practice model, never
    mix into the table: if you change the instructions, the codebook or the
    model, you are asked whether to keep the old results apart (in a new
    folder, ``extraction_2``) or redo the pages.

    Every page's tokens and estimated cost are recorded, at Google's paid
    prices (see :func:`gemini_models`). In an interactive session with 25 or
    more pages, the projected cost is shown after the first two pages and
    you are asked before the rest are sent. ``budget`` stops a run before it
    spends more than you allow. ``batch=True`` (Google's Batch API) and
    ``flex=True`` cost half; a batch job can outlive Python, so it is recorded
    in the results folder and collected by running the same line later.

    The key is read from the GEMINI_API_KEY environment variable (set it in
    the terminal: ``export GEMINI_API_KEY="..."``); ``model="mock"`` invents
    replies without a key, and ``"example"`` replays the example project's.

    Parameters
    ----------
    images : Images, folder, or list of image paths
        The page images, usually the result of :func:`images`.
    instructions : Instructions or path, optional
        From :func:`instructions`, or a file. Default: general instructions.
    codebook : Codebook, path or DataFrame, optional
        From :func:`codebook`, or a codebook file. Default: free text.
    model : str
        The Gemini model (see :func:`gemini_models`), or "mock"/"example".
    batch : bool
        Use the Batch API (half price, usually minutes, at most 24 hours).
    save_to : path or False, optional
        The results folder. Default ``<images folder>/extraction``; False keeps
        everything in memory (nothing is written).
    continue_from : Result, optional
        An earlier result (e.g. read back from a .pkl) whose pages are kept.
    resolution : str
        "low", "medium", "high" or "ultra_high" (default): how much detail of
        each image Google reads.
    thinking : str
        "low" (default), "medium", "high", or "minimal" where accepted.
    max_tokens : int
        Longest reply allowed, thinking included. Default 32768.
    temperature : float, optional
        Not sent by default (Gemini 3.6 and newer ignore it).
    flex : bool
        Google's flex tier: half price, each page may take minutes.
    only : file names, or a table with a ``file`` column, optional
        Process only these pages, e.g. ``pages.file[:5]`` or a benchmark.
    limit : int, optional
        Process at most this many pages now (0 sends nothing and returns
        what is done).
    redo : "all", "failed", "check", or file names, optional
        Pages to extract again; their earlier results are moved to
        ``superseded/`` (never deleted) as the new ones arrive. A redo that
        fails keeps the earlier result. Also needed to extract again many
        page images that were made again (three or more whose files changed).
    budget : float, optional
        Most to spend in this run, in US dollars. Before any page is
        measured, one page is sent alone; then each round sends only as many
        pages as the cost measured so far allows. ``budget=0`` sends nothing.
    confirm : bool, optional
        Ask before spending on 25 or more pages. Default: when interactive.
    workers : int
        Pages sent at the same time (1-8). Default 1.
    batch_size : int
        Most pages per batch job. Default 500.
    wait : bool
        For batch: wait for the results (default), or return at once.
    log : bool or path, optional
        The logs of the run (run_manifest.jsonl, token_usage.txt, errors.log):
        by default in the results folder; False for none, or a folder.
    retries : int
        Retries when Google is busy or the network fails (5, 10, 20, 40 s).
    timeout : float, optional
        Seconds to wait for one reply (default by model: 10 minutes, 20 for
        gemini-4-argon, 15 with flex).
    prices : dict, optional
        Your own prices per million tokens: {"input":, "output":, "cached":}.
    api_args : dict, optional
        For experts: merged into every REST request body, e.g.
        ``{"generationConfig": {"seed": 1}}``.
    quiet : bool
        Print only warnings and errors.

    Returns
    -------
    Result
        A DataFrame of records (pages whose reply matched the codebook),
        starting with file, source, source_page and row. ``res.pages`` lists
        every page with its status, tokens and cost.
    """
    with quietly(quiet):
        return _transcribe(images, instructions, codebook, model=model, batch=batch, save_to=save_to,
                           continue_from=continue_from, resolution=resolution, thinking=thinking,
                           max_tokens=max_tokens, temperature=temperature, flex=flex, only=only, limit=limit,
                           redo=redo, budget=budget, confirm=confirm, workers=workers, batch_size=batch_size,
                           wait=wait, log=log, retries=retries, timeout=timeout, prices=prices, api_args=api_args)


def _num_arg(x, arg, minimum, none_ok=False):
    if x is None and none_ok:
        return
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or x < minimum:
        raise InputError(f"`{arg}` must be a single number of at least {minimum:g}.")


def _transcribe(images, instructions, codebook, *, model, batch, save_to, continue_from, resolution, thinking,
                max_tokens, temperature, flex, only, limit, redo, budget, confirm, workers, batch_size, wait, log,
                retries, timeout, prices, api_args, _restarted=False):
    # ------- check every argument before reading a file or calling Google
    model = check_model(model)
    if resolution not in RESOLUTIONS:
        raise InputError(f"`resolution` must be one of {', '.join(repr(r) for r in RESOLUTIONS)}.", ("x", f"You gave {resolution!r}."))
    _num_arg(max_tokens, "max_tokens", 1)
    _num_arg(retries, "retries", 0)
    _num_arg(workers, "workers", 1)
    _num_arg(batch_size, "batch_size", 1)
    _num_arg(budget, "budget", 0, none_ok=True)
    _num_arg(limit, "limit", 0, none_ok=True)
    _num_arg(timeout, "timeout", 1, none_ok=True)
    if temperature is not None:
        _num_arg(temperature, "temperature", 0)
    if workers > 8:
        raise InputError("`workers` can be at most 8.")
    if batch and flex:
        raise InputError("Use either batch=True or flex=True, not both (both cost half).")
    if save_to is False and batch:
        # a job can outlive Python, and its record must too
        try:
            suggest = default_results_dir(as_images(images))
        except Exception:
            suggest = "results"
        raise InputError("A batch job can take hours, so it needs a folder where it is recorded until its results are collected, "
                         "even if Python is closed.",
                         f"Leave out save_to=False (the default folder is {show_path(suggest)}), give another folder, "
                         "or leave out batch=True to read the pages one by one, at full price.")
    if continue_from is not None and result_info(continue_from) is None:
        raise InputError("`continue_from` must be an earlier result of transcribe().",
                         "For example continue_from=res, or a result saved with output(res, 'res.pkl') and read back with output('res.pkl').")
    if api_args is not None and not isinstance(api_args, dict):
        raise InputError("`api_args` must be a dict.")
    if prices is not None and (not isinstance(prices, dict) or "input" not in prices or "output" not in prices):
        raise InputError("`prices` must look like {\"input\": 0.75, \"output\": 3.75} (dollars per million tokens).")
    minfo = model_info(model)
    thinking_level = thinking_for(model, thinking)
    if minfo is not None and max_tokens > minfo["max_output"]:
        attention(f"{model!r} replies with at most {fmt_int(minfo['max_output'])} tokens; using max_tokens={minfo['max_output']}.")
        max_tokens = minfo["max_output"]
    if temperature is not None and (minfo is None or not re.search(r"2\.5|^gemini-3-|3\.1|3\.5-flash-lite", model)):
        info("Gemini 3.6 and newer models ignore `temperature`; it is sent anyway as you asked.")
    if budget is not None and price_for(model, prices=prices) is None and model not in PRACTICE_MODELS:
        raise InputError(f"`budget` needs prices, and {model!r} is not in the package's price table.",
                         "Give them with prices={\"input\": ..., \"output\": ...} (dollars per million tokens).")

    pages = as_images(images)
    instr = as_instructions(instructions)
    cb = as_codebook(codebook)
    key = "mock-key" if model in PRACTICE_MODELS else gemini_key()
    if model == "example":
        from ._practice import check_example_pages
        check_example_pages(pages)
    settings = {
        "model": model, "resolution": resolution, "thinking": thinking, "thinking_level": thinking_level,
        "max_tokens": int(max_tokens), "temperature": temperature,
        "engine": "batch" if batch else "flex" if flex else "sync", "flex": bool(flex), "api_args": api_args or {},
        "timeout": timeout or (900 if flex else (minfo or {}).get("timeout") or 600),
        "retries": int(retries), "prices": prices, "schema_channel": None,
    }
    con = contract(instr, cb)
    folder = results_dir(pages, save_to)
    lock = take_lock(folder) if folder is not None else None
    try:
        return _transcribe_in(images, instr, cb, pages, settings, con, folder, lock, key, model=model, batch=batch,
                              continue_from=continue_from, resolution=resolution, thinking=thinking, max_tokens=max_tokens,
                              temperature=temperature, flex=flex, only=only, limit=limit, redo=redo, budget=budget,
                              confirm=confirm, workers=workers, batch_size=batch_size, wait=wait, log=log, retries=retries,
                              timeout=timeout, prices=prices, api_args=api_args)
    finally:
        release_lock(lock)


def _transcribe_in(images, instr, cb, pages, settings, con, folder, lock, key, *, model, batch, continue_from, resolution,
                   thinking, max_tokens, temperature, flex, only, limit, redo, budget, confirm, workers, batch_size, wait,
                   log, retries, timeout, prices, api_args):
    """The run itself, with the results folder ready and locked."""
    if folder is not None:
        # the temporary files of an interrupted write, now that no other run
        # can be writing them (the folder is locked)
        remove_leftovers(folder)
    log_folder = log_dir(log, folder)

    # ------- what is already there: the folder, then continue_from, then what
    # this Python session already read for these images with these settings
    records = read_records(folder)
    run = Run(settings, con, folder, log_folder, pages, key)
    records = add_earlier_records(records, run, pages, continue_from)
    run.records = records
    if folder is not None:
        from ._batch import collect_pending_jobs

        save_contract(folder, con)
        try:
            run.earlier_jobs = [j.get("name") or "" for j in read_jobs(folder)]
        except InputError:
            run.earlier_jobs = []
        collect_pending_jobs(run, records)  # never pay twice for a page in a pending job
        records = run.records_on_disk if run.records_on_disk is not None else records
        # pages collected from a job sent by an earlier call were paid for then
        run.count_collected()
    plan_ = plan(pages, records, run, only=only, redo=redo, limit=limit)
    if plan_.get("restart"):
        # results for these settings are kept apart in another folder: continue there
        release_lock(lock)
        return _transcribe(images, instr, cb, model=model, batch=batch, save_to=plan_["restart"],
                           continue_from=continue_from, resolution=resolution, thinking=thinking,
                           max_tokens=max_tokens, temperature=temperature, flex=flex, only=only, limit=limit,
                           redo=redo, budget=budget, confirm=confirm, workers=workers, batch_size=batch_size,
                           wait=wait, log=log, retries=retries, timeout=timeout, prices=prices, api_args=api_args,
                           _restarted=True)
    announce_plan(plan_, run)
    run.records = records
    confirm = is_interactive() if confirm is None else bool(confirm)
    interrupted = None
    try:
        if batch and (plan_["todo"] or run.pending_jobs):
            from ._batch import run_batch
            run_batch(run, plan_, batch_size=int(batch_size), wait=wait, budget=budget)
        elif plan_["todo"]:
            run_sync(run, plan_, workers=int(workers), budget=budget, confirm=confirm)
    except KeyboardInterrupt as e:
        interrupted = e
    res = finish(run, pages, plan_)
    if run.error is not None:
        # the pages read stay reachable from the error, as Interrupted.result
        try:
            run.error.result = res
        except (AttributeError, TypeError):  # pragma: no cover - an exception type without a __dict__
            pass
        raise run.error
    if interrupted is not None or run.interrupted:
        raise Interrupted(run.stopped or "Stopped by you.", res)
    return res


# ------- the run ----------------------------------------------------------------------

class Run:
    """The state of one transcribe() call."""

    def __init__(self, settings, con, folder, log_folder, pages, key):
        self.settings = settings
        self.contract = con
        self.dir = folder
        self.log_dir = log_folder
        self.pages = pages
        self.key = key
        self.records = {}
        self.records_on_disk = None
        self.pending_jobs = []
        self.started = stamp()
        self.spent = 0.0          # this run's sends only
        self.n_sent = 0           # records written for requests sent in this run
        self.n_priced = 0         # ... of which with a cost
        self.collected = 0        # records collected from jobs sent by an earlier call
        self.earlier_jobs = []
        self.statuses = []
        self.written = []         # page_key() of every page written by this run
        self.redo_failed = {}     # file -> why: a redo that failed, the earlier result kept
        self.moved = 0
        self.moved_to = None
        self.stopped = None
        self.interrupted = False
        self.error = None
        self.from_session = []
        self.from_continue = []
        self._client = None

    @property
    def practice(self) -> bool:
        return self.settings["model"] in PRACTICE_MODELS

    def client(self, model: str | None = None):
        """The SDK client: a practice model's own local server, else Google."""
        model = model or self.settings["model"]
        if model in PRACTICE_MODELS:
            from ._practice import handler
            if getattr(self, "_practice_client", None) is None or self._practice_model != model:
                self._practice_client = make_client("mock-key", handler=handler(model))
                self._practice_model = model
            return self._practice_client
        if self._client is None:
            self._client = make_client(self.key if self.key != "mock-key" else gemini_key())
        return self._client

    def count_collected(self) -> None:
        self.collected += self.n_sent
        self.n_sent = 0
        self.n_priced = 0
        self.spent = 0.0

    def page_index(self, k: str):
        for i, f in enumerate(self.pages["file"]):
            if page_key(f) == k:
                return i
        return None


def run_fingerprint(run) -> str:
    s = run.settings
    return fingerprint(run.contract["id"], s["model"], s["resolution"], s["thinking"], s.get("temperature"), s.get("api_args"))


def results_dir(pages, save_to):
    """None for save_to=False (memory only); else the folder, created."""
    if save_to is False:
        return None
    if save_to is None or save_to is True:
        save_to = default_results_dir(pages)
    save_to = user_path(save_to)
    if not isinstance(save_to, (str, os.PathLike)):
        raise InputError("`save_to` must be a folder path, or False to keep the results in memory only.")
    refuse_library_path(save_to)
    # save_to="results.xlsx" would make a folder of that name (one file per
    # page), and output(res, "results.xlsx") then fails on it
    e = ext(save_to)
    if e in OUTPUT_EXTS and not os.path.isdir(save_to):
        raise InputError(f"`save_to` is the results folder, where every page is saved as it arrives; {os.path.basename(str(save_to))!r} "
                         "looks like a file name.",
                         f"Give a folder (or leave out save_to), and save the table afterwards with output(res, {os.path.basename(str(save_to))!r}).")
    os.makedirs(save_to, exist_ok=True)
    folder = show_path(save_to)
    if os.name == "nt" and len(folder) + max(len(f) for f in pages["file"]) + 6 > 240:  # pragma: no cover
        warn(f"The results folder path is very long ({len(folder)} characters); Windows may refuse to write some files.",
             "Move the project to a shorter path, e.g. C:/research/.")
    return folder


# the file types output() writes: a folder named like one of them is a mistake
OUTPUT_EXTS = ("xlsx", "xls", "csv", "tsv", "dta", "pkl", "pickle", "json", "txt", "md", "docx", "rds", "html", "pdf")


def log_dir(log, folder):
    if log is False:
        return None
    if log is None:
        return folder
    if log is True:
        if folder is None:
            raise InputError("log=True writes the log files into the results folder, and save_to=False keeps none.",
                             "Leave out save_to=False, or give a folder for the logs alone: log=\"logs\".")
        return folder
    if not isinstance(log, (str, os.PathLike)):
        raise InputError("`log` must be True, False or a folder.")
    log = user_path(log)
    os.makedirs(log, exist_ok=True)
    return show_path(log)


def add_earlier_records(records: dict, run: Run, pages, continue_from) -> dict:
    """Pages read before, outside the folder: those of ``continue_from`` and
    those this Python session already read for these images with these
    settings. The folder comes first; pages it lacks are written into it.
    Pages of ``continue_from`` read with other settings stop the run before
    anything is spent: continuing them would mix two settings, and reading
    them again silently would pay twice."""
    fp = run_fingerprint(run)
    keys = [page_key(f) for f in pages["file"]]
    from_continue = {}
    if continue_from is not None:
        cinfo = result_info(continue_from)
        recs = {k: r for k, r in cinfo.records.items() if k in keys}
        recs = {k: r for k, r in recs.items() if not (is_practice(r) and not run.practice)}
        other = [k for k, r in recs.items() if r.get("status") in ("ok", "check") and record_fingerprint(r) != fp]
        if other:
            what = describe_continue_change(cinfo, run)
            raise ContractError(f"{plural(len(other), 'page')} in continue_from {'was' if len(other) == 1 else 'were'} read with other settings:",
                                *[("*", w) for w in what],
                                ("i", "Continue with the same instructions, codebook, model, resolution and thinking, or leave out "
                                      "continue_from to read the pages again with the new settings (they are paid again)."))
        if not recs and cinfo.records:
            attention("None of the pages in continue_from are among these images; nothing to continue.")
        from_continue = recs
    from_session = session_records(fp, pages)
    extra = dict(from_continue)
    for k, r in from_session.items():
        extra.setdefault(k, r)
    # a page with data read earlier wins over a failure in the folder: a reply
    # whose record could not be written is held in the session store, and the
    # folder may hold an older failure of that page
    extra = {k: r for k, r in extra.items()
             if k not in records or (records[k].get("status") == "failed" and r.get("status") in ("ok", "check"))}
    if not extra:
        return records
    n_cont = sum(1 for k in extra if k in from_continue)
    n_sess = len(extra) - n_cont
    if run.dir is not None:
        for r in extra.values():
            write_record(run.dir, r)
        info(f"Saved {plural(len(extra), 'page')} read earlier ({n_cont} from continue_from, {n_sess} from this Python session) "
             f"in {run.dir}; {'it is' if len(extra) == 1 else 'they are'} not sent again.")
    else:
        img = dict(zip(keys, pages["image"]))
        for k, r in extra.items():
            session_keep(r, img.get(k), fp)
        run.from_session = [k for k in extra if k not in from_continue]
        run.from_continue = [k for k in extra if k in from_continue]
    out = dict(records)
    out.update(extra)
    return out


def describe_continue_change(cinfo, run) -> list[str]:
    s = run.settings
    what = []
    if cinfo.contract != run.contract["id"]:
        try:
            old = contract_entry({"codebook": cinfo.codebook, "instructions": cinfo.instructions})
            what += describe_change(old, contract_entry(run.contract))
        except Exception:
            what.append("instructions or codebook")
    os_ = cinfo.settings or {}
    if canonical_model(os_.get("model") or "") != canonical_model(s["model"]):
        what.append(f"model: {os_.get('model')} -> {s['model']}")
    if (os_.get("resolution") or "") != s["resolution"]:
        what.append(f"resolution: {os_.get('resolution')} -> {s['resolution']}")
    if (os_.get("thinking") or "") != s["thinking"]:
        what.append(f"thinking: {os_.get('thinking')} -> {s['thinking']}")
    what += _extra_changes([os_.get("temperature")], [os_.get("api_args")], s)
    return list(dict.fromkeys(what)) or ["settings"]


def _extra_changes(temps, api_args, s) -> list[str]:
    """Words for a temperature or api_args that differ from this run's."""
    from ._result import _extra_settings, sent_temperature

    out = []
    now_t = _extra_settings(s.get("temperature"), None)
    old_t = [sent_temperature(t) for t in temps if _extra_settings(t, None) != now_t]
    if old_t:
        shown = ", ".join(dict.fromkeys("not sent" if t is None else str(t) for t in old_t))
        new = sent_temperature(s.get("temperature"))
        out.append(f"temperature: {shown} -> {'not sent' if new is None else new}")
    now_a = _extra_settings(None, s.get("api_args"))
    if any(_extra_settings(None, a) != now_a for a in api_args):
        out.append("api_args: " + ("other api_args" if s.get("api_args") else "none now"))
    return out


# ------- planning -------------------------------------------------------------------

def resolve_only(x, pages) -> list[int]:
    """Positions (from 0) of the pages named: file names (with or without the
    extension), a list of them, or a table with a ``file`` column (pages,
    a result, a benchmark)."""
    import pandas as pd

    if isinstance(x, pd.DataFrame):
        if "file" not in x.columns:
            raise InputError("A table given to `only`/`redo` needs a `file` column.")
        x = list(dict.fromkeys(x["file"].tolist()))
    elif isinstance(x, pd.Series):
        x = x.tolist()
    elif isinstance(x, (str, os.PathLike)):
        x = [str(x)]
    if not isinstance(x, (list, tuple)) or not all(isinstance(v, (str, os.PathLike)) for v in x):
        raise InputError("Pages must be given as file names, e.g. only=pages.file[:5], or a table with a file column.",
                         "Page numbers are not accepted: use the file names images() gave the pages.")
    keys = [page_key(stem(f)) for f in pages["file"]]
    out, missing = [], []
    for v in x:
        k = page_key(stem(str(v)))
        if k in keys:
            out.append(keys.index(k))
        else:
            missing.append(str(v))
    if missing:
        raise InputError(f"These pages are not among the images: {', '.join(missing[:5])}.",
                         f"Use file names as listed by images() (e.g. {pages['file'].iloc[0]!r}).")
    return out


def should_retry(rec: dict, s: dict, redo_mode):
    """True to retry a failed page now, or a sentence saying why not."""
    if redo_mode == "failed":
        return True
    problem = rec.get("problem") or "server"
    attempts = int(rec.get("attempts") or 1)
    if problem == "truncated":
        old = int((rec.get("settings") or {}).get("max_tokens") or 0)
        return True if s["max_tokens"] > old else f"cut off at max_tokens = {old}; run again with a larger max_tokens to retry it"
    if problem in ("blocked", "not_json", "control_chars", "empty"):
        return True if attempts < 2 else f"{rec.get('message') or problem} (tried {attempts} times; use redo=\"failed\" to try again)"
    if problem in ("image", "internal"):
        # a newer version of the package may have fixed the cause: try once
        # more. "internal": the package could not process a reply it paid for
        # (the reply is kept in the record); sent again only on request
        if rec.get("amanuensis") != __version__ or rec.get("implementation") != "python":
            return True
        default = "Google could not use this image" if problem == "image" else "amanuensis could not process the reply"
        return f"{rec.get('message') or default} (use redo=\"failed\" to try again)"
    return True


def plan(pages, records: dict, run: Run, only=None, redo=None, limit=None) -> dict:
    """Which pages to send: todo, done, held (failed pages not retried, with
    reasons), supersede (files whose old records move aside), restart."""
    s = run.settings
    fp_now = run_fingerprint(run)
    files = list(pages["file"])
    keys = [page_key(f) for f in files]
    scope = list(range(len(files))) if only is None else resolve_only(only, pages)
    redo_mode, redo_keys = None, set()
    if redo is not None:
        if isinstance(redo, str) and redo in ("all", "failed", "check"):
            redo_mode = redo
        else:
            redo_keys = {keys[i] for i in resolve_only(redo, pages)}
    in_job = {page_key(f) for f in run.pending_jobs}
    todo, done, changed, supersede_ = [], [], [], []
    held, practice, image_changed, image_results = {}, [], [], []

    def settle(i, rec):
        """The usual decision for a page with a record (its image unchanged)."""
        if rec.get("status") in ("ok", "check"):
            if is_practice(rec) and not run.practice:
                practice.append(i)
                todo.append(i)
            elif record_fingerprint(rec) != fp_now:
                changed.append(i)
            elif rec["status"] == "check" and redo_mode == "check":
                todo.append(i)
                supersede_.append(files[i])
            else:
                done.append(i)
            return
        retry = should_retry(rec, s, redo_mode)
        if retry is True:
            todo.append(i)
        else:
            held[files[i]] = retry

    for i in scope:
        k = keys[i]
        if k in in_job:
            continue
        rec = records.get(k)
        forced = redo_mode == "all" or k in redo_keys
        if rec is None:
            todo.append(i)
            continue
        if forced:
            todo.append(i)
            if rec.get("status") in ("ok", "check"):
                supersede_.append(files[i])
            continue
        img = pages["image"].iloc[i]
        if rec.get("image_bytes") is not None and img is not None and os.path.exists(img) and \
                float(rec["image_bytes"]) != float(os.path.getsize(img)):
            if rec.get("status") in ("ok", "check"):
                image_results.append(i)  # decided below, all together
            else:
                image_changed.append(i)
                todo.append(i)
            continue
        settle(i, rec)
    # One page image that changed is a page scanned again, and is extracted
    # again. Many at once are nearly always the same pages made again (another
    # computer, another version of the libraries, another dpi): paying for all
    # of them again needs a yes, or redo=.
    if image_results:
        if len(image_results) < MASS_IMAGE_CHANGE or images_changed_choice(image_results, pages) == "redo":
            for i in image_results:
                image_changed.append(i)
                todo.append(i)
                supersede_.append(files[i])
        else:
            for i in image_results:
                settle(i, records[keys[i]])
    if changed and run.dir is None:
        todo += changed
        changed = []
    if changed:
        alt = find_alternate(run, pages) if not done else None
        choice = {"action": "restart", "folder": alt} if alt else ask_changed(changed, pages, records, run)
        if choice["action"] == "redo":
            todo += changed
            supersede_ += [files[i] for i in changed]
        elif choice["action"] == "restart":
            return {"restart": choice["folder"]}
    todo = sorted(set(todo))
    skipped = 0
    if limit is not None and len(todo) > limit:
        skipped = len(todo) - int(limit)
        todo = todo[: int(limit)]
        if redo_mode == "all" and limit:
            say(("!", f"redo=\"all\" with limit redoes the first {plural(int(limit), 'page')} each time this line runs."),
                ("i", "To go on with the next pages, name them instead: redo= or only= with file names."))
    _inform_redone(len(practice), sum(1 for i in practice if i in todo),
                   "{n} held practice (mock/example) results; extracting {now} for real now")
    _inform_redone(len(image_changed), sum(1 for i in image_changed if i in todo),
                   "{n} changed since {it} extracted; extracting {now} again", image=True)
    todo_files = {files[i] for i in todo}
    return {"todo": todo, "done": done, "held": held, "supersede": [f for f in supersede_ if f in todo_files],
            "restart": None, "scope": scope, "skipped_by_limit": skipped,
            "in_job": sum(1 for i in scope if keys[i] in in_job)}


MASS_IMAGE_CHANGE = 3


def images_changed_choice(idx, pages) -> str:
    """Many page images changed at once: "redo" (asked, and answered yes) or
    "keep" (their earlier results stay; nothing is sent for them)."""
    n = len(idx)
    names = [pages["file"].iloc[i] for i in idx]
    head = f"{plural(n, 'page image')} changed since they were extracted (their files have other sizes): {listing(names, 3)}."
    why = ("So many at once usually means the images were made again (on another computer, with other versions of the "
           "libraries, or at another dpi or quality), not that the pages were scanned again.")
    if can_ask():
        say(("!", head), ("i", why))
        choice = menu([f"Keep the earlier results of these {n} pages (nothing is sent for them)",
                       f"Extract these {n} pages again (paid again; the earlier results are kept in superseded/)",
                       "Stop"], "What should happen?")
        if choice == 1:
            return "keep"
        if choice == 2:
            return "redo"
        raise ContractError("Stopped. Nothing was sent.")
    say(("!", head), ("i", why),
        ("i", "Their earlier results are kept and nothing is sent for them. To extract them again (paid again), name them "
              "in redo=, or use redo=\"all\"."))
    return "keep"


def _inform_redone(n: int, now: int, template: str, image: bool = False) -> None:
    if not n:
        return
    now_what = ("it" if n == 1 else "them") if now == n else ("none of them" if now == 0 else str(now))
    later = f" ({n - now} left for later)" if now < n else ""
    what = plural(n, "page image") if image else plural(n, "page")
    text = template.format(n=what, now=now_what, it="it was" if n == 1 else "they were")
    info(text + later + ".")


def ask_changed(changed, pages, records, run) -> dict:
    """Pages already extracted with other settings: ask (interactive) or stop."""
    s = run.settings
    recs = [records[page_key(pages["file"].iloc[i])] for i in changed]
    contracts = read_contracts(run.dir)
    new_entry = contract_entry(run.contract)
    what = []
    for cid in dict.fromkeys(r.get("contract") or "" for r in recs):
        if cid == run.contract["id"]:
            continue
        old = contracts.get(cid)
        what += describe_change(old, new_entry) if old else ["instructions or codebook"]
    old_models = list(dict.fromkeys(r.get("model") or "" for r in recs))
    other = [m for m in old_models if canonical_model(m) != canonical_model(s["model"])]
    if other:
        what.append(f"model: {', '.join(other)} -> {s['model']}")
    old_res = list(dict.fromkeys((r.get("settings") or {}).get("resolution") or "" for r in recs))
    if any(r != s["resolution"] for r in old_res):
        what.append(f"resolution: {', '.join(r for r in old_res if r != s['resolution'])} -> {s['resolution']}")
    old_th = list(dict.fromkeys((r.get("settings") or {}).get("thinking") or "" for r in recs))
    if any(t != s["thinking"] for t in old_th):
        what.append(f"thinking: {', '.join(t for t in old_th if t != s['thinking'])} -> {s['thinking']}")
    what += _extra_changes([(r.get("settings") or {}).get("temperature") for r in recs],
                           [(r.get("settings") or {}).get("api_args") for r in recs], s)
    what = list(dict.fromkeys(what)) or ["settings"]
    n = len(changed)
    new_folder = next_free_folder(run.dir)
    head = f"{plural(n, 'page')} in {run.dir} {'was' if n == 1 else 'were'} extracted with different settings:"
    if not can_ask():
        raise ContractError(head, *[("*", w) for w in what],
                            ("i", f"To keep those results and save the new ones apart: save_to=\"{new_folder}\" "
                                  "(afterwards this same line continues there too)."),
                            ("i", f"To redo {'that page' if n == 1 else 'those pages'} with the new settings: redo=\"all\" "
                                  "(old results are kept in superseded/)."))
    say(("!", head), *[("*", w) for w in what])
    choice = menu([f"Keep the old results and save the new ones in {new_folder}",
                   f"Redo these {n} pages with the new settings (old results kept in superseded/)",
                   "Stop"], "What should happen?")
    if choice == 1:
        info(f"The new results are saved in {new_folder}. Running this same line again continues there.")
        return {"action": "restart", "folder": new_folder}
    if choice == 2:
        return {"action": "redo"}
    raise ContractError("Stopped. Nothing was sent.")


def next_free_folder(folder: str) -> str:
    i = 2
    while os.path.exists(f"{folder}_{i}"):
        i += 1
    return f"{folder}_{i}"


def alternate_folders(folder: str) -> list[str]:
    base = os.path.basename(folder) + "_"
    parent = os.path.dirname(folder)
    try:
        cands = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    except OSError:
        return []
    cands = [d for d in cands if d.startswith(base) and d[len(base):].isdigit()]
    cands.sort(key=lambda d: int(d[len(base):]))
    return [os.path.join(parent, d) for d in cands]


def find_alternate(run: Run, pages):
    """The first <folder>_N that holds results (or a pending job) made with the
    current settings, or was started for them and holds nothing yet."""
    if run.dir is None:
        return None
    fp = run_fingerprint(run)
    for alt in alternate_folders(run.dir):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            recs = read_records(alt)
        try:
            jobs = read_jobs(alt)
        except InputError:
            jobs = []
        mine = {k: record_fingerprint(r) == fp for k, r in recs.items()}
        job_fp = [fingerprint(j.get("contract"), (j.get("settings") or {}).get("model"),
                              (j.get("settings") or {}).get("resolution"), (j.get("settings") or {}).get("thinking"),
                              (j.get("settings") or {}).get("temperature"), (j.get("settings") or {}).get("api_args"))
                  for j in jobs]
        started = not recs and not jobs and run.contract["id"] in read_contracts(alt)
        if any(mine.values()) or fp in job_fp or started:
            pkeys = {page_key(f) for f in pages["file"]}
            n_done = sum(1 for k, r in recs.items() if mine[k] and r.get("status") in ("ok", "check") and k in pkeys)
            alt = show_path(alt)
            info(f"Continuing in {alt}, which holds results made with these settings ({n_done} of {plural(len(pages), 'page')} done there). "
                 f"The results in {run.dir} were made with other settings and stay as they are.")
            info(f"To go there directly, use save_to=\"{alt}\".")
            return alt
    return None


def announce_plan(p: dict, run: Run) -> None:
    n_scope, n_done, n_todo = len(p["scope"]), len(p["done"]), len(p["todo"])
    if run.dir is None:
        where = ("Nothing is written to your computer (save_to=False): the results stay in Python. If the run stops, "
                 "running the same line again in this Python session continues it.")
    else:
        where = f"Results are saved in {run.dir}, one file per page, so an interrupted run can resume."
    done_keys = [page_key(run.pages["file"].iloc[i]) for i in p["done"]]
    n_sess = sum(1 for k in done_keys if k in run.from_session)
    n_cont = sum(1 for k in done_keys if k in run.from_continue)
    earlier = [x for x in (f"{fmt_int(n_sess)} read earlier in this Python session" if n_sess else None,
                           f"{fmt_int(n_cont)} kept from continue_from" if n_cont else None) if x]
    earlier = f" ({', '.join(earlier)}, not sent again)" if earlier else ""
    line = (f"{plural(n_scope, 'page')}: {fmt_int(n_done)} already done{earlier}, {fmt_int(n_todo)} to do"
            + (f", {p['in_job']} in a pending batch job" if p["in_job"] else "")
            + (f", {len(p['held'])} failed earlier and not retried" if p["held"] else "")
            + (f" ({p['skipped_by_limit']} more left for later: limit)" if p["skipped_by_limit"] else "") + ".")
    lines = [("i", line)]
    if n_todo:
        lines.append(("i", where))
    if p["supersede"] and run.dir is not None:
        k = len(p["supersede"])
        lines.append(("i", f"{plural(k, 'earlier result')} will be moved to superseded/ (kept, not deleted) as the new ones arrive."))
    say(*lines)
    if run.dir is None and n_todo >= options.long_run and run.settings["engine"] != "batch":
        suggest = show_path(default_results_dir(run.pages))
        info(f"A long run: if Python is closed or crashes, the pages read would be lost. To keep each page on disk as it "
             f"arrives (and resume even after closing Python), leave out save_to=False (the default folder is {suggest}).")


# ------- the page-by-page engine ------------------------------------------------------

def run_sync(run: Run, p: dict, workers: int = 1, budget=None, confirm: bool = False) -> None:
    s = run.settings
    todo = p["todo"]
    n = len(todo)
    log_manifest(run, n)
    ask_after = 2 if (confirm and n >= 25 and budget is None and not run.practice) else None
    bar = Progress("Page", n, status=lambda: f"{fmt_usd(run.spent)} so far")
    streak = {"n": 0, "why": "", "problem": ""}
    i = 0
    w = min(workers, n)
    try:
        while i < n:
            size = w
            if budget is not None:
                size, why = budget_round(run, budget, w)
                if size < 1:
                    run.stopped = why
                    break
            chunk = todo[i: i + size]
            outcomes, interrupted = send_pages(run, chunk)
            crash = None
            for o in outcomes:
                # a stop (key refused, quota used up) ends the run, but the
                # other replies of the same round were paid for: they are kept
                if o["action"] == "stop":
                    run.stopped = run.stopped or o["message"]
                    continue
                if o["action"] == "skip":
                    continue
                if o["action"] == "crash":
                    # raised once the rest of the round is kept
                    crash = crash or o["error"]
                    continue
                keep_record(run, o["record"])
                if o.get("error") is not None:  # a reply the package could not process: kept, then raised
                    crash = crash or o["error"]
                # the same failure page after page is not about one page: stop
                problem = o["record"].get("problem") or ""
                if o["record"]["status"] == "failed" and problem in ("server", "network", "image", "request"):
                    same = problem == streak["problem"] and (problem not in ("image", "request") or o["record"].get("message") == streak["why"])
                    streak["n"] = streak["n"] + 1 if same else 1
                    streak["problem"] = problem
                    streak["why"] = o["record"].get("message")
                else:
                    streak = {"n": 0, "why": "", "problem": ""}
            if crash is not None:
                raise crash
            if interrupted:
                raise KeyboardInterrupt
            # halve once per round in which Google answered 429
            if w > 1 and any(o.get("slow_down") for o in outcomes):
                w = max(1, w // 2)
                info(f"Google asked to slow down; sending {plural(w, 'page')} at a time from now on.")
            if run.stopped is not None:
                break
            if streak["n"] >= (3 if streak["problem"] in ("image", "request") else 5):
                run.stopped = (f"{streak['n']} pages in a row failed the same way ({streak['why']}). "
                               "Stopping so you can check before more pages fail; run the same line again later.")
                break
            bar.update(len(chunk))
            i += len(chunk)
            if ask_after is not None and i >= ask_after and i < n:
                ask_after = None
                if not confirm_projection(run, n - i):
                    run.stopped = "Stopped by you after the first pages; run the same line again to continue."
                    break
    except KeyboardInterrupt:
        read = len(run.written)
        run.stopped = f"Stopped by you after {read} of {plural(n, 'page')}. {how_to_continue(run)}"
        run.interrupted = True
    except Exception as e:  # a bug or a full disk: the pages already paid for stay
        read = len(run.written)
        if isinstance(e, AmanuensisError):  # a problem the package explains (a file it cannot write or read)
            run.stopped = (f"The run stopped after {read} of {plural(n, 'page')}: {short_message(e.head, 300)} "
                           f"{how_to_continue(run)}")
        else:
            run.stopped = (f"An unexpected error stopped the run after {read} of {plural(n, 'page')}: {short_message(str(e), 300)}. "
                           f"{how_to_continue(run)} Please report the error.")
        run.error = e
    finally:
        bar.close()


def page_input_cost(run):
    """What one page's input should cost before any page is measured: the
    image tokens of the resolution and the prompt (about 4 characters a
    token), at this run's price; None without a price."""
    s = run.settings
    price = price_for("gemini-3.8-flash" if s["model"] in PRACTICE_MODELS else s["model"],
                      discount=s["engine"] in ("batch", "flex"), prices=s.get("prices"))
    if price is None:
        return None
    tokens = (IMAGE_TOKENS[s["resolution"]] + len(run.contract["instructions"].full) / 4
              + len(to_json(run.contract["codebook"].schema)) / 4)
    return tokens * price["input"] / 1e6


def budget_round(run, budget, w: int):
    """(pages, why): how many pages the next round may send without going over
    the budget. Once pages are measured, as many as their average cost allows,
    at most ``w``. Before that, one page alone, so that the round with no
    measure yet is never multiplied by ``workers``; none when even its input
    would pass the budget, so budget=0 sends nothing."""
    left = budget - run.spent
    stop = f"Stopped before going over your budget of {fmt_usd(budget)} (spent {fmt_usd(run.spent)})."
    if run.n_priced > 0:
        avg = run.spent / run.n_priced
        if avg <= 0:
            return w, None
        k = min(w, int(math.floor(left / avg + 1e-9)))
        return (k, None) if k >= 1 else (0, stop)
    est = page_input_cost(run)
    if left <= 0 or (est is not None and est > left):
        about = f" (one page's input alone costs about {fmt_usd(est)})" if est is not None else ""
        return 0, f"Stopped before sending anything: your budget of {fmt_usd(budget)} is less than one page is likely to cost{about}."
    return 1, None


def how_to_continue(run: Run) -> str:
    if run.dir is not None:
        return f"The pages read are saved in {run.dir}; run the same line again to continue."
    return ("The pages read are kept for the rest of this Python session: run the same line again to read the others "
            "(pages already read are not sent again), or with limit=0 to get the result without sending anything.")


def confirm_projection(run: Run, remaining: int) -> bool:
    if run.n_priced == 0:
        return True
    avg = run.spent / run.n_priced
    half = avg / (2 if run.settings["engine"] == "sync" else 1)
    msg = (f"{plural(run.n_sent, 'page')} cost {fmt_usd(run.spent)}. The other {fmt_int(remaining)} would cost about "
           f"{fmt_usd(avg * remaining)}" + (f" (about {fmt_usd(half * remaining)} with batch=True)" if run.settings["engine"] == "sync" else "")
           + " at Google's paid prices. Continue?")
    return ask_yes_no(msg, default=True)


def _request_for(run: Run, i: int):
    """(contents, config) for one page. A page image that cannot be read stops
    the run with a plain message before anything of its round is sent."""
    s = run.settings
    pages = run.pages
    path = pages["image"].iloc[i]
    try:
        with open(path, "rb") as f:
            image = f.read()
    except OSError as e:
        raise InputError(f"Could not read the page image {show_path(path)}: {e.strerror or e}.",
                         "Nothing was sent for it in this round. If the images are on an external drive, a network share or in "
                         "a cloud folder (OneDrive, Dropbox, iCloud), check that the file is there and can be opened, then run "
                         "the same line again.") from None
    schema = schema_for_api(run.contract["codebook"].schema)
    headers = {"x-amanuensis-file": pages["file"].iloc[i]} if run.practice else None
    return generate_request(run.contract["instructions"].full, image, mime_of(path), schema, s, headers=headers)


def _perform(run: Run, i: int, retries: int, request=None):
    """One page's request: ("body", raw text) or ("error", exception). Any
    other exception is raised (the caller keeps it as that page's crash)."""
    contents, config = request if request is not None else _request_for(run, i)
    client = run.client()
    model = run.settings["model"]
    try:
        resp = call(lambda: client.models.generate_content(model=model, contents=contents, config=config),
                    retries=retries)
        return ("body", resp.sdk_http_response.body if resp.sdk_http_response is not None else "")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if is_api_error(e) or is_network_error(e) or type(e).__name__ == "NetworkError":
            return ("error", e)
        raise


def send_pages(run: Run, idx: list[int]):
    """Send a chunk of pages; one outcome per page and whether Ctrl-C came.

    Every request of the round is built (its image read) before any is sent,
    so a page that cannot be read stops the run before anything of the round
    is paid. Once requests are out, an unexpected exception of one page is
    that page's "crash" outcome, never raised here: the other replies of the
    round were paid for and are kept first."""
    s = run.settings
    interrupted = False
    requests = [_request_for(run, i) for i in idx]
    if len(idx) == 1:
        try:
            responses = [_perform(run, idx[0], s["retries"], requests[0])]
        except KeyboardInterrupt:
            raise
        except Exception as e:
            responses = [("crash", e)]
        slow = _is_429(responses[0])
    else:
        # a first round all at once, each request once; then the transient
        # failures one by one with the usual waits (not after Ctrl-C: those
        # pages are tried again on the next run)
        ex = cf.ThreadPoolExecutor(max_workers=len(idx))
        futures = [ex.submit(_perform, run, i, 0, req) for i, req in zip(idx, requests)]
        try:
            cf.wait(futures)
        except KeyboardInterrupt:
            interrupted = True
            try:
                cf.wait(futures)  # the requests already sent were paid for
            except KeyboardInterrupt:
                pass
        responses = []
        for fu in futures:
            if fu.done() and not fu.cancelled():
                try:
                    responses.append(fu.result())
                except KeyboardInterrupt:
                    interrupted = True
                    responses.append(None)
                except Exception as e:
                    responses.append(("crash", e))
            else:
                responses.append(None)
        ex.shutdown(wait=False, cancel_futures=True)
        slow = any(_is_429(r) for r in responses)
        if not interrupted and not any(r is not None and r[0] == "crash" for r in responses):
            for j, r in enumerate(responses):
                if r is not None and r[0] == "error" and _transient(r[1]):
                    try:
                        responses[j] = _perform(run, idx[j], s["retries"], requests[j])
                    except KeyboardInterrupt:
                        interrupted = True
                        break
                    except Exception as e:
                        responses[j] = ("crash", e)
                        break
        slow = slow or any(_is_429(r) for r in responses)
    out = []
    for j, i in enumerate(idx):
        r = responses[j]
        if r is None:  # never finished after Ctrl-C: no record, sent next time
            out.append({"action": "skip"})
            continue
        o = outcome(r, run, i)
        o["slow_down"] = slow
        out.append(o)
    return out, interrupted


def _is_429(r) -> bool:
    return r is not None and r[0] == "error" and is_api_error(r[1]) and getattr(r[1], "code", None) == 429


def _transient(e) -> bool:
    from ._gemini import TRANSIENT

    return (is_api_error(e) and getattr(e, "code", None) in TRANSIENT) or is_network_error(e)


def outcome(r, run: Run, i: int) -> dict:
    page = run.pages.iloc[i]
    s = run.settings
    kind, value = r
    if kind == "crash":
        return {"action": "crash", "error": value}
    if kind == "error":
        e = value
        if not is_api_error(e):
            return {"action": "keep", "record": failed_record(page, run, "network", str(e))}
        status = getattr(e, "code", 0) or 0
        msg = error_message(e)
        why = classify_failure(status, msg, model=s["model"], engine=s["engine"], details=error_details(e))
        if status == 400 and re.search(r"media_resolution|mediaresolution", msg.lower()):
            why = {"action": "stop", "message": f"Google rejected the resolution setting ({msg}). Use resolution=\"high\"."}
        if why["action"] == "stop":
            return {"action": "stop", "message": why["message"]}
        problem = why.get("problem") or ("server" if status >= 500 or status == 408 else "image")
        return {"action": "keep", "record": failed_record(page, run, problem, why["message"])}
    raw = value or ""
    try:
        reply = json.loads(raw)
    except ValueError:
        reply = None
    if not isinstance(reply, dict):
        return {"action": "keep", "record": failed_record(page, run, "server", "Google's reply could not be read.")}
    try:
        return {"action": "keep", "record": record_from_reply(reply, raw, page, run)}
    except Exception as e:
        # the reply was paid for: it is saved before the error stops the run
        return {"action": "keep", "record": internal_record(page, run, reply, raw, e), "error": e}


def internal_record(page, run: Run, reply: dict, raw: str, e: Exception, settings=None, contract_id=None) -> dict:
    """A reply the package could not turn into a record (a bug, or a schema
    the local check cannot handle): a failed record with problem "internal"
    that keeps Google's whole reply, its tokens and cost when they can be
    read. It is not sent again automatically by this version (should_retry)."""
    rec = failed_record(page, run, "internal",
                        f"amanuensis could not process Google's reply ({type(e).__name__}: {short_message(str(e), 300)}); "
                        "the reply is kept in this record. Please report this.", settings, contract_id)
    rec["reply"] = raw
    try:
        fill_usage(rec, read_reply(reply, raw), settings or run.settings)
    except Exception:
        pass
    return rec


# ------- turning a reply into a record --------------------------------------------------

def base_record(page, run: Run, settings=None, contract_id=None) -> dict:
    s = settings or run.settings
    path = page.get("image") if hasattr(page, "get") else None
    size = os.path.getsize(path) if isinstance(path, str) and os.path.exists(path) else None
    return {
        "amanuensis": __version__, "implementation": "python", "record_version": RECORD_VERSION,
        "file": page["file"], "source": page.get("source") if not is_missing(page.get("source")) else page["file"],
        "source_page": int(page.get("source_page")) if not is_missing(page.get("source_page")) else 1,
        "image_bytes": size, "status": None, "problem": None, "message": None, "notes": [], "errors": [],
        "attempts": 1, "contract": contract_id or run.contract["id"],
        "model": s["model"], "model_version": None,
        "settings": {"resolution": s["resolution"], "thinking": s["thinking"], "max_tokens": s["max_tokens"],
                     "temperature": "not sent" if s.get("temperature") is None else s["temperature"],
                     "engine": s["engine"], "schema_channel": s.get("schema_channel") or "json",
                     "api_args": s.get("api_args") or None},
        "batch_job": s.get("batch_job"),
        "finish_reason": None, "tokens": None, "price": None, "cost_usd": None, "other_attempts_cost_usd": None,
        "date": utc_now(), "reply": None, "data": None,
    }


def failed_record(page, run: Run, problem: str, message: str, settings=None, contract_id=None) -> dict:
    rec = base_record(page, run, settings, contract_id)
    rec["status"] = "failed"
    rec["problem"] = problem
    rec["message"] = short_message(message, 1000)
    return rec


_REPEAT_RE = re.compile(r"((?:[^\w\s]|_))\s?\1{20,}")


def fill_usage(rec: dict, r: dict, s: dict) -> None:
    """The model version, finish reason, tokens, price and cost of a reply."""
    rec["model_version"] = r["model_version"]
    rec["finish_reason"] = r["finish_reason"]
    tk = r["tokens"]
    if tk["total"] is not None and tk["input"] is not None:
        billed = tk["total"] - tk["input"]
    else:
        billed = sum(v for v in (tk["output"], tk["thinking"]) if v is not None)
    rec["tokens"] = {"input": tk["input"], "cached": tk["cached"], "image": r["image_tokens"], "thinking": tk["thinking_raw"],
                     "output": tk["output"], "billed_output": billed, "total": tk["total"]}
    discount = s["engine"] in ("batch", "flex")
    price = price_for("gemini-3.8-flash" if s["model"] in PRACTICE_MODELS else s["model"], discount=discount, prices=s.get("prices"))
    if price is not None:
        rec["price"] = {**price, "discount": discount, "date": today()}
        rec["cost_usd"] = cost({"input": tk["input"], "cached": tk["cached"], "output": billed, "thinking": 0}, price)


def record_from_reply(reply: dict, raw_body: str, page, run: Run, settings=None, contract_id=None, cb=None) -> dict:
    s = settings or run.settings
    cb = cb or run.contract["codebook"]
    rec = base_record(page, run, s, contract_id)
    r = read_reply(reply, raw_body)
    fill_usage(rec, r, s)

    def fail(problem, message, raw=r["text"]):
        rec.update(status="failed", problem=problem, message=message, reply=raw)
        return rec

    if r["block_reason"] is not None:
        return fail("blocked", f"Google blocked this page before reading it ({r['block_reason']}).", None)
    if r["finish_reason"] != "STOP":
        problem = "truncated" if r["finish_reason"] in ("MAX_TOKENS", "CONTINUATION") else "blocked"
        msg = finish_message(r["finish_reason"], s["max_tokens"])
        if problem == "truncated" and r["text"] is not None:
            cap = (model_info(s["model"]) or {}).get("max_output") or 65536
            if s["max_tokens"] >= cap:
                msg = (f"The reply was cut off at the model's limit of {fmt_int(cap)} tokens. " +
                       ("It repeats one punctuation mark over and over: add 'never write the same punctuation mark more than "
                        "three times in a row' to the instructions." if _REPEAT_RE.search(r["text"]) else
                        "The page holds too much text for one reply: split the image in two."))
        return fail(problem, msg)
    if r["nul"]:
        return fail("control_chars", "The reply contained NUL control characters, which would cut text short; the page will be tried again.")
    if r["text"] is None or not r["text"].strip():
        return fail("empty", "Google sent an empty reply.")
    try:
        data = json.loads(r["text"])
    except ValueError:
        return fail("not_json", "The reply was not valid JSON (it may have been cut short).")
    rec["reply"] = r["text"]
    rec["data"] = data
    errors = validate(data, cb.schema)
    rec["notes"] = reply_notes(data, r, s, cb)
    if errors:
        rec["status"] = "check"
        rec["problem"] = "invalid"
        rec["message"] = (f"The reply does not match the codebook: {errors[0]}" +
                          (f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""))
        rec["errors"] = errors
    else:
        rec["status"] = "ok"
    return rec


def _all_strings(x, path=""):
    if isinstance(x, str):
        return [(path or "value", x)]
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += _all_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x, 1):
            out += _all_strings(v, f"{path}[{i}]")
    return out


_CTRL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\t]")


def reply_notes(data, r: dict, s: dict, cb) -> list[str]:
    """Cheap mechanical checks on a reply, written as sentences."""
    notes = []
    strings = _all_strings(data)
    if strings:
        ctrl = [p for p, v in strings if _CTRL_RE.search(v)]
        if ctrl:
            notes.append(f"{len(ctrl)} value(s) contain a tab or control character (a backslash in the source may have been misread), e.g. {ctrl[0]}.")
        rep = [p for p, v in strings if _REPEAT_RE.search(v)]
        if rep:
            notes.append(f"The model may have got stuck repeating a mark in {rep[0]}.")
        sp = [p for p, v in strings if v and (v[0].isspace() or v[-1].isspace())]
        if sp:
            notes.append(f"{len(sp)} value(s) start or end with a space.")
    path = records_path(cb)
    if path is not None:
        rows = page_rows(data, path)
        if not rows:
            notes.append("No rows were found on this page.")
        else:
            empty = sum(1 for row in rows if all(v is None for v in row.values()))
            if empty:
                notes.append(f"{empty} row(s) are completely empty.")
    tk = r["tokens"]
    if tk["total"] is not None and tk["input"] is not None:
        billed = tk["total"] - tk["input"]
        if billed > 0.9 * s["max_tokens"]:
            notes.append(f"The reply used {round(100 * billed / s['max_tokens'])}% of max_tokens; a denser page would be cut off.")
    routes = (model_info(s["model"]) or {}).get("routes_to") or s["model"]
    mv = r["model_version"]
    if mv and s["model"] not in PRACTICE_MODELS and not mv.startswith(s["model"]) and not mv.startswith(routes):
        notes.append(f"Google answered with {mv} although {s['model']} was requested.")
    if s["resolution"] == "ultra_high" and r["image_tokens"] is not None and r["image_tokens"] < 1600 and s["model"] not in PRACTICE_MODELS:
        notes.append(f"Only {int(r['image_tokens'])} image tokens were counted: Google may have read the image at a lower resolution than ultra_high.")
    return notes


def _add_costs(*vals):
    """The sum of the costs that are known; None when none is."""
    known = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and not is_missing(v)]
    return sum(known) if known else None


def _hold_in_session(run: Run, k: str, rec: dict) -> None:
    """Keep a paid record in this Python session's store: the same line run
    again then saves it instead of sending the page again."""
    i = run.page_index(k)
    session_keep(rec, run.pages["image"].iloc[i] if i is not None else None, record_fingerprint(rec))


def _count(run: Run, rec: dict) -> None:
    """What a new record adds to this run's counts and spending."""
    run.statuses.append(rec["status"])
    if rec.get("batch_job") is not None and rec["batch_job"] in run.earlier_jobs:
        run.collected += 1
        return
    run.n_sent += 1
    c = rec.get("cost_usd")
    if isinstance(c, (int, float)) and not is_missing(c):
        run.spent += c
        run.n_priced += 1


def keep_record(run: Run, rec: dict) -> dict:
    """Store a new record (disk and memory) and log it.

    - The attempt count of a failed earlier record is carried over, and so is
      what the earlier attempts cost (``other_attempts_cost_usd``), so the
      costs shown count every paid attempt, not the last one only.
    - A page's earlier result (ok or to check) is replaced only by a new
      result. A redo that failed keeps it, in the folder and in the table:
      the failed attempt is logged and its cost added to the earlier record.
    - The earlier result is moved to superseded/ only here, when its
      replacement has arrived, so a run that stops early never leaves a page
      without its result; when it can be neither moved nor copied there, it
      is not written over and the run stops.
    - A record that cannot be written is held in this Python session's
      store before the error stops the run, so the same line run again
      saves it instead of paying for the page again."""
    k = page_key(rec["file"])
    old = run.records.get(k)
    if old is not None and old.get("status") == "failed" and rec.get("status") == "failed":
        rec["attempts"] = int(old.get("attempts") or 1) + 1
    has_result = old is not None and old.get("status") in ("ok", "check") and not is_practice(old)
    if has_result and rec.get("status") not in ("ok", "check"):
        run.redo_failed[old["file"]] = rec.get("message") or rec.get("problem") or "failed"
        if _add_costs(rec.get("cost_usd"), rec.get("other_attempts_cost_usd")):
            old["other_attempts_cost_usd"] = _add_costs(old.get("other_attempts_cost_usd"), rec.get("cost_usd"),
                                                        rec.get("other_attempts_cost_usd"))
            write_record(run.dir, old)
        _count(run, rec)
        log_page(run, rec)
        return rec
    if old is not None and is_practice(old) == is_practice(rec):
        rec["other_attempts_cost_usd"] = _add_costs(old.get("cost_usd"), old.get("other_attempts_cost_usd"),
                                                    rec.get("other_attempts_cost_usd"))
    if has_result and run.dir is not None:
        run.moved_to = run.moved_to or superseded_dir(run.dir, run.started)
        if not move_aside(run.dir, old["file"], run.moved_to):
            run.records[k] = rec
            _count(run, rec)
            raise InputError(f"The earlier result of {old['file']} could not be moved to {show_path(run.moved_to)}, so it "
                             "was not replaced (results are never deleted).",
                             "The earlier result is still in the results folder; the new reply is in the error's result "
                             "(e.result). Check that the folder can be written to (a sync client or an antivirus may be holding "
                             "the file), then run the same line again.")
        run.moved += 1
    run.records[k] = rec
    _count(run, rec)
    run.written.append(k)
    if run.dir is None:
        _hold_in_session(run, k, rec)
    else:
        try:
            write_record(run.dir, rec)
        except Exception:
            _hold_in_session(run, k, rec)
            raise
    log_page(run, rec)
    return rec


# ------- logs ------------------------------------------------------------------------------

def log_manifest(run: Run, n_images: int, extra: dict | None = None) -> None:
    if run.log_dir is None:
        return
    s = run.settings
    line = {"date": now(), "script": f"amanuensis {__version__} (Python)", "model": s["model"],
            "temperature": "not sent" if s.get("temperature") is None else s["temperature"],
            "media_resolution": media_level(s["resolution"]), "thinking_level": s["thinking_level"],
            "max_output_tokens": s["max_tokens"], "prompt_sha256": run.contract["instructions"].sha12,
            "schema_sha256": run.contract["codebook"].sha12, "images": n_images, "engine": s["engine"],
            "schema_channel": s.get("schema_channel") or "json", "api_args": s.get("api_args") or None,
            "python_version": platform.python_version()}
    line.update(extra or {})
    append_text(to_json(line), os.path.join(run.log_dir, "run_manifest.jsonl"))


def log_page(run: Run, rec: dict) -> None:
    if run.log_dir is None:
        return
    tk = rec.get("tokens")
    if tk:
        def fmt(v):
            return "None" if v is None else (str(int(v)) if float(v) == int(v) else str(v))
        append_text(f"{rec['file']}\tinput={fmt(tk.get('input'))}\tthinking={fmt(tk.get('thinking'))}\t"
                    f"output={fmt(tk.get('output'))}\ttotal={fmt(tk.get('total'))}",
                    os.path.join(run.log_dir, "token_usage.txt"))
    if rec.get("status") != "ok":
        append_text(f"{now()} {rec['file']} {rec.get('message') or rec.get('status')}", os.path.join(run.log_dir, "errors.log"))

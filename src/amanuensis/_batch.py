"""The Batch API: half price, results usually within minutes, at most 24 hours.

Lessons from real batch runs, built in here:

- a JSONL file uploaded through the Files API, not inline requests (the inline
  limit is 20 MB), written to a temporary file one line at a time;
- every request keyed by its image file name, and results matched by key,
  never by position (responses come back out of order);
- jobs of at most ``batch_size`` pages, submitted one at a time, and split
  further so no file passes Google's 2 GB limit;
- the job is recorded in batch-jobs.json *before* it is created ("submitting",
  with a unique display name) and again right after, so a crash in between
  never leaves an unknown paid job: the next run finds it by its display
  name. Creating a job is not idempotent, so the entry is dropped only when
  Google clearly refused the job (a 4xx); after a server error or a lost
  reply it stays until a complete listing of the account's jobs says whether
  the job exists;
- each job entry carries its own contract and settings, and its results are
  stamped with those, whatever the current call says;
- every transcribe() call collects pending jobs first and never sends their
  pages again; one job that cannot be collected never blocks the others;
- a job Google says it does not know (HTTP 404) is dropped, and its pages
  sent again, only when it is older than the weeks Google keeps jobs: a
  younger one most likely belongs to another project's key, and stays;
- per-line errors inside a successful job become failed pages, retried on the
  next run; the uploaded input file is deleted afterwards.
"""

from __future__ import annotations

import base64
import json
import math
import os
import signal
import tempfile
from datetime import datetime, timezone

from ._errors import ApiError, InputError, NetworkError
from ._gemini import (IMAGE_TOKENS, batch_body, batch_cancel, batch_create, batch_get, batch_results_file, batch_state,
                      download, file_delete, find_job_by_display, gemini_key, mime_of, schema_for_api, upload)
from ._models import PRACTICE_MODELS, cost, price_for
from ._options import ask_yes_no, can_ask, options, sleep
from ._say import attention, done, info, plural, say, warn
from ._store import contract_objects, job_files, read_contracts, read_jobs, read_records, write_jobs
from ._text import fmt_usd, page_key, sha12, show_path, to_json, utc_now


def poll_seconds(elapsed: float, model: str) -> float:
    if model in PRACTICE_MODELS:
        return options.poll_seconds if options.poll_seconds is not None else 0
    base = options.poll_seconds if options.poll_seconds is not None else 30
    return max(base, 120) if elapsed > 600 else base


def schema_channel(cb) -> str:
    """The package's own codebooks go through the OpenAPI channel (verified in
    v1beta batch); a JSON schema you wrote goes as it is through the JSON
    channel. ``options.batch_schema`` forces either."""
    if options.batch_schema in ("json", "openapi"):
        return options.batch_schema
    return "json" if cb.kind == "json" else "openapi"


def run_batch(run, plan: dict, batch_size: int = 500, wait: bool = True, budget=None) -> None:
    s = run.settings
    s["schema_channel"] = schema_channel(run.contract["codebook"])
    todo = plan["todo"]
    if todo:
        est = batch_estimate(run, len(todo))
        check_batch_budget(est, budget)
        info(f"Sending {plural(len(todo), 'page')} as a batch (half price): about {fmt_usd(est['cost'])}{est['note']} at Google's paid prices.")
        if budget is not None and not est["measured"] and est["cost"] is not None:
            info(f"Even if every reply ran to max_tokens, it would cost at most {fmt_usd(est['ceiling'])}, within your budget of {fmt_usd(budget)}.")
        if can_ask() and len(todo) >= 25 and budget is None and s["model"] not in PRACTICE_MODELS:
            if not ask_yes_no("Send the batch?", default=True):
                run.stopped = "Batch not sent."
                return
        chunks = batch_chunks(run, todo, batch_size)
        if len(chunks) > math.ceil(len(todo) / batch_size):
            info(f"The pages go in {len(chunks)} jobs, so that no batch file passes Google's 2 GB limit.")
        for j, chunk in enumerate(chunks, 1):
            share = {"total": budget, "share": budget * len(chunk) / len(todo)} if budget is not None else None
            if not submit_job(run, chunk, j, budget=share):
                left = sum(len(c) for c in chunks[j - 1:])
                run.stopped = (f"Google's batch queue for your account is full; {left} page(s) were not sent yet. "
                               "Run the same line again after the current job(s) finish.")
                break
    jobs = read_jobs(run.dir)
    run.pending_jobs = job_files(jobs)
    # nothing to wait for when every recorded job is one whose creation is not confirmed yet
    if not any(j.get("name") for j in jobs):
        return
    if not wait:
        info("Sent. Run the same line again later to collect the results; Google usually finishes within minutes, at most 24 hours.")
        return
    wait_for_jobs(run)


def check_batch_budget(est: dict, budget) -> None:
    """A batch cannot be stopped once it is sent, so the budget is checked once,
    before sending: against the cost measured on earlier pages, or, before any
    is measured, against the most the batch could cost (every reply running to
    max_tokens), since the input-only figure once let a batch spend several
    times its budget."""
    if budget is None or est["cost"] is None:
        return
    if est["measured"] and est["cost"] > budget:
        raise InputError(f"The batch would cost about {fmt_usd(est['cost'])}{est['note']}, more than your budget of {fmt_usd(budget)}.",
                         "A batch cannot be stopped once it is sent. Use `limit` to send fewer pages, or raise `budget`.")
    if not est["measured"] and est["ceiling"] > budget:
        raise InputError(f"No page of this folder has been measured yet, so this batch could cost up to {fmt_usd(est['ceiling'])} "
                         f"(if every reply ran to max_tokens), more than your budget of {fmt_usd(budget)}.",
                         "A batch cannot be stopped once it is sent. Run a few pages first (for example limit=5) so the cost per page "
                         "is measured, then run the same line again; or raise `budget`.")


def batch_estimate(run, n: int) -> dict:
    s = run.settings
    price = price_for("gemini-3.8-flash" if s["model"] in PRACTICE_MODELS else s["model"], discount=True, prices=s.get("prices"))
    if price is None:
        return {"cost": None, "ceiling": None, "measured": False, "note": " (cost unknown for this model)"}
    measured = [r for r in run.records.values() if r.get("tokens") and r.get("model") == s["model"]]
    if len(measured) >= 2:
        costs = [cost({"input": r["tokens"].get("input") or 0, "cached": r["tokens"].get("cached") or 0,
                       "output": r["tokens"].get("billed_output") or 0, "thinking": 0}, price) for r in measured]
        per = sum(costs) / len(costs)
        return {"cost": per * n, "ceiling": None, "measured": True, "note": ", measured on earlier pages"}
    image = IMAGE_TOKENS[s["resolution"]]
    prompt = len(run.contract["instructions"].full) / 4 + len(to_json(run.contract["codebook"].schema)) / 4
    inp = n * (image + prompt) * price["input"] / 1e6
    return {"cost": inp, "ceiling": inp + n * (s["max_tokens"] or 65536) * price["output"] / 1e6, "measured": False,
            "note": " for the input alone (the output adds more; run a few pages first with limit=5 to measure it)"}


def batch_chunks(run, todo: list[int], batch_size: int) -> list[list[int]]:
    """Jobs of at most batch_size pages whose files stay under the size limit;
    each line's size is known before anything is written."""
    limit = options.batch_max_bytes
    fixed = len(run.contract["instructions"].full.encode("utf-8")) + 2 * len(to_json(run.contract["codebook"].schema).encode("utf-8")) + 1000
    chunks, cur, size = [], [], 0
    for i in todo:
        try:
            b = math.ceil(os.path.getsize(run.pages["image"].iloc[i]) / 3) * 4 + fixed
        except OSError:
            b = fixed
        if cur and (len(cur) >= batch_size or size + b > limit):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += b
    if cur:
        chunks.append(cur)
    return chunks


def job_display_name(run, j: int) -> str:
    """A name no other job has: the folder, the second it was made, the job's
    number in this call, and a random part (two jobs made in the same second
    for one folder must not share it: an unclear creation is looked up by
    it)."""
    import secrets

    return f"amanuensis-{sha12(run.dir or 'memory')[:8]}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{j}-{secrets.token_hex(3)}"


def page_info(page) -> dict:
    return {"file": page["file"], "source": page.get("source") or page["file"],
            "source_page": int(page.get("source_page") or 1), "image": page.get("image")}


def write_batch_file(path: str, run, idx: list[int]) -> None:
    s = run.settings
    schema = schema_for_api(run.contract["codebook"].schema)
    with open(path, "wb") as f:
        for i in idx:
            img = run.pages["image"].iloc[i]
            with open(img, "rb") as fi:
                b64 = base64.b64encode(fi.read()).decode("ascii")
            body = batch_body(run.contract["instructions"].full, b64, mime_of(img), schema, s)
            f.write((to_json({"key": run.pages["file"].iloc[i], "request": body}) + "\n").encode("utf-8"))


def create_refused(e) -> bool:
    """True when Google clearly refused to create a job (a 4xx other than 408)."""
    st = getattr(e, "status", None)
    return isinstance(e, ApiError) and st is not None and 400 <= st < 500 and st != 408


def submit_job(run, idx: list[int], j: int, budget=None) -> bool:
    """Build, upload and create one job. True on success, False when Google's
    queue is full (429 at creation)."""
    s = run.settings
    display = job_display_name(run, j)
    # 1. the file, built and checked before anything is recorded or sent
    fd, jsonl = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        write_batch_file(jsonl, run, idx)
        size = os.path.getsize(jsonl)
        if size > options.batch_max_bytes:
            raise InputError(f"This batch file would be {size / 1e9:.1f} GB; Google accepts at most 2 GB.", "Use a smaller batch_size.")
        # 2. recorded as "submitting" before anything billable happens
        entry = {"name": None, "display_name": display, "state": "submitting", "input_file": None, "submitted": utc_now(),
                 "contract": run.contract["id"],
                 "settings": {k: s.get(k) for k in ("model", "resolution", "thinking", "thinking_level", "max_tokens",
                                                    "temperature", "engine", "schema_channel", "api_args", "retries", "timeout")},
                 "pages": [page_info(run.pages.iloc[i]) for i in idx]}
        if s.get("prices"):
            entry["settings"]["prices"] = dict(s["prices"])
        entry["budget"] = budget
        jobs = read_jobs(run.dir)
        jobs.append(entry)
        pos = len(jobs) - 1
        write_jobs(run.dir, jobs)
        # 3. upload, then create
        info(f"Uploading batch job {j}: {plural(len(idx), 'page')}, {round(size / 1e6)} MB.")
        client = run.client()
        stage = "upload"
        try:
            file_name = upload(client, jsonl, display_name=display)
            jobs[pos]["input_file"] = file_name
            write_jobs(run.dir, jobs)
            stage = "create"
            # creation and recording happen together: Ctrl-C waits until both are done
            with _hold_interrupts():
                created = None
                try:
                    created = batch_create(client, s["model"], file_name, display)
                finally:
                    if created is not None:
                        jobs[pos]["name"] = created.name
                        jobs[pos]["state"] = "submitted"
                        write_jobs(run.dir, jobs)
        except (ApiError, NetworkError) as e:
            if stage == "create" and not create_refused(e):
                # the entry and its input file stay: the next run looks the job up first
                raise type(e)(f"Creating batch job {j} ended without a clear answer from Google, so the job may or may not exist.",
                              "Its pages are not sent again until that is known: run the same line again in a few minutes, "
                              "and it first looks the job up by its name.", status=getattr(e, "status", None)) from e
            uploaded = jobs[pos].get("input_file")
            del jobs[pos]
            write_jobs(run.dir, jobs)
            if uploaded:
                file_delete(client, uploaded)
            if stage == "create" and (getattr(e, "status", None) == 429 or "RESOURCE_EXHAUSTED" in str(e) or "HTTP 429" in str(e)):
                return False
            raise
        from ._transcribe import log_manifest
        log_manifest(run, len(idx), extra={"batch_job": jobs[pos]["name"]})
        done(f"Batch job {j} created ({jobs[pos]['name']}): {plural(len(idx), 'page')}.")
        return True
    finally:
        if os.path.exists(jsonl):
            os.unlink(jsonl)


class _hold_interrupts:
    """Ctrl-C is held (and delivered afterwards) while a job is created and recorded."""

    def __enter__(self):
        self.got = False
        self.old = None
        try:
            self.old = signal.signal(signal.SIGINT, self._handler)
        except ValueError:  # not the main thread
            self.old = None
        return self

    def _handler(self, *a):
        self.got = True

    def __exit__(self, *exc):
        if self.old is not None:
            signal.signal(signal.SIGINT, self.old)
        if self.got:
            raise KeyboardInterrupt
        return False


def collect_pending_jobs(run, records: dict) -> None:
    """At the start of every transcribe(): collect finished jobs, and list the
    pages of unfinished ones so they are not sent again."""
    jobs = read_jobs(run.dir)
    if not jobs:
        return
    run.records = records
    info(f"{plural(len(jobs), 'batch job')} recorded in {run.dir}; checking {'it' if len(jobs) == 1 else 'them'} first.")
    jobs = recover_submitting(jobs, run)
    write_jobs(run.dir, jobs)
    poll_once(run, announce=True)
    run.records_on_disk = read_records(run.dir)


def recover_submitting(jobs: list, run) -> list:
    """A "submitting" entry without a job name: Python stopped between upload
    and creation, or Google's answer was unclear. Look the job up by its
    display name; drop the entry only when a complete listing says it is not
    there. When Google cannot be asked, the entry stays."""
    keep = []
    for job in jobs:
        if job.get("name"):
            keep.append(job)
            continue
        model = (job.get("settings") or {}).get("model") or ""
        n = len(job.get("pages") or [])
        try:
            found = find_job_by_display(run.client(model), job.get("display_name"))
        except Exception as e:
            say(("!", "A batch job was being created when an earlier run stopped, and Google could not be asked whether it exists."),
                ("x", str(e)),
                ("i", f"Its {plural(n, 'page')} {'is' if n == 1 else 'are'} not sent again until that is known; the next run asks again."))
            keep.append(job)
            continue
        if found:
            job["name"] = found
            job["state"] = "submitted"
            info(f"Found batch job {found}, created just before an earlier run stopped.")
            keep.append(job)
        else:
            if job.get("input_file"):
                file_delete(run.client(model), job["input_file"])
            info("A batch job was being created when an earlier run stopped, and Google has no record of it; its pages will be sent again.")
    return keep


def poll_once(run, announce: bool = False) -> int:
    """Check every recorded job once; collect those that ended. Returns the
    number of jobs worth waiting for."""
    jobs = read_jobs(run.dir)
    still, status, waiting = [], [], 0
    for job in jobs:
        if not job.get("name"):
            still.append(job)
            status.append(f"{job.get('display_name') or 'a batch job'}: not known yet whether Google created it")
            continue
        model = (job.get("settings") or {}).get("model") or ""
        try:
            got = batch_get(run.client(model), job["name"])
        except Exception as e:
            if isinstance(e, ApiError) and getattr(e, "status", None) == 404:
                # Google keeps a job and its results for weeks: a 404 for a job
                # sent more recently says more about the key (one of another
                # Google project) than about the job, and dropping the entry
                # would pay for its pages again while the job's results stay
                # out of reach
                age = job_age_days(job)
                if age is not None and age > RESULTS_KEPT_DAYS:
                    attention(f"Google no longer knows batch job {job['name']}, sent {round(age)} days ago; its pages will be sent again.")
                    continue
                still.append(job)
                # not counted as worth waiting for (the wait would never end),
                # and said once per run, not at every poll
                said = run.__dict__.setdefault("_unknown_jobs", set())
                if job["name"] in said:
                    continue
                said.add(job["name"])
                n = len(job.get("pages") or [])
                when = f"sent {_age_words(age)} ago" if age is not None else "of unknown age"
                say(("!", f"Google does not know batch job {job['name']} ({when}), although it keeps jobs for weeks."),
                    ("i", "GEMINI_API_KEY may now hold a key of another Google project than the one that sent the job. Run the "
                          "same line with the key that sent it to collect it."),
                    ("i", f"The job stays recorded and its {plural(n, 'page')} {'is' if n == 1 else 'are'} not sent again. If you "
                          f"know the job is gone, delete batch-jobs.json in {run.dir}: its pages are then sent again (and paid again)."))
                continue
            if not isinstance(e, (ApiError, NetworkError)):
                raise
            still.append(job)
            waiting += 1
            status.append(f"{job['name']}: could not check ({e})")
            continue
        state = batch_state(got)
        if state in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED"):
            try:
                finished = collect_job(run, job, got, "SUCCEEDED" if state == "PARTIALLY_SUCCEEDED" else state)
            except (ApiError, NetworkError, InputError, OSError, ValueError) as e:
                still.append(job)
                say(("!", f"Could not collect batch job {job['name']}; it stays recorded and is tried again next run."), ("x", str(e)))
                continue
            if finished is not True:
                still.append(job)
                if finished == "wait":
                    waiting += 1
        else:
            still.append(job)
            waiting += 1
            status.append(f"{job['name']}: {state.lower()}, {plural(len(job.get('pages') or []), 'page')} in the job")
    write_jobs(run.dir, still)
    run.pending_jobs = job_files(still)
    if announce and status:
        say(*[("*", x) for x in status])
    return waiting


def wait_for_jobs(run) -> None:
    import time

    start = time.monotonic()
    model = run.settings["model"]
    info("Waiting for Google. Batch jobs usually finish within minutes (at most 24 hours).")
    info("You can press Ctrl-C or close Python at any time: the job keeps running, and running this same line later collects the results.")
    try:
        while True:
            left = poll_once(run)
            if not left:
                break
            elapsed = time.monotonic() - start
            info(f"Waiting for {plural(left, 'batch job')} ({round(elapsed / 60)} min so far)")
            sleep(poll_seconds(elapsed, model))
    except KeyboardInterrupt:
        run.stopped = "Stopped waiting. The batch job keeps running at Google; run the same line later to collect it."
        run.interrupted = True
    run.records = read_records(run.dir)


def job_prices(p):
    """Prices stored with a job: a dict, or (older entries) a list read in the
    documented order input, output, cached."""
    if not p:
        return None
    if isinstance(p, dict):
        return p if "input" in p and "output" in p else None
    if isinstance(p, list) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p):
        return dict(zip(["input", "output", "cached"], p[:3]))
    return None


RESULTS_KEPT_DAYS = 42


def job_age_days(job):
    try:
        t = datetime.strptime(job.get("submitted") or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400


def _age_words(days: float) -> str:
    if days < 1 / 24:
        return f"{max(1, round(days * 1440))} min"
    if days < 2:
        return f"{round(days * 24)} hours"
    return f"{round(days)} days"


def collect_job(run, job: dict, got, state: str):
    """Turn a finished job into page records, stamped with the job's own
    contract and settings. True when done with; False to keep it recorded;
    "wait" when it is worth waiting for."""
    from ._transcribe import failed_record, internal_record, keep_record, record_from_reply

    settings = dict(job.get("settings") or {})
    settings["batch_job"] = job["name"]
    settings["flex"] = False
    settings["prices"] = job_prices(settings.get("prices"))
    settings.setdefault("engine", "batch")
    contracts = read_contracts(run.dir)
    entry = contracts.get(job.get("contract"))
    cb = contract_objects(entry, job["contract"])[0] if entry else run.contract["codebook"]
    pages = job.get("pages") or []
    keys = [p["file"] for p in pages]
    model = settings.get("model") or ""
    n = len(pages)
    client = run.client(model)
    again = "go back on the list of pages to send, and are paid again when they are sent"
    if state != "SUCCEEDED":
        err = getattr(got, "error", None)
        msg = getattr(err, "message", None) if err is not None else None
        why = {"EXPIRED": "Google did not finish the batch job within 48 hours",
               "CANCELLED": "the batch job was cancelled"}.get(state, f"the batch job failed ({msg or 'no reason given'})")
        for p in pages:
            keep_record(run, failed_record(p, run, "batch", f"{why}; the page will be sent again.", settings, job["contract"]))
        attention(f"Batch job {job['name']} ended as {state}: {why}. Its {plural(n, 'page')} {again}.")
    else:
        file = batch_results_file(got)
        if not file:
            attention(f"Batch job {job['name']} finished but Google gave no results file yet; checking again later.")
            return "wait"
        try:
            text = download(client, file)
        except (ApiError, NetworkError) as e:
            age = job_age_days(job)
            if getattr(e, "status", None) == 404 and age is not None and age > RESULTS_KEPT_DAYS:
                why = "Google no longer has this batch job's results (it keeps them for about 6 weeks)"
                for p in pages:
                    keep_record(run, failed_record(p, run, "batch", f"{why}; the page will be sent again.", settings, job["contract"]))
                attention(f"Batch job {job['name']}: {why}. Its {plural(n, 'page')} {again}.")
                if job.get("input_file"):
                    file_delete(client, job["input_file"])
                return True
            lines = [("!", f"Could not download the results of batch job {job['name']}; it stays recorded and is tried again next run, "
                           "so its pages are not paid for twice."), ("x", str(e))]
            if getattr(e, "status", None) == 404:
                lines.append(("i", f"If you know these results are gone, delete batch-jobs.json in {run.dir}: the job's pages are "
                                   "then sent again (and paid again)."))
            say(*lines)
            return False
        seen, spent = [], 0.0
        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            k = page_key(str(obj.get("key") or ""))
            hit = next((i for i, f in enumerate(keys) if page_key(f) == k), None)
            if hit is None:
                continue
            seen.append(keys[hit])
            p = pages[hit]
            resp = obj.get("response")
            if isinstance(resp, dict) and obj.get("error") is None:
                try:
                    rec = record_from_reply(resp, line, p, run, settings, job["contract"], cb)
                except Exception as e:  # paid for: kept, never lost
                    rec = internal_record(p, run, resp, line, e, settings, job["contract"])
            else:
                err = obj.get("error") or obj.get("status") or {}
                msg = err.get("message") if isinstance(err, dict) else None
                rec = failed_record(p, run, "batch", f"{msg or 'Google returned no reply for this page'}; the page will be sent again.",
                                    settings, job["contract"])
            if isinstance(rec.get("cost_usd"), (int, float)):
                spent += rec["cost_usd"]
            keep_record(run, rec)
        for p in pages:
            if p["file"] not in seen:
                keep_record(run, failed_record(p, run, "batch", "No reply in Google's results file; the page will be sent again.",
                                               settings, job["contract"]))
        done(f"Collected batch job {job['name']}: {len(seen)} of {plural(n, 'page')} returned.")
        warn_over_budget(job, spent)
    if job.get("input_file"):
        file_delete(client, job["input_file"])
    return True


def warn_over_budget(job: dict, spent: float) -> None:
    b = job.get("budget")
    if not isinstance(b, dict) or not isinstance(b.get("share"), (int, float)) or not isinstance(b.get("total"), (int, float)):
        return
    if spent <= b["share"] + 1e-9:
        return
    allowed = (f"your budget of {fmt_usd(b['total'])}" if abs(b["share"] - b["total"]) < 1e-9
               else f"its share of your budget ({fmt_usd(b['share'])} of {fmt_usd(b['total'])})")
    warn(f"Batch job {job['name']} cost {fmt_usd(spent)}, more than {allowed}.",
         "A batch cannot be stopped once it is sent: the budget was checked before sending, against the cost measured on "
         "earlier pages, and these pages cost more.")


def cancel_batch(save_to) -> list[str]:
    """Cancel the batch jobs recorded in a results folder.

    Ctrl-C while transcribe() waits for a batch job only stops the waiting:
    the job keeps running at Google, and you pay for it when it finishes.
    This cancels the jobs recorded in the folder (pages Google already
    finished may still be billed). The next transcribe() run on the folder
    records the cancellation and sends those pages again, paid again.
    Returns the names of the cancelled jobs.
    """
    from ._transcribe import Run

    if not isinstance(save_to, (str, os.PathLike)):
        raise InputError("`save_to` must be the results folder of the run.")
    save_to = os.path.expanduser(os.fspath(save_to))
    if not os.path.isdir(save_to):
        raise InputError(f"Cannot find the folder {show_path(save_to)}.")
    folder = show_path(save_to)
    jobs = read_jobs(folder)
    if not jobs:
        info(f"No pending batch jobs in {folder}.")
        return []
    cancelled, n_pages = [], 0
    clients = {}
    for job in jobs:
        model = (job.get("settings") or {}).get("model") or ""
        if model not in clients:
            from ._gemini import make_client
            if model in PRACTICE_MODELS:
                from ._practice import handler
                clients[model] = make_client("mock-key", handler=handler(model))
            else:
                clients[model] = make_client(gemini_key())
        client = clients[model]
        name = job.get("name")
        if not name:
            # created when an earlier run stopped, or with an unclear answer: look it up
            try:
                name = find_job_by_display(client, job.get("display_name"))
            except Exception as e:
                say(("!", f"Could not check whether batch job {job.get('display_name')!r} was created; run cancel_batch() again later."),
                    ("x", str(e)))
                continue
            if not name:
                continue
        ok, why = batch_cancel(client, name)
        if ok:
            cancelled.append(name)
            n_pages += len(job.get("pages") or [])
        else:
            attention(f"Could not cancel {name}: {why}")
    if cancelled:
        done(f"Asked Google to cancel {plural(len(cancelled), 'job')}: {', '.join(cancelled)}.")
        info(f"The next transcribe() run on this folder records the cancellation and sends the {plural(n_pages, 'cancelled page')} "
             "again, paid again (half price with batch=True, full price without). Run it when you want them extracted.")
    return cancelled

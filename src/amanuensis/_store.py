"""Where transcribe() keeps what it has read.

By default every page is saved as it arrives in a results folder next to the
page images (``<images folder>/extraction``), so a run that stops - Ctrl-C, a
crash, the budget, a closed laptop - resumes where it stopped and nothing is
paid twice. The folder holds

    <page>.json        one record per page image (status, settings, tokens,
                       cost, the model's verbatim reply and its parsed data)
    contract.json      the full instructions and codebook of every contract
                       used in this folder, keyed by contract id
    batch-jobs.json    only while a batch job is pending
    superseded/        earlier results replaced by a redo, one folder per run
                       (never deleted: they were paid for)
    transcribe.lock    only while a run is working in the folder
    run_manifest.jsonl, token_usage.txt, errors.log   the logs of the run

Every write is atomic (``_text.write_bytes_atomic``). The record format is the
R package's, so a folder can be continued by either.

``save_to=False`` keeps everything in memory instead: every record is then
held in the "session store" below until Python exits, so the same line run
again in the same session still continues instead of paying twice.
"""

from __future__ import annotations

import os
import re

from ._errors import InputError
from ._options import options
from ._say import info, plural, warn
from ._text import page_key, read_json, safe_stem, show_path, stem, stamp, utc_now, write_bytes_atomic, write_json
from ._version import __version__

RECORD_VERSION = 1
CONTROL_FILES = ("contract.json", "batch-jobs.json")
RESULTS_FOLDER_NAME = "extraction"


def record_file(folder, file) -> str:
    """A page's record file. A page called contract.jpg would write its record
    over contract.json: stems spelled like a control file get one more leading
    underscore (contract -> _contract, _contract -> __contract)."""
    s = safe_stem(stem(file))
    if re.sub(r"^_+", "", s).lower() + ".json" in CONTROL_FILES:
        s = "_" + s
    return os.path.join(folder, s + ".json")


def default_results_dir(pages) -> str:
    folder = getattr(pages, "folder", None) or os.path.dirname(pages["image"].iloc[0])
    return os.path.join(folder, RESULTS_FOLDER_NAME)


def write_record(folder, rec: dict) -> None:
    if folder is not None:
        write_json(rec, record_file(folder, rec["file"]), pretty=True)


def read_records(folder) -> dict:
    """Every page record of a folder, keyed by ``page_key(file)``.

    A record that was read and is not a record (damaged) is skipped, with one
    warning: its page will be redone. A record that cannot be opened at all
    right now (an antivirus or OneDrive lock, a cloud-only file while offline,
    a permission) is not damaged, and taking it for damaged would pay for the
    page again and write over it: that stops the run instead, before anything
    is sent."""
    if folder is None or not os.path.isdir(folder):
        return {}
    out, damaged, unreadable = {}, [], []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".json") or name.lower() in CONTROL_FILES or name.startswith("."):
            continue
        try:
            rec = read_json(os.path.join(folder, name))
        except OSError as e:
            unreadable.append((name, e))
            continue
        except ValueError:
            rec = None
        if not isinstance(rec, dict) or rec.get("file") is None or rec.get("status") is None:
            damaged.append(name)
            continue
        out[page_key(rec["file"])] = rec
    if unreadable:
        raise InputError(f"{plural(len(unreadable), 'result file')} in {show_path(folder)} could not be opened: "
                         f"{', '.join(n for n, _ in unreadable[:5])}{' ...' if len(unreadable) > 5 else ''}.",
                         ("x", str(unreadable[0][1].strerror or unreadable[0][1])),
                         "Nothing was sent. Close the program that is using them (Excel, an antivirus, a sync client), or make "
                         "them available offline (OneDrive: Always keep on this device), then run the same line again.")
    if damaged:
        warn(f"{plural(len(damaged), 'result file')} could not be read and will be redone: {', '.join(damaged[:5])}.")
    return out


# ------- contracts -----------------------------------------------------------------

def contract(instr, cb) -> dict:
    """The contract of a run: instructions plus codebook, and their id. The ids
    are taken again from what is sent: the objects are read-only, but a
    change made inside the codebook's schema dict (or to an object made by
    hand) would otherwise be sent under the id of what it was before."""
    from ._text import sha12, to_json

    if sha12(instr.full) != instr.sha12:
        raise InputError("The instructions were changed after instructions() made them, so their id no longer matches what "
                         "would be sent.", "Make them again with instructions(...) so the change is recorded with the results.")
    if sha12(to_json(cb.schema)) != cb.sha12:
        raise InputError("The codebook's schema was changed after codebook() made it, so its id no longer matches what would "
                         "be sent.", "Change the codebook file (or the DataFrame) and call codebook() again.")
    return {"id": f"{instr.sha12}-{cb.sha12}", "instructions": instr, "codebook": cb}


def contract_entry(c: dict) -> dict:
    cb, instr = c["codebook"], c["instructions"]
    return {
        "instructions": instr.full,
        "instructions_source": instr.source,
        "rules": instr.rules_id or "none",
        "schema": cb.schema,
        "codebook": [{"variable": v.variable, "type": v.type, "level": v.level, "description": v.description,
                      "values": list(v.values)} for v in cb.vars],
        "kind": cb.kind, "free_text": cb.free_text, "record": cb.record, "codebook_source": cb.source,
        "generator": f"amanuensis {__version__} (Python)",
        "first_used": utc_now(),
    }


def _load_contracts(folder):
    """(contracts, damaged): the contracts of contract.json, and whether the
    file exists but is not valid JSON. A file that cannot be opened raises:
    it is never taken for an empty one."""
    path = os.path.join(folder, "contract.json")
    if not os.path.exists(path):
        return {}, False
    try:
        x = read_json(path)
    except OSError as e:
        raise InputError(f"{show_path(path)} could not be opened.", ("x", str(e.strerror or e)),
                         "It holds the full instructions and codebook of every result in the folder. Nothing was sent: close the "
                         "program that is using it, or make it available offline, then run the same line again.") from None
    except ValueError:
        return {}, True
    if not isinstance(x, dict) or not isinstance(x.get("contracts") or {}, dict):
        return {}, True
    return x.get("contracts") or {}, False


def read_contracts(folder) -> dict:
    if folder is None:
        return {}
    return _load_contracts(folder)[0]


def save_contract(folder, c: dict) -> None:
    """Add the contract to contract.json. A contract.json that could not be
    opened stops (read raises); one that is damaged is copied aside before a
    new one is started, so the text it held is never lost."""
    if folder is None:
        return
    every, damaged = _load_contracts(folder)
    if c["id"] in every:
        return
    path = os.path.join(folder, "contract.json")
    if damaged:
        # not ending in .json, so it is never read as a page record
        to = os.path.join(folder, f"contract.json.damaged-{stamp()}")
        try:
            os.replace(path, to)
        except OSError:
            from ._text import unwritable
            raise unwritable(path) from None
        warn(f"{show_path(path)} was damaged (not valid JSON); it is kept as {os.path.basename(to)} and a new one is started.",
             "The pages already in the folder keep their contract ids; output(folder) needs the old file to rebuild them.")
    every[c["id"]] = contract_entry(c)
    write_json({"contracts": every}, path, pretty=True)


def contract_objects(entry: dict, cid: str):
    """The codebook and instructions rebuilt from a stored contract entry."""
    from ._codebook import Codebook, Variable
    from ._instructions import Instructions

    vars_ = [Variable(v.get("variable"), v.get("type"), v.get("level"), v.get("description") or "",
                      [str(x) for x in (v.get("values") or [])]) for v in entry.get("codebook") or []]
    ids = cid.split("-")
    cb = Codebook(vars=vars_, schema=entry.get("schema"), kind=entry.get("kind") or "table",
                  free_text=bool(entry.get("free_text")), record=entry.get("record"),
                  source=entry.get("codebook_source") or "contract.json", sha12=ids[1] if len(ids) > 1 else "")
    rules = entry.get("rules")
    instr = Instructions(text=entry.get("instructions") or "", rules=rules not in (None, "none"),
                         rules_id=rules or "none", full=entry.get("instructions") or "",
                         source=entry.get("instructions_source") or "contract.json", sha12=ids[0])
    return cb, instr


def describe_change(old: dict, new: dict) -> list[str]:
    """Plain words for what differs between two contract entries."""
    out = []
    if old.get("instructions") != new.get("instructions"):
        a = (old.get("instructions") or "").split("\n")
        b = (new.get("instructions") or "").split("\n")
        n = len(set(b) - set(a)) + len(set(a) - set(b))
        out.append(f"instructions: {n} line{'s' if n != 1 else ''} changed")
    vo = [v.get("variable") for v in old.get("codebook") or []]
    vn = [v.get("variable") for v in new.get("codebook") or []]
    added = [v for v in vn if v not in vo]
    removed = [v for v in vo if v not in vn]
    if added:
        out.append("codebook: added " + ", ".join(added))
    if removed:
        out.append("codebook: removed " + ", ".join(removed))
    if not added and not removed and old.get("schema") != new.get("schema"):
        out.append("codebook: descriptions, types or values changed")
    if old.get("kind") != new.get("kind") or bool(old.get("free_text")) != bool(new.get("free_text")):
        out.append("codebook: table/free text setting changed")
    return out or ["the package's standard wording changed"]


# ------- batch jobs ------------------------------------------------------------------

def jobs_path(folder) -> str:
    return os.path.join(folder, "batch-jobs.json")


def read_jobs(folder) -> list:
    if folder is None or not os.path.exists(jobs_path(folder)):
        return []
    try:
        x = read_json(jobs_path(folder))
    except (OSError, ValueError):
        raise InputError(f"The batch job list {jobs_path(folder)} is damaged.",
                         "If you know the job finished or was cancelled, delete that file and run again.") from None
    return (x or {}).get("jobs") or []


def write_jobs(folder, jobs: list) -> None:
    path = jobs_path(folder)
    if not jobs:
        if os.path.exists(path):
            os.unlink(path)
        return
    write_json({"jobs": jobs}, path, pretty=True)


def job_files(jobs) -> list[str]:
    return [p.get("file") or "" for j in jobs for p in j.get("pages") or []]


def pending_job_files(folder) -> list[str]:
    try:
        return job_files(read_jobs(folder))
    except InputError:
        return []


# ------- superseded records -------------------------------------------------------------

def superseded_dir(folder, when: str | None = None) -> str:
    return os.path.join(folder, "superseded", when or stamp())


def move_aside(folder, file, dest) -> bool:
    """Put the record of one page into ``dest`` (superseded/<run>/), never
    deleting it, and never writing over a record already there.

    Moved when it can be; when the move fails (on Windows an antivirus or
    OneDrive holding the file refuses it) the record is copied there instead,
    and the caller then writes the new record over the original. True when
    the earlier record is safe in ``dest`` (or there was none); False when it
    could be neither moved nor copied: it must then not be written over."""
    src = record_file(folder, file)
    if not os.path.exists(src):
        return True
    try:
        os.makedirs(dest, exist_ok=True)
    except OSError:
        return False
    to = os.path.join(dest, os.path.basename(src))
    k = 1
    while os.path.exists(to):
        to = os.path.join(dest, f"{stem(src)}-{k}.json")
        k += 1
    try:
        os.rename(src, to)
        return True
    except OSError:
        pass
    try:
        with open(src, "rb") as f:
            data = f.read()
        write_bytes_atomic(data, to)
    except (OSError, InputError):
        return False
    return True


def supersede(folder, files, dest=None, quiet: bool = False) -> int:
    """Move the records of these pages into superseded/<time>/ (see
    ``move_aside``); the number of records now kept there."""
    if folder is None or not files:
        return 0
    dest = dest or superseded_dir(folder)
    moved = 0
    for f in files:
        if os.path.exists(record_file(folder, f)) and move_aside(folder, f, dest):
            moved += 1
    if moved and not quiet:
        info(f"Moved {plural(moved, 'earlier result')} to {show_path(dest)} (kept, not deleted).")
    return moved


# ------- one run at a time in a folder ---------------------------------------------------

LOCK_NAME = "transcribe.lock"
LOCK_MAX_HOURS = 48
_HELD: set = set()  # locks this Python process holds now


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, owned by someone else
        return True
    except OSError:
        return False
    return True


def take_lock(folder) -> str | None:
    """Mark the results folder as in use by this run (``transcribe.lock``).

    Two runs on one folder (two terminals or kernels running the same line)
    would each read the folder at the start and each pay for every page, and
    the second could delete a temporary file the first is writing. A lock
    left by a run that is no longer there is taken over: its process is gone
    (same computer), or it is older than LOCK_MAX_HOURS (another computer
    sharing the folder). Returns the lock's path, for ``release_lock``."""
    import json as _json
    import socket
    import time

    path = os.path.join(folder, LOCK_NAME)
    me = {"pid": os.getpid(), "host": socket.gethostname(), "started": utc_now(), "time": time.time(),
          "by": f"amanuensis {__version__} (Python)"}
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            try:
                with open(path, encoding="utf-8") as f:
                    other = _json.load(f)
            except (OSError, ValueError):
                other = {}
            pid, host = other.get("pid"), other.get("host")
            try:
                age_h = (time.time() - float(other.get("time") or 0)) / 3600
            except (TypeError, ValueError):
                age_h = float("inf")
            same_host = host == me["host"]
            # this process's own lock is stale unless this process holds it now
            mine_left = pid == me["pid"] and show_path(path) not in _HELD
            stale = (not other or (same_host and isinstance(pid, int) and (mine_left or (pid != me["pid"] and not _alive(pid))))
                     or (not same_host and age_h > LOCK_MAX_HOURS))
            if stale and attempt == 0:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                info(f"Took over {LOCK_NAME} in {show_path(folder)}, left by a run that is no longer there.")
                continue
            where = f"process {pid} on this computer" if same_host else f"process {pid} on {host or 'another computer'}"
            raise InputError(f"Another transcribe() is working in {show_path(folder)} ({where}, started {other.get('started') or 'at an unknown time'}).",
                             "Two runs on one folder would both pay for the same pages. Wait for it to finish (or stop it), then run "
                             "this line again.",
                             f"If no other run is going on, delete {show_path(path)} and run the line again.")
        except OSError:
            from ._text import unwritable
            raise unwritable(path) from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_json.dumps(me))
        _HELD.add(show_path(path))
        return path
    return None  # pragma: no cover - the second attempt either takes the lock or raises


def release_lock(path) -> None:
    if path:
        _HELD.discard(show_path(path))
        try:
            os.unlink(path)
        except OSError:
            pass


# ------- the session store ------------------------------------------------------------------

# Every page record of a run without a results folder is also kept here until
# Python exits, keyed by the run's fingerprint (contract, model, resolution,
# thinking) and the image's full path, so two folders that both hold p001.jpg
# never share a record. At most SESSION_MAX_RUNS fingerprints are kept (the
# oldest is dropped); options.session = False turns the store off.
_SESSION: dict = {}
SESSION_MAX_RUNS = 20


def session_keep(rec: dict, image, fingerprint: str) -> None:
    if not options.session or image is None:
        return
    if fingerprint not in _SESSION:
        _SESSION[fingerprint] = {}
        while len(_SESSION) > SESSION_MAX_RUNS:
            _SESSION.pop(next(iter(_SESSION)))
    _SESSION[fingerprint][show_path(image)] = rec


def session_records(fingerprint: str, pages) -> dict:
    if not options.session:
        return {}
    held = _SESSION.get(fingerprint) or {}
    out = {}
    for img, file in zip(pages["image"], pages["file"]):
        rec = held.get(show_path(img))
        if rec is not None:
            out[page_key(file)] = rec
    return out


def session_reset() -> None:
    """Forget every page this Python session read without a results folder."""
    _SESSION.clear()

"""check_setup(): one look at everything a first session needs. It spends
nothing and changes nothing: the key is checked by asking Google for its list
of models (free), and the only thing written is a one-byte temporary file in
the working folder, deleted at once (permission checks cannot be trusted on
Windows or in synced folders, which is exactly where saving fails).

Each check is one line: OK, NOTE (works, but worth knowing) or PROBLEM
(something will fail), and every PROBLEM says how to fix it.
"""

from __future__ import annotations

import locale
import os
import sys
import tempfile
import textwrap
from importlib import metadata

import pandas as pd

from ._errors import InputError, MissingKeyError

# what pyproject.toml requires, for when the package metadata cannot be read
_REQUIRED = {"google-genai": "2.31", "httpx": "0.27", "numpy": "1.26", "openpyxl": "3.1", "pandas": "2.2",
             "pillow": "10.0", "pypdfium2": "4.25"}
_MODULES = {"google-genai": "google.genai", "httpx": "httpx", "numpy": "numpy", "openpyxl": "openpyxl", "pandas": "pandas",
            "pillow": "PIL", "pypdfium2": "pypdfium2"}


class SetupCheck(pd.DataFrame):
    """One row per check: check, status (OK, NOTE, PROBLEM), detail, fix.

    check_setup() prints the report itself, so the object shows only one line
    when a notebook or the Python prompt displays it (the report is not shown
    twice); ``print(x.report())`` shows it again and ``pd.DataFrame(x)`` is the
    plain table. Selections of it are plain DataFrames."""

    @property
    def _constructor(self):
        return pd.DataFrame

    def _summary(self) -> str:
        n_p = int((self["status"] == "PROBLEM").sum())
        n_n = int((self["status"] == "NOTE").sum())
        return (f"<amanuensis setup check: {len(self)} checks, {n_p} problem{'s' if n_p != 1 else ''}, "
                f"{n_n} note{'s' if n_n != 1 else ''}; print(x.report()) for the report, pd.DataFrame(x) for the table>")

    def __repr__(self) -> str:
        return self._summary()

    def _repr_html_(self):
        import html as _html
        return f"<p>{_html.escape(self._summary())}</p>"

    def report(self) -> str:
        lines = ["amanuensis setup check"]
        w = max([len(c) for c in self["check"]] + [14]) + 2
        indent = " " * (2 + w + 9)
        for _, r in self.iterrows():
            text = r["detail"] + (f" Fix: {r['fix']}" if r["fix"] and r["status"] != "OK" else "")
            wrapped = textwrap.wrap(text, width=max(78 - len(indent), 30)) or [""]
            lines.append(f"  {r['check']:<{w}}{r['status']:<9}{wrapped[0]}")
            lines += [indent + x for x in wrapped[1:]]
        n_p = int((self["status"] == "PROBLEM").sum())
        n_n = int((self["status"] == "NOTE").sum())
        if not n_p:
            lines.append("Everything needed is in place" + (f" ({n_n} note{'s' if n_n > 1 else ''} above)" if n_n else "") + ".")
        else:
            lines.append(f"{n_p} problem{'s' if n_p > 1 else ''} to fix (lines marked PROBLEM)" + (f", and {n_n} note{'s' if n_n > 1 else ''}" if n_n else "") + ".")
        return "\n".join(lines)


def _row(check, status, detail, fix=""):
    return {"check": check, "status": status, "detail": detail, "fix": fix}


def check_setup(key: bool = True, online: bool = True) -> SetupCheck:
    """Check that everything is ready: Python, the packages amanuensis relies on
    (and which image formats Pillow reads), whether GEMINI_API_KEY is set,
    whether Google can be reached and accepts the key, and whether you can
    save files in your working folder. Nothing is changed or paid for.

    Prints one line per check (OK, NOTE or PROBLEM, with how to fix it) and
    returns the table. ``key=False`` skips the key (practice with
    ``model="mock"``); ``online=False`` does not contact Google.
    """
    if not isinstance(key, bool):
        raise InputError("`key` must be True or False.")
    if not isinstance(online, bool):
        raise InputError("`online` must be True or False.")
    rows = [_python(), _encoding()] + _packages()
    if key:
        rows += _key(online)
    elif online:
        rows.append(_reachable(None))
    rows += [_folder(), _synced()]
    out = SetupCheck(rows, columns=["check", "status", "detail", "fix"])
    print(out.report())
    return out


def _python():
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        return _row("Python", "OK", f"Python {s}")
    return _row("Python", "PROBLEM", f"Python {s} is older than 3.10, which the package needs.",
                "Install a current Python from https://www.python.org, then install amanuensis again.")


def _encoding():
    enc = (getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False) or "").lower()
    if "utf" in enc and "8" in enc:
        return _row("text encoding", "OK", "UTF-8")
    return _row("text encoding", "NOTE", f"The console is not UTF-8 ({enc or 'unknown'}): accented names may look wrong there. Files are still saved correctly.",
                "Set PYTHONUTF8=1 in the terminal (export PYTHONUTF8=1), or use Python's UTF-8 mode (python -X utf8).")


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _ok(have, need) -> bool:
    if have is None:
        return False
    if not need:
        return True

    def parts(v):
        out = []
        for p in v.split("."):
            digits = "".join(ch for ch in p if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    return parts(have) >= parts(need)


def _required():
    try:
        reqs = metadata.requires("amanuensis") or []
    except metadata.PackageNotFoundError:
        return dict(_REQUIRED)
    out = {}
    for r in reqs:
        if ";" in r:
            continue
        name = r.split(">=")[0].split("==")[0].split("<")[0].strip().lower()
        out[name] = r.split(">=")[1].split(",")[0].strip() if ">=" in r else ""
    return out or dict(_REQUIRED)


def _packages():
    need = _required()
    rows = []
    for dist in ("pypdfium2", "pillow", "openpyxl", "pandas", "google-genai", "httpx", "numpy"):
        have = _version(dist)
        n = need.get(dist, _REQUIRED.get(dist, ""))
        label = {"pillow": "Pillow"}.get(dist, dist)
        if have is None:
            rows.append(_row(label, "PROBLEM", "not installed.", f"In the terminal: pip install \"{dist}>={n}\"" if n else f"pip install {dist}"))
        elif not _ok(have, n):
            rows.append(_row(label, "PROBLEM", f"{have} is older than the {n} the package needs.", f"In the terminal: pip install --upgrade \"{dist}>={n}\""))
        else:
            rows.append(_row(label, "OK", have))
        if dist == "pillow" and have is not None:
            rows.append(_formats())
    for dist, what in (("rapidfuzz", "for faster scoring of long free-text pages"),):
        have = _version(dist)
        rows.append(_row(dist, "OK", f"{have} ({what})") if have else
                    _row(dist, "NOTE", f"not installed; optional, {what}.", f"If you need it: pip install {dist}"))
    return rows


def _formats():
    try:
        from PIL import features
    except ImportError:  # pragma: no cover
        return _row("image formats", "PROBLEM", "Pillow could not report which formats it reads.", "pip install --upgrade pillow")
    have = {"JPEG": features.check("jpg"), "TIFF": features.check("libtiff") or True, "WEBP": features.check("webp"),
            "JPEG 2000": features.check("jpg_2000")}
    heic = _version("pillow-heif") is not None or (sys.platform == "darwin" and os.path.exists("/usr/bin/sips"))
    have["HEIC"] = heic
    if not have["JPEG"]:
        return _row("image formats", "PROBLEM", "Pillow cannot read or write JPEG, which every page image needs.", "pip install --upgrade pillow")
    if all(have.values()):
        return _row("image formats", "OK", ", ".join(have) + (" (HEIC through macOS)" if _version("pillow-heif") is None else ""))
    missing = [k for k, v in have.items() if not v]
    fix = "pip install pillow-heif" if missing == ["HEIC"] else "Only matters for sources in those formats: convert them to JPEG or PNG with another program."
    return _row("image formats", "NOTE", f"Pillow reads {', '.join(k for k, v in have.items() if v)} but not {', '.join(missing)} "
                                         "(HEIC is the format of iPhone photos).", fix)


def _key(online):
    from ._gemini import check_key_shape, key_how_to, mask_key

    gem = os.environ.get("GEMINI_API_KEY", "").strip()
    goo = os.environ.get("GOOGLE_API_KEY", "").strip()
    rows = []
    if not gem:
        detail = "GEMINI_API_KEY is not set" + (" (GOOGLE_API_KEY is, but amanuensis reads only GEMINI_API_KEY)" if goo else "") + "."
        rows.append(_row("Gemini key", "PROBLEM", detail,
                         key_how_to() + " Never write the key in a script. To practise without a key, use model=\"mock\" or \"example\"."))
        if online:
            rows.append(_reachable(None))
        return rows
    try:
        clean = check_key_shape(gem)
        shape = None
    except MissingKeyError as e:
        clean, shape = None, e.head
    if shape:
        rows.append(_row("Gemini key", "PROBLEM", shape, "Set only the key, copied from https://aistudio.google.com/apikey. " + key_how_to()))
    else:
        quoted = " (the quotes around it are left out)" if clean != gem else ""
        rows.append(_row("Gemini key", "OK", f"{mask_key(clean)}, from GEMINI_API_KEY{quoted}"))
        gem = clean
    if goo and goo != gem:
        rows.append(_row("second key", "NOTE", f"GOOGLE_API_KEY is also set, to a different key ({mask_key(goo)}); amanuensis uses only GEMINI_API_KEY.",
                         "Nothing to do if that is intended."))
    if online:
        rows.append(_reachable(None if shape else gem))
    return rows


def _reachable(key):
    from . import _gemini
    from ._options import options

    base = options.base_url or _gemini.DEFAULT_BASE_URL
    host = base.split("//", 1)[-1].split("/", 1)[0]
    unreachable = _row("Google", "PROBLEM", f"Could not reach Google ({host}).",
                       f"Check the internet connection. On a university or office network, ask whether a proxy or firewall blocks {host}.")
    if key is None:
        # no key to check: only whether Google answers at all (any HTTP reply
        # counts). The one request not made through google-genai: it carries
        # no key and calls no model.
        import httpx
        if _gemini._MOCKS:
            client = httpx.Client(transport=httpx.MockTransport(_gemini._MOCKS[-1]))
        elif not _gemini.ALLOW_NETWORK:
            raise AssertionError("a test tried to reach Google")
        else:
            client = httpx.Client(timeout=20)
        try:
            with client:
                client.get(base.rstrip("/") + "/v1beta/models")
        except httpx.HTTPError:
            return unreachable
        return _row("Google", "OK", "reachable (no key to check)")
    ok, why = _gemini.check_key_online(key)
    if ok is True:
        return _row("Google", "OK", "reachable, and it accepts the key")
    if ok is False:
        return _row("Google", "PROBLEM", f"Google refused the key ({why}).",
                    "Copy the key again from https://aistudio.google.com/apikey and export it again. With a university Google account, "
                    "AI Studio may be switched off by the administrator: try a personal account.")
    return unreachable


def _folder():
    wd = os.getcwd()
    try:
        fd, tmp = tempfile.mkstemp(prefix=".amanuensis-check-", dir=wd)
        os.write(fd, b"\x01")
        os.close(fd)
        os.unlink(tmp)
        return _row("working folder", "OK", f"{wd} (you can save files there)")
    except OSError:
        return _row("working folder", "PROBLEM", f"Cannot save files in your working folder, {wd}.",
                    "Work in a folder of your own: cd into it in the terminal before starting Python, or os.chdir('/path/to/project').")


def _synced():
    wd = os.path.realpath(os.getcwd())
    low = wd.lower()
    service = None
    if "onedrive" in low:
        service = "OneDrive"
    elif "dropbox" in low:
        service = "Dropbox"
    elif "/library/mobile documents/" in low or "icloud" in low:
        service = "iCloud Drive"
    elif "google drive" in low or "googledrive" in low or "/my drive" in low:
        service = "Google Drive"
    if service is None:
        for var in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
            root = os.environ.get(var)
            if root and low.startswith(os.path.realpath(root).lower()):
                service = "OneDrive"
    if service is None:
        return _row("synced folder", "OK", "not inside OneDrive, Dropbox, iCloud Drive or Google Drive")
    return _row("synced folder", "NOTE", f"Your working folder is inside {service}, which can lock a file for a moment while it syncs it.",
                f"If saving ever fails with \"permission denied\", run the same line again, pause syncing during long runs, or work in a folder outside {service}.")

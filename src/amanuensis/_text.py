"""Small helpers shared by every module: text files, JSON, CSV, hashing,
names and natural order, number formatting and argument checks.

Policy, carried over from the R package:

- Every file the package writes goes through ``write_bytes_atomic()``: written
  to a temporary file in the same folder, then moved into place in one step
  (``os.replace``), so a crash or Ctrl-C never leaves half a file and the old
  file is never deleted first. The per-page records are what resuming relies
  on, and ``batch-jobs.json`` may stand for a paid job.
- Text is read as UTF-8 (BOM stripped, CRLF -> LF); UTF-16/32 files with a
  byte-order mark are decoded; a file that is not UTF-8 is read as
  Windows-1252 with a warning rather than refused.
- ``to_json()`` writes the compact form the contract hash is taken from. It is
  the form jsonlite gives in R (no spaces, UTF-8 kept, whole numbers without
  a decimal point, other numbers to 15 significant digits), so a codebook
  hashes the same in both packages.
"""

from __future__ import annotations

import csv
import decimal
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from ._errors import InputError, InternalError
from ._options import options

# ------- names ----------------------------------------------------------------

_EXT_RE = re.compile(r"\.([^\W_]+)$")
_STEM_RE = re.compile(r"([^.]+)\.[^\W_]+$")


def ext(path) -> str:
    """The extension, lower case, without the dot ("" when there is none)."""
    m = _EXT_RE.search(os.path.basename(str(path)))
    return m.group(1).lower() if m else ""


def stem(path) -> str:
    """The file name without its extension."""
    return _STEM_RE.sub(r"\1", os.path.basename(str(path)))


def user_path(x):
    """A path the user gave, with "~" expanded (other values unchanged)."""
    if isinstance(x, os.PathLike) or (isinstance(x, str) and x.startswith("~")):
        return os.path.expanduser(os.fspath(x))
    return x


def show_path(path) -> str:
    """The absolute path, links resolved: what messages show and records keep."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def nfc(x):
    if x is None:
        return None
    return unicodedata.normalize("NFC", str(x))


def page_key(file) -> str:
    """The key pages are matched by: case- and Unicode-insensitive (macOS and
    Windows file systems are case-insensitive, and a sync between them can
    turn an accented name from NFC into NFD)."""
    return nfc(file).lower()


_FOLD_FROM = "áàäâãéèëêíìïîóòöôõúùüûñçÁÀÄÂÃÉÈËÊÍÌÏÎÓÒÖÔÕÚÙÜÛÑÇ"
_FOLD_TO = "aaaaaeeeeiiiiooooouuuuncAAAAAEEEEIIIIOOOOOUUUUNC"
_FOLD_TABLE = str.maketrans(_FOLD_FROM, _FOLD_TO)


def fold_accents(x: str) -> str:
    """Latin accents folded (the letters of Spanish, Catalan, French, Portuguese)."""
    return nfc(x).translate(_FOLD_TABLE)


def norm_header(x) -> str:
    """A column header for matching: lower case, accents folded, ``_`` and ``.``
    as spaces, spacing collapsed."""
    x = fold_accents("" if x is None else str(x)).lower()
    x = re.sub(r"[_.]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def safe_stem(x: str) -> str:
    """Characters Windows cannot have in a file name replaced, and its reserved names avoided."""
    x = re.sub(r'[<>:"/\\|?*\x01-\x1f]', "_", str(x))
    x = re.sub(r"[. ]+$", "", x)
    if x.upper() in _WIN_RESERVED:
        x += "_"
    return x


def _natural_key(name: str, fold) -> str:
    key = fold(name)
    parts = re.split(r"([0-9]+)", key)
    out = []
    for p in parts:
        if p and p.isdigit() and p.isascii():
            # as R's formatC(as.numeric(p), width = 15, format = "d", flag = "0")
            out.append("%015.0f" % float(p))
        else:
            out.append(p)
    return "".join(out)


def natural_key(name: str) -> str:
    """Sort key for natural order: "p2" before "p10", ignoring case."""
    return _natural_key(str(name), str.lower)


def natural_order(names) -> list[int]:
    """Positions of ``names`` in natural order (stable)."""
    names = [str(n) for n in names]
    return sorted(range(len(names)), key=lambda i: natural_key(names[i]))


def natural_sorted(names) -> list[str]:
    names = list(names)
    return [names[i] for i in natural_order(names)]


# ------- text files -------------------------------------------------------------

def file_bom(x) -> str | None:
    """The byte-order mark a file (a path, or its first bytes) starts with."""
    if isinstance(x, (str, os.PathLike)):
        p = str(x)
        if not os.path.isfile(p):
            return None
        with open(p, "rb") as f:
            x = f.read(4)
    b = bytes(x)
    if b.startswith(b"\xff\xfe\x00\x00"):
        return "UTF-32LE"
    if b.startswith(b"\x00\x00\xfe\xff"):
        return "UTF-32BE"
    if b.startswith(b"\xef\xbb\xbf"):
        return "UTF-8"
    if b.startswith(b"\xff\xfe"):
        return "UTF-16LE"
    if b.startswith(b"\xfe\xff"):
        return "UTF-16BE"
    return None


def read_text(path, hint: str | None = None) -> str:
    """A whole text file as one string: UTF-8, BOM stripped, CRLF -> LF."""
    from ._say import warn

    data = Path(path).read_bytes()
    hint = hint or "Save it as plain text (UTF-8) and try again."
    bom = file_bom(data)
    if bom and bom != "UTF-8":
        skip = 4 if bom.startswith("UTF-32") else 2
        try:
            text = data[skip:].decode(bom.lower().replace("le", "-le").replace("be", "-be"))
        except (UnicodeDecodeError, LookupError):
            raise InputError(f"{os.path.basename(str(path))} is saved as {bom} text but could not be read as such.", hint)
        return re.sub(r"\r\n?", "\n", text)
    if bom == "UTF-8":
        data = data[3:]
    if b"\x00" in data:
        raise InputError(f"{os.path.basename(str(path))} does not look like a text file.", hint)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
        warn(f"{os.path.basename(str(path))} is not saved as UTF-8; read it as Windows-1252.",
             "Check that accented letters look right, or re-save the file as UTF-8.")
    return re.sub(r"\r\n?", "\n", text)


def _rename(src: str, dst: str) -> bool:
    try:
        os.replace(src, dst)
        return True
    except OSError:
        return False


def _copy(src: str, dst: str) -> bool:
    try:
        shutil.copyfile(src, dst)
        return True
    except OSError:
        return False


def unwritable(path) -> InputError:
    return InputError(f"Could not write {show_path(path)}.",
                      "If the file is open in Excel or Word, close it and run the line again.",
                      "Otherwise check that the folder exists and that you may write to it.")


def _current_umask() -> int:
    """The process's umask (read once, at import, before any thread of the
    package runs: reading it means setting it)."""
    m = os.umask(0o022)
    os.umask(m)
    return m


_UMASK = _current_umask()


def _file_mode(path: str) -> int:
    """The permissions a written file gets: those of the file it replaces, else
    what open() would give a new file (0o666 less the umask). The temporary
    file mkstemp makes is owner-only (0o600), and a rename keeps that, so
    records, outputs and page images would be unreadable to colleagues on a
    shared server or group folder."""
    import stat

    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return 0o666 & ~_UMASK


def write_bytes_atomic(data: bytes, path, mode: int | None = None) -> str:
    """Write ``data`` to ``path`` through a temporary file and one rename.

    The old file is never deleted first: a rename replaces it in one step on
    every system, so until the new file is in place the old one is there whole.
    Windows antivirus or OneDrive may hold the new file for a moment, hence the
    retries; a copy is the last resort. The file keeps the permissions of the
    one it replaces (a new one gets the usual ones).
    """
    path = str(path)
    folder = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".amanuensis-", dir=folder)
    except OSError:
        raise unwritable(path) from None
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        elif os.name != "nt":
            try:  # best effort: some network file systems refuse chmod
                os.chmod(tmp, _file_mode(path))
            except OSError:
                pass
        for _ in range(3):
            if _rename(tmp, path):
                return path
            options.sleep(0.2)
        if not _copy(tmp, path):
            raise unwritable(path)
        return path
    except OSError:
        raise unwritable(path) from None
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def write_text(text, path) -> str:
    """UTF-8 text with LF line endings; a list is joined with newlines."""
    if not isinstance(text, str):
        text = "\n".join(str(t) for t in text)
    return write_bytes_atomic(text.encode("utf-8"), path)


def append_text(lines, path) -> str:
    """Add lines (each ending in a newline) to a text file."""
    if isinstance(lines, str):
        lines = [lines]
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "ab") as f:
        f.write(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def remove_leftovers(folder) -> None:
    """Temporary files of an interrupted write."""
    try:
        for name in os.listdir(folder):
            if name.startswith(".amanuensis-"):
                try:
                    os.unlink(os.path.join(folder, name))
                except OSError:
                    pass
    except OSError:
        pass


# ------- numbers as text ----------------------------------------------------------

def is_missing(x) -> bool:
    if x is None:
        return True
    try:
        import pandas as pd
        if x is pd.NA or x is pd.NaT:
            return True
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, float) and math.isnan(x):
        return True
    try:
        import numpy as np
        if isinstance(x, np.floating) and np.isnan(x):
            return True
    except ImportError:  # pragma: no cover
        pass
    return False


def _plain_number(v: float) -> str:
    """A float as text without scientific notation, to 15 significant digits."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    s = format(decimal.Decimal("%.15g" % v), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def chr_value(x, dec: str = "."):
    """One value as text: numbers without scientific notation, TRUE/FALSE for
    booleans, None for missing. ``dec`` is the decimal mark of numbers with
    decimals; whole numbers, booleans and text are never touched."""
    if is_missing(x):
        return None
    try:
        import numpy as np
        if isinstance(x, np.bool_):
            x = bool(x)
        elif isinstance(x, np.integer):
            x = int(x)
        elif isinstance(x, np.floating):
            x = float(x)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        s = _plain_number(x)
        return s.replace(".", dec, 1) if dec != "." else s
    return str(x)


# How a yes/no value may be written (lower case, accents folded). S and N, as
# Spanish and Catalan registers write them, are both accepted. One list for
# the benchmark reader, the editor and the model's output, so that a value is
# read the same everywhere.
YES = ("yes", "y", "si", "s", "sí", "sì", "true", "t", "1", "x", "oui", "ja", "vrai", "verdadero", "cert")
NO = ("no", "n", "false", "f", "0", "non", "nein", "falso", "fals", "faux")


def parse_yes_no(x, extra_yes=(), extra_no=()):
    """True / False / None for one cell (lower case, accents folded)."""
    if is_missing(x):
        return None
    key = fold_accents(str(x).strip()).lower()
    if key in YES or key in extra_yes:
        return True
    if key in NO or key in extra_no:
        return False
    return None


_NUMBER_RE = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
_COMMA_RE = re.compile(r"^\s*[-+]?[0-9]*,[0-9]+\s*$")


def as_number(x, dec: str = "."):
    """Text read as a number (float), or NaN. With ``dec=","`` a plain number
    with one decimal comma ("2,5") reads as 2.5; "1.250,5" is never guessed at."""
    if is_missing(x):
        return math.nan
    if isinstance(x, bool):
        return math.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    if dec == "," and _COMMA_RE.match(s):
        s = s.replace(",", ".", 1)
    if _NUMBER_RE.match(s):
        return float(s)
    t = s.strip()
    if t in ("Inf", "inf"):
        return math.inf
    if t in ("-Inf", "-inf"):
        return -math.inf
    return math.nan


# ------- CSV ------------------------------------------------------------------------

def _csv_cell(v, sep: str, dec: str) -> str:
    s = chr_value(v, dec=dec)
    if s is None:
        return ""
    specials = '"\n\r' + ("\t" if sep == "\t" else sep)
    if any(c in s for c in specials):
        return '"' + s.replace('"', '""') + '"'
    return s


def csv_text(df, sep: str = ",", dec: str = ".") -> str:
    """A table as CSV text the same on every system: fields quoted only when
    they hold the separator, a quote or a line break; empty for missing."""
    if dec == sep:
        raise InternalError("The decimal mark and the column separator must differ.")
    cols = list(df.columns)
    lines = [sep.join(_csv_cell(str(c), sep, dec) for c in cols)]
    columns = [df[c].tolist() for c in cols]
    for i in range(len(df)):
        lines.append(sep.join(_csv_cell(col[i], sep, dec) for col in columns))
    return "\n".join(lines) + "\n"


def write_csv(df, path, bom: bool = True, sep: str = ",", dec: str = ".") -> str:
    data = csv_text(df, sep=sep, dec=dec).encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    return write_bytes_atomic(data, path)


def read_delim_rows(path):
    """(rows, sep): the non-empty lines of a CSV/TSV file as lists of stripped
    cells, and the delimiter, sniffed from the first line (comma, semicolon or
    tab; Spanish Excel saves with ";"), or from the next lines when the first
    has none (a title line above the table)."""
    text = read_text(path)
    lines = [l for l in text.split("\n") if l.strip()]
    counts = {}
    for line in lines[:10]:
        counts = {",": line.count(","), ";": line.count(";"), "\t": line.count("\t")}
        if any(counts.values()):
            break
    sep = max(counts, key=lambda k: counts[k]) if any(counts.values()) else ","
    reader = csv.reader(io.StringIO(text), delimiter=sep, quotechar='"')
    return [[c.strip() for c in r] for r in reader if any(c.strip() for c in r)], sep


def read_delim_text(path):
    """A CSV/TSV file as a DataFrame of text, nothing guessed: the delimiter is
    sniffed (``read_delim_rows``), every cell stripped, short lines padded.
    ``df.attrs["sep"]`` says which delimiter was found."""
    import pandas as pd

    rows, sep = read_delim_rows(path)
    if not rows:
        df = pd.DataFrame()
        df.attrs["sep"] = sep
        return df
    return delim_frame(rows, sep)


def delim_frame(rows, sep=","):
    """A DataFrame of text from rows whose first is the header."""
    import pandas as pd

    head = [c.strip() for c in rows[0]]
    body = []
    for r in rows[1:]:
        r = [c.strip() for c in r]
        if len(r) < len(head):
            r = r + [""] * (len(head) - len(r))
        body.append(r[: len(head)])
    data = {}
    names = []
    for j, h in enumerate(head):
        name = h
        k = 2
        while name in data:  # duplicate headers stay readable
            name = f"{h}.{k}"
            k += 1
        names.append(name)
        data[name] = [r[j] for r in body]
    df = pd.DataFrame(data, columns=names, dtype=object)
    df.attrs["sep"] = sep
    df.attrs["header"] = head
    return df


# ------- JSON -------------------------------------------------------------------------

def jsonable(x):
    """Plain JSON values: NaN and pandas NA become None, numpy scalars Python ones."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, os.PathLike):
        return str(x)
    try:
        import numpy as np
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return jsonable(float(x))
        if isinstance(x, np.ndarray):
            return [jsonable(v) for v in x.tolist()]
    except ImportError:  # pragma: no cover
        pass
    if is_missing(x):
        return None
    return str(x)


def _json_number(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return "%.15g" % v


def _compact(x, out: list) -> None:
    if x is None:
        out.append("null")
    elif x is True:
        out.append("true")
    elif x is False:
        out.append("false")
    elif isinstance(x, int):
        out.append(str(x))
    elif isinstance(x, float):
        out.append(_json_number(x))
    elif isinstance(x, str):
        out.append(json.dumps(x, ensure_ascii=False))
    elif isinstance(x, dict):
        out.append("{")
        first = True
        for k, v in x.items():
            if not first:
                out.append(",")
            first = False
            out.append(json.dumps(str(k), ensure_ascii=False))
            out.append(":")
            _compact(v, out)
        out.append("}")
    elif isinstance(x, (list, tuple)):
        out.append("[")
        for i, v in enumerate(x):
            if i:
                out.append(",")
            _compact(v, out)
        out.append("]")
    else:
        _compact(jsonable(x), out)


def to_json(x, pretty: bool = False) -> str:
    """JSON text: compact (the form hashes are taken from) or indented."""
    x = jsonable(x)
    if pretty:
        return json.dumps(x, ensure_ascii=False, indent=2)
    out: list = []
    _compact(x, out)
    return "".join(out)


def parse_json(text: str):
    return json.loads(text)


def read_json(path):
    return parse_json(read_text(path))


def write_json(x, path, pretty: bool = True) -> str:
    return write_text(to_json(x, pretty=pretty), path)


# ------- hashing ------------------------------------------------------------------------

def sha12(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------- time ----------------------------------------------------------------------------

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return datetime.now().date().isoformat()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ------- formatting -----------------------------------------------------------------------

def fmt_int(n) -> str:
    if is_missing(n):
        return "-"
    return f"{int(round(float(n))):,}"


def fmt_usd(x) -> str:
    if is_missing(x):
        return "unknown"
    x = float(x)
    return f"${x:.4f}" if x < 0.1 else f"${x:.2f}"


def fmt_pct(x) -> str:
    return "-" if is_missing(x) else f"{100 * float(x):.2f}%"


# ------- argument checks ----------------------------------------------------------------------

def is_string(x) -> bool:
    return isinstance(x, (str, os.PathLike)) and not isinstance(x, bytes)


def check_string(x, arg: str):
    if not is_string(x):
        raise InputError(f"`{arg}` must be a single piece of text (or a path).")
    return str(x)


def check_choice(x, choices, arg: str):
    if not isinstance(x, str) or x not in choices:
        raise InputError(f"`{arg}` must be one of {', '.join(repr(c) for c in choices)}.", ("x", f"You gave {x!r}."))
    return x


def check_number(x, arg: str, minimum: float = -math.inf, none_ok: bool = False):
    if none_ok and x is None:
        return x
    if isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(float(x)) or x < minimum:
        extra = f" of at least {minimum:g}" if math.isfinite(minimum) else ""
        raise InputError(f"`{arg}` must be a single number{extra}.")
    return x


def check_flag(x, arg: str) -> bool:
    if not isinstance(x, bool):
        raise InputError(f"`{arg}` must be True or False.")
    return x


def merge_dicts(old: dict, new: dict) -> dict:
    """Deep merge: values in ``new`` replace those in ``old``, dicts merge
    recursively, and None deletes a key."""
    out = dict(old)
    for k, v in (new or {}).items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def retry_sleep(seconds: float) -> None:  # pragma: no cover - a seam for tests
    time.sleep(seconds)

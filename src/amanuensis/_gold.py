"""benchmark(): the gold standard - your own transcription of a few
pages, made before you look at the model's output, which score() compares
the model with. One call makes it and, run again, reads it back.

The draw. The pages are a seeded permutation of the page file names
(Unicode-normalised, natural order with ASCII-only case folding and a byte
sort, so the order never depends on the computer's language settings), and
the benchmark is its first n pages. A later call with a larger n keeps every
page already transcribed and adds the next ones of the same permutation: the
draw is never silently redone. The permutation is R's (``_rng``), so a seed
draws the same pages here and in the R package. Without a seed one is made
from the file names (sha256), so the draw is still reproducible; a seed
taken from the source itself (its year, its shelfmark) is better, as a reader
can check it, hence the advice printed with it.

Reading. Every cell is read as text and converted to the codebook's type
here, never guessed. A value that cannot be converted stops the read with its
sheet and cell: turned into an empty value it would silently count for or
against the model. Excel's damage to text (1-2 turned into a date, long
numbers kept to 15 digits, 007 into 7) is warned about, and so is a number
that may be thousands written with a point (1.250). A cell Excel stored as a
date in a number or yes/no column stops the read, and stops a rewrite of the
file (the editor, a larger n). Only pages marked done are used: a half-typed
page would count every value not yet typed as a model error. An explicit
"no" always keeps a page out; a done mark that says neither yes nor no, or a
pages sheet without its file column, stops the read; only when the done
column was left empty for every page are the pages with values used (said
aloud).

A rewrite (the editor, a larger n) keeps what the user added to the file:
columns of the benchmark sheet, lines of other pages, sheets, and notes
above the first page of a text benchmark.

Prefilling from the model's output is allowed (it saves typing on dense
tables) but recorded, because people correcting a machine's output accept
some of its mistakes: accuracy measured against it is optimistic.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from dataclasses import replace

import pandas as pd

from ._codebook import (Codebook, TEXT_DESCRIPTION_WITH_RECORDS, codebook, codebook_from_table, codebook_table,
                        codebook_text_only, read_codebook_sheet, records_path, row_columns, subset)
from ._docx import write_plain_docx
from ._errors import GoldEmptyError, InputError, NotFoundError
from ._flatten import object_scalars, page_rows
from ._images import as_images
from ._instructions import read_docx_text
from ._options import can_ask, is_interactive
from ._result import Result, load_pickle, result_info, result_units
from ._rng import r_sample_int
from ._say import done, info, listing, plural, quietly, say, warn
from ._text import (NO, YES, as_number, chr_value, ext, fold_accents, is_missing, natural_key, nfc, now, page_key,
                    parse_json, parse_yes_no, read_delim_text, read_json, read_text, show_path, stamp, stem, to_json,
                    user_path, write_text)
from ._version import __version__
from ._xlsx import SheetSpec, col_letter, read_cells, read_sheet, read_table, sheet_names, write_xlsx

META_SHEETS = ("pages", "about", "codebook", "how_to")
XLSX_EXTS = ("xlsx", "xlsm", "xls")
TEXT_EXTS = ("txt", "md", "docx")
ALL_EXTS = XLSX_EXTS + ("csv", "tsv", "dta", "pkl", "pickle", "json") + TEXT_EXTS
ABOUT_ITEMS = ("package", "made", "seed", "n", "method", "drawn", "rows", "prefilled", "codebook_source", "free_text", "images")
QUIET_COLS = ("source", "source_page", "page_status", "notes", "note", "comment", "comments")


class Gold(pd.DataFrame):
    """A benchmark: the layout of the results table (file, row, variables, typed
    as the codebook says), with how it was made in ``.gold`` (codebook, pages
    done, the draw, rows, prefilled, source)."""

    _metadata = ["_gold"]

    @property
    def _constructor(self):
        return Gold

    @property
    def gold(self) -> dict | None:
        return getattr(self, "_gold", None)

    def __repr__(self) -> str:
        g = self.gold
        df = pd.DataFrame(self)
        if g is None:
            return df.to_string(index=False)
        p = g["pages"]
        n_done = int(p["done"].sum())
        src = f" ({os.path.basename(g['source'])})" if g.get("source") else ""
        head = f"Benchmark{src}: {plural(n_done, 'page')} done" + (f" of {len(p)}" if len(p) > n_done else "")
        cb = g["codebook"]
        if any(v.level == "row" for v in cb.vars):
            head += f", {plural(int(p.loc[p['done'], 'n_rows'].fillna(0).sum()), 'row')}"
        head += ", free text" if cb.kind == "text" else f", {plural(len(cb.vars), 'variable')}"
        lines = [head]
        d = g.get("draw")
        if d:
            when = f" on {str(d.get('date'))[:10]}" if d.get("date") else ""
            lines.append(f"Drawn with seed {_fmt_seed(d.get('seed'))} ({plural(len(d.get('files') or []), 'page')} drawn{when}).")
        else:
            lines.append("Not drawn by benchmark() (a file of your own).")
        if g.get("rows"):
            lines.append(f"Only the first {g['rows']} rows of each page are compared.")
        if g.get("prefilled"):
            lines.append("Prefilled with the model's output: accuracy measured against it is optimistic.")
        body = df.head(10).to_string(index=False) if len(df) else "(no rows)"
        if len(df) > 10:
            body += f"\n... and {len(df) - 10:,} more rows"
        return "\n".join(lines + [body])


def _fmt_seed(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "unknown"
    return str(int(s)) if float(s) == int(s) else str(s)


def new_gold(tab, cb, pages, draw=None, rows=None, prefilled=False, source=None) -> Gold:
    g = Gold(tab)
    g._gold = {"codebook": cb, "pages": pages, "draw": draw, "rows": rows, "prefilled": bool(prefilled), "source": source}
    return g


def benchmark(file, images=None, codebook=None, *, n: int | None = None, seed: int | None = None, rows: int | None = None,
              app: bool | None = None, prefill=None):
    """Make or read a benchmark: your own transcription of a few pages.

    The first time (``file`` does not exist yet), ``n`` pages are drawn at
    random from ``images`` and a file is written for you to fill in: an
    Excel file (sheets ``benchmark``, ``pages`` - mark each finished page
    ``done = yes`` -, ``about``, ``codebook`` and ``how_to``; every column is
    formatted as Text and category and yes/no columns have drop-down lists),
    or for free text a ``.txt``, ``.md`` or ``.docx`` file with a line
    ``=== <file name> ===`` per page. In an interactive session an editor
    opens in your browser instead (the page image beside a form built from
    your codebook); it saves the same file after every page.

    Run the same line again once the file is filled in: it reads your
    transcription (only pages marked done) and returns it for :func:`score`.
    A larger ``n`` adds the next pages of the same draw and keeps what you
    typed. Choose the seed from the source itself (its year, say) and report
    it; anyone can then check which pages were drawn.

    Besides its own template, benchmark() reads a transcription made another
    way: an ``.xlsx``, ``.csv``, ``.dta`` or ``.pkl`` file with the columns
    ``file``, ``row`` and your variables, a ``.json`` file as :func:`output`
    writes it, or a text file with ``=== <file name> ===`` lines (give
    ``codebook`` for these).

    Parameters
    ----------
    file : path
        The benchmark file: made if it does not exist, read if it does.
    images : Images, folder or paths, optional
        The page images; needed to make the benchmark.
    codebook : Codebook or path, optional
        Needed to make it; the Excel template keeps a copy for reading.
    n : int, optional
        How many pages to draw. Default 2; 10 or more for a published figure.
        Given for an existing benchmark, pages are added up to ``n``.
    seed : int, optional
        A whole number that fixes the draw, ideally from the source (its year).
    rows : int, optional
        Type only the first ``rows`` rows of each page (dense tables).
    app : bool, optional
        True for the editor in your browser, False for a file to fill in.
        Default: the editor in an interactive session.
    prefill : Result, optional
        Start the drawn pages from the model's output, to correct
        (recorded: accuracy measured this way is optimistic).

    Returns
    -------
    Gold or None
        The benchmark when reading; None after writing a file to fill in.
    """
    if file is None:
        raise InputError("Say which file holds the benchmark.",
                         "For example benchmark('gold.xlsx', images=pages, codebook=cb): the first time it writes the file "
                         "for you to fill in; run again, the same line reads it.")
    if not isinstance(file, (str, os.PathLike)):
        raise InputError("`file` must be a file name, e.g. 'gold.xlsx'.")
    file = str(user_path(file))
    n_given = n is not None
    n = 2 if n is None else n
    _check_count(n, "n")
    if rows is not None:
        _check_count(rows, "rows")
    if seed is not None:
        _check_seed(seed)
    if app is not None and not isinstance(app, bool):
        raise InputError("`app` must be True or False.")
    e = ext(file)
    if e not in ALL_EXTS:
        raise InputError("A benchmark must be an Excel file (.xlsx), or a .csv, .dta, .pkl or .json file; for free text also "
                         ".txt, .md or .docx.", ("x", f"{os.path.basename(file)} has the extension {e!r}."))
    pages = None if images is None else as_images(images)
    if prefill is not None:
        from ._result import as_result
        prefill = as_result(prefill)
    cb = None if codebook is None else gold_as_codebook(codebook)
    app_given = app is not None
    if os.path.exists(file):
        if n_given and extend(file, pages, cb, n, seed, prefill):
            if use_app(app, app_given):
                return after_app(file, cb, pages)
            return None
        if app is True and e in ("xlsx",) + TEXT_EXTS and use_app(app, True):
            return after_app(file, cb, pages)
        return gold_read(file, cb, pages, rows)
    return create(file, pages, cb, n, seed, rows, prefill, use_app(app, app_given))


# ------- arguments -------------------------------------------------------------------

def _check_count(x, arg):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x < 1 or x != int(x):
        raise InputError(f"`{arg}` must be a whole number of at least 1.")


def _check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, float)) or seed != int(seed) or abs(seed) > 2147483647:
        raise InputError("`seed` must be a whole number, such as the year of the source (seed=1930).", ("x", f"You gave {seed!r}."))


def gold_as_codebook(x) -> Codebook:
    if isinstance(x, Codebook):
        return x
    if isinstance(x, (pd.DataFrame, str, os.PathLike)):
        return codebook(x)
    raise InputError("`codebook` must come from codebook(), or be a codebook file.")


def use_app(app, app_given) -> bool:
    """The editor opens only in an interactive session (it needs nothing else)."""
    if app is False:
        return False
    if not is_interactive():
        if app_given and app:
            info("The benchmark editor needs an interactive Python session; writing a file to fill in instead.")
        return False
    return True if app is None else bool(app)


# ------- small helpers ----------------------------------------------------------------

def _program(e: str) -> str:
    return "Excel" if e in XLSX_EXTS + ("csv", "tsv") else "Word" if e == "docx" else "the file"


def gold_columns(cb) -> list[str]:
    return (["file"] + (["row"] if any(v.level == "row" for v in cb.vars) else [])
            + [v.variable for v in cb.vars if v.level == "page"] + [v.variable for v in cb.vars if v.level == "row"])


def parse_done(x):
    return parse_yes_no(x, extra_yes=("done", "checked", "ok", "hecho", "fet", "terminado", "acabat"),
                        extra_no=("not done", "to check", "pending", "pendiente", "pendent"))


def gold_order(files) -> list[int]:
    """Natural order the same on every computer: only ASCII letters are
    case-folded and ties are broken by a byte sort."""
    files = [nfc(f) for f in files]
    ascii_lower = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    from ._text import _natural_key
    keys = [_natural_key(f, lambda s: s.translate(ascii_lower)) for f in files]
    return sorted(range(len(files)), key=lambda i: (keys[i], files[i]))


def ordered(files) -> list[str]:
    files = list(files)
    return [files[i] for i in gold_order(files)]


def gold_cell_text(x) -> str:
    if is_missing(x):
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    s = chr_value(x)
    return "" if s is None else s


def match_columns(have, want) -> dict:
    """Columns of a file matched to the names wanted: exactly, then ignoring
    case and spaces, then as Stata and Excel users rename them."""
    have = ["" if h is None else str(h) for h in have]

    def norm1(x):
        return nfc(x).strip().lower()

    def norm2(x):
        x = fold_accents(x.strip()).lower()
        x = re.sub(r"[^a-z0-9]+", "_", x)
        return x.strip("_")

    passes = [nfc, norm1, norm2, lambda x: norm2(x)[:32]]
    out = {w: None for w in want}
    taken = [False] * len(have)
    for p in passes:
        h = [p(x) for x in have]
        for w in want:
            if out[w] is not None:
                continue
            pw = p(w)
            for j in range(len(have)):
                if not taken[j] and h[j] and h[j] == pw:
                    out[w] = j
                    taken[j] = True
                    break
    return out


def canonical(files, image_files=None):
    """Typed file names mapped to the page images' own spelling (case, Unicode
    form, a missing or different extension); names that match no image apart."""
    files = [nfc(f).strip() for f in files]
    if image_files is None or not files:
        return files, []
    ik = [page_key(f) for f in image_files]
    isk = [page_key(stem(f)) for f in image_files]
    out, unknown = [], []
    for f in files:
        k = page_key(f)
        if k in ik:
            out.append(image_files[ik.index(k)])
        elif page_key(stem(f)) in isk:
            out.append(image_files[isk.index(page_key(stem(f)))])
        else:
            out.append(f)
            if f and f not in unknown:
                unknown.append(f)
    return out, unknown


# ------- the draw ---------------------------------------------------------------------------

def seed_from_names(files) -> int:
    """sha256 of the sorted, normalised names, cut to 28 bits."""
    files = sorted(nfc(f) for f in files)
    hexd = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()
    return int(hexd[:7], 16)


def permutation(files, seed: int) -> list[str]:
    base = ordered([nfc(f) for f in files])
    perm = r_sample_int(len(base), int(seed))
    return [base[i - 1] for i in perm]


def draw(files, n: int, seed: int) -> list[str]:
    return permutation(files, seed)[: min(n, len(files))]


def extend_draw(files, drawn, n: int, seed: int):
    """Pages to add to an earlier draw so that it holds n. With the same images
    these are exactly the next pages of the same permutation; if images were
    added or removed, the earlier pages are kept and the new ones are the
    first of the current permutation not drawn yet (same = False)."""
    full = permutation(files, seed)
    k = len(drawn)
    dk = [page_key(d) for d in drawn]
    same = k <= len(full) and [page_key(f) for f in full[:k]] == dk
    rest = [f for f in full if page_key(f) not in dk]
    return rest[: max(n - k, 0)], same


def method_text(derived: bool) -> str:
    return ("seeded random order of the page file names (natural order, R's Mersenne-Twister generator, sample.int with "
            "rejection sampling, reproduced in Python); the first n pages taken" + ("; seed made from the file names" if derived else ""))


def describe_draw(drawn, total, seed, derived, cb, rows, units) -> None:
    n = len(drawn)
    out = [("v", f"Drew {n} of {plural(total, 'page')} for the benchmark (seed {seed}): {listing(drawn, 6)}.")]
    if derived:
        out.append(("i", "The seed was made from the file names, so the same pages always give the same draw. For a draw others "
                         "can check, choose the seed yourself: use a number from the source itself, e.g. its year (seed=1930)."))
    if n >= total:
        out.append(("i", "The benchmark covers every page."))
    else:
        cats = [v for v in cb.vars if v.type == "category"]
        gaps = []
        if cats:
            shown = "; ".join(f"{v.variable}: {listing(v.values, 4)}" for v in cats[:3])
            gaps.append(f"category values that do not occur on {'that page' if n == 1 else 'those pages'} go unchecked ({shown})")
        pv = [v.variable for v in cb.vars if v.level == "page"]
        if pv and any(v.level == "row" for v in cb.vars):
            gaps.append(f"page variables ({listing(pv, 4)}) are compared on {plural(n, 'value')} each")
        line = (f"A benchmark of {plural(n, 'page')} measures accuracy on what is on {'that page' if n == 1 else 'those pages'} only"
                + (": " + "; ".join(gaps) if gaps else "") + ".")
        if n < 10:
            line += " For a figure to publish, use n=10 or more."
        out.append(("!", line))
    if any(v.level == "row" for v in cb.vars):
        if rows is not None:
            out.append(("i", f"Only the first {rows} rows of each page need typing (at most {rows * n:,} rows in all); this is recorded "
                             "in the file and score() compares only those."))
        elif units:
            typed = sum(len(next(iter(u["rows"].values()), [])) for u in units.values())
            out.append(("i", f"About {plural(typed, 'row')} to check, from the model's output."))
        else:
            out.append(("i", "Every row of each page is to be typed; for dense pages, rows=20 asks for only the first 20 rows of each page."))
    say(*out)


# ------- making the benchmark ------------------------------------------------------------------

def create(file, pages, cb, n, seed, rows, prefill, app):
    e = ext(file)
    if pages is None or (cb is None and prefill is None):
        raise InputError(f"{os.path.basename(file)} does not exist yet.",
                         f"To make it, give the page images and the codebook: benchmark({os.path.basename(file)!r}, images=pages, codebook=cb).",
                         f"To read an existing benchmark, check the spelling. Relative paths start from your working folder, {os.getcwd()}.")
    if cb is None:
        cb = prefill.info.codebook
        info("Using the codebook of the extraction given in `prefill`.")
    check_template_ext(e, cb, file)
    folder = getattr(pages, "folder", None) or os.path.dirname(pages["image"].iloc[0])
    total = len(pages)
    derived = seed is None
    if derived:
        seed = seed_from_names(pages["file"])
    drawn = draw(list(pages["file"]), n, int(seed))
    if n > total:
        info(f"There are only {plural(total, 'page image')}; the benchmark takes all of them.")
    drawn = canonical(drawn, list(pages["file"]))[0]
    files = ordered(drawn)
    units = {} if prefill is None else prefill_units(prefill, files, cb, rows)
    prefilled = bool(units)
    about = gold_about(int(seed), derived, drawn, rows, prefilled, cb, folder)
    book = {"file": file, "ext": e, "cb": cb, "about": about, "lines": lines_chr(units, cb, files),
            "pages": pd.DataFrame({"file": files, "done": initial_done(e, files, units), "notes": [None] * len(files)})}
    describe_draw(drawn, total, int(seed), derived, cb, rows, units)
    if prefill is not None:
        prefill_messages(units, files)
    book_write(book)
    if app:
        return after_app(file, cb, pages)
    fill_message(file, files, folder, prefilled)
    return None


def check_template_ext(e, cb, file):
    name, st = os.path.basename(file), stem(file)
    if e == "xlsx":
        return
    if e in TEXT_EXTS:
        if cb.kind == "text":
            return
        raise InputError(f"A .{e} benchmark holds only the text of each page, but the codebook asks for a table.",
                         f"Use an Excel file: benchmark({st + '.xlsx'!r}, ...).")
    raise InputError("A new benchmark is written as an Excel file" + (" (or .txt, .md, .docx for free text)" if cb.kind == "text" else "") + ".",
                     ("x", f"{name} does not exist, and a .{e} file cannot be made into a template to fill in."),
                     f"Use benchmark({st + '.xlsx'!r}, ...); files of other kinds are read once they exist.")


def initial_done(e, files, units) -> list:
    """Pages start "not done". In a text file an unmarked page with text counts
    as done, so only prefilled pages (whose text is the model's) are marked."""
    if e in TEXT_EXTS:
        uk = {page_key(k) for k in units}
        return ["no" if page_key(f) in uk else None for f in files]
    return ["no"] * len(files)


def join_drawn(files) -> str:
    """The draw as the about sheet records it: the names joined with "; ", as
    the R package writes it; a JSON list when a name holds a ";" (or would
    not survive the split), which splitting on ";" would cut in two."""
    files = [str(f) for f in files]
    if any(";" in f or f != f.strip() for f in files) or (files and files[0].startswith("[")):
        return to_json(files)
    return "; ".join(files)


def split_drawn(x) -> list[str]:
    """The names of a recorded draw, from either form ``join_drawn`` writes."""
    if is_missing(x) or not str(x).strip():
        return []
    s = str(x).strip()
    if s.startswith("["):
        try:
            v = parse_json(s)
        except ValueError:
            v = None
        if isinstance(v, list):
            return [str(f) for f in v]
    return [d.strip() for d in s.split(";") if d.strip()]


def gold_about(seed, derived, drawn, rows, prefilled, cb, folder) -> dict:
    return {"package": f"amanuensis {__version__} (Python)", "made": now(), "seed": str(seed), "n": str(len(drawn)),
            "method": method_text(derived), "drawn": join_drawn(drawn), "rows": None if rows is None else str(rows),
            "prefilled": "yes" if prefilled else "no", "codebook_source": str(cb.source),
            "free_text": "yes" if cb.free_text else "no", "images": None if folder is None else str(folder)}


def fill_message(file, files, folder, prefilled):
    e = ext(file)
    if e in TEXT_EXTS:
        what = ("Open it and correct the model's text of each page against the image; then change \"not done\" to \"done\" on its === line."
                if prefilled else "Open it and type the text of each page under its === line, exactly as written.")
    else:
        what = ("Open it in Excel and correct the model's values against each page image; then write yes in the done column of the pages sheet."
                if prefilled else
                "Open it in Excel and type what each page says: one line per row of the page's table (the first line of each page is "
                "there already). Then write yes in the done column of the pages sheet.")
    say(("v", f"Wrote {show_path(file)} with {plural(len(files), 'page')} to transcribe."), ("i", what),
        ("i", f"The page images are in {folder}."),
        (">", f"Then save, close {_program(e)}, and run the same line again to read it."))


def prefill_units(prefill, files, cb, rows=None) -> dict:
    """The model's output for the drawn pages, as template text."""
    units = result_units(prefill)
    keys = {page_key(k): k for k in units}
    pv = [v.variable for v in cb.vars if v.level == "page"]
    rv = [v.variable for v in cb.vars if v.level == "row"]
    out = {}
    for f in files:
        k = keys.get(page_key(f))
        if k is None or units[k]["status"] not in ("ok", "check"):
            continue
        u = units[k]
        page = {v: gold_cell_text(u["page"].get(v)) for v in pv}
        n = u["n"]
        rws = {v: [gold_cell_text(x) for x in u["rows"].get(v, [None] * n)] for v in rv}
        if rows is not None:
            rws = {v: x[:rows] for v, x in rws.items()}
        out[f] = {"page": page, "rows": rws}
    return out


def prefill_messages(units, files):
    missing = [f for f in files if page_key(f) not in {page_key(k) for k in units}]
    if units:
        warn(f"The benchmark starts from the model's own output for {plural(len(units), 'page')}.",
             ("!", "Correct every value against the page image before marking a page done."),
             "Accuracy measured against corrected model output is optimistic: mistakes you do not notice count as right. "
             "This is recorded in the file.")
    if missing:
        info(f"The model has no output for {listing(missing)}; {'it is' if len(missing) == 1 else 'they are'} left empty to type from scratch.")


# ------- the book: the file's content as text ------------------------------------------------------
#
# The template, the extension of a draw and the editor all work on one plain
# description of a benchmark file: file, ext, cb; lines (DataFrame of text in
# the sheet layout, or file + text for a text file); pages (file, done
# "yes"/"no"/None, notes); about (the "about" items).

def book_write(book) -> None:
    e = book["ext"]
    if e in ("txt", "md"):
        write_text(text_lines(book), book["file"])
    elif e == "docx":
        write_plain_docx(text_lines(book), book["file"])
    else:
        write_gold_xlsx(book)


def gold_load(file, fallback=None) -> dict:
    """A benchmark file as a book, as text, converting nothing. The editor and
    benchmark(..., n=larger) rewrite the file from it, so what a rewrite
    would change is refused here: a date Excel made in a number, yes/no or
    row column (rewritten, its serial number would read as typed), and a
    done mark that is neither yes nor no."""
    e = ext(file)
    if e in TEXT_EXTS:
        raw = raw_text(read_docx_text(file) if e == "docx" else read_text(file), file)
        check_done_words(raw)
        cb = codebook_text_only()
        pages = raw["pages"][["file", "done", "notes"]]
        lines = raw["lines"]
    elif e in XLSX_EXTS:
        raw = raw_xlsx(file)
        cb = raw["codebook"] or fallback
        if cb is None:
            raise InputError(f"{os.path.basename(file)} has no 'codebook' sheet.",
                             f"Give the codebook: benchmark({os.path.basename(file)!r}, codebook=cb, ...).")
        refuse_date_cells(raw, cb)
        if raw["pages"] is not None:
            check_done_words(raw)
        lines = raw["lines"]
        pages = raw["pages"]
        if pages is None:
            fc = match_columns(lines.columns, ["file"])["file"]
            fs = [] if fc is None else list(dict.fromkeys(str(v).strip() for v in lines.iloc[:, fc] if not is_missing(v)))
            pages = pd.DataFrame({"file": fs, "done": ["no"] * len(fs), "notes": [None] * len(fs)})
        else:
            pages = pages[["file", "done", "notes"]]
    else:
        raise InputError("Only benchmarks in .xlsx, .txt, .md or .docx files can be edited or extended.")
    return {"file": file, "ext": e, "cb": cb, "lines": lines, "pages": pages.reset_index(drop=True), "about": raw.get("about") or {},
            "preamble": raw.get("preamble"), "bench_sheet": raw.get("bench_sheet")}


def refuse_date_cells(raw, cb) -> None:
    """Cells Excel stored as dates in the columns that must hold numbers,
    yes/no or row numbers stop a rewrite of the file (the reader stops on
    them too)."""
    dates = raw.get("date_cells")
    if not dates:
        return
    want = ["row"] + [v.variable for v in cb.vars if v.type in ("number", "integer", "boolean")]
    cols = match_columns(raw["lines"].columns, want)
    found = []
    for var, j in cols.items():
        if j is None:
            continue
        for i in range(len(raw["lines"])):
            if i < len(dates) and j < len(dates[i]) and dates[i][j] and not is_missing(raw["lines"].iloc[i, j]):
                found.append(f"{raw['loc'](i, var, j)} reads \"{raw['lines'].iloc[i, j]}\"")
    if found:
        raise InputError(f"Excel turned what was typed into a date in {plural(len(found), 'cell')} of {raw['label']}, so the file "
                         "cannot be rewritten: the date's serial number would then read as if it had been typed.",
                         *[("x", f) for f in found[:10]],
                         ("i", "Type those values again (as numbers, or yes/no), save, close Excel, and run the line again."))


def check_done_words(raw) -> None:
    """A done mark must say yes or no (done or not done): any other word
    ("to do", "not done yet") once counted as blank, so a half-typed page
    with values was scored."""
    ps = raw["pages"]
    if ps is None or not len(ps):
        return
    dvals = list(ps["done"])
    bad = [i for i, d in enumerate(dvals) if not (is_missing(d) or not str(d).strip()) and parse_done(d) is None]
    if not bad:
        return
    label = raw["label"]
    text = raw["kind"] == "text"
    where = [f"{raw['page_loc'](ps['_line'].iloc[i])} says \"{dvals[i]}\"" for i in bad]
    program = "the file" if text else "Excel"
    raise InputError((f"After a page's === line in {label}, write done or not done: " if text
                      else f"The done column of {label} must say yes or no: ") + f"{listing(where)}.",
                     f"Correct it, save, close {program}, and run the line again.")


def backup(file) -> str:
    """A copy of a file about to be rewritten, next to it, so nothing typed is lost."""
    to = os.path.join(os.path.dirname(os.path.abspath(file)), f"{stem(file)}-backup-{stamp()}{os.path.splitext(file)[1]}")
    try:
        shutil.copyfile(file, to)
    except OSError:
        from ._text import unwritable
        raise unwritable(to) from None
    return to


def extend(file, pages, cb, n, seed, prefill) -> bool:
    """Add pages to an existing drawn benchmark when a larger n is asked for."""
    e = ext(file)
    name = os.path.basename(file)
    if e not in ("xlsx",) + TEXT_EXTS:
        info(f"`n` draws pages for a new benchmark; {name} is read as it is.")
        return False
    book = gold_load(file, fallback=cb)
    a = book["about"]
    drawn = split_drawn(a.get("drawn"))
    old_seed = as_number(a.get("seed"))
    if not drawn or math.isnan(old_seed):
        info(f"{name} has no record of a draw, so pages cannot be added to it; `n` is ignored.")
        return False
    if n <= len(drawn):
        if n < len(drawn):
            info(f"{name} already has {len(drawn)} drawn pages; pages are never taken out of a benchmark, so n={n} is ignored.")
        return False
    if pages is None:
        raise InputError(f"To add pages to {name}, give the page images too.", f"For example benchmark({name!r}, images=pages, n={n}).")
    if seed is not None and seed != old_seed:
        info(f"{name} was drawn with seed {int(old_seed)}; the added pages continue that draw (the seed you gave is not used).")
    add_raw, same = extend_draw(list(pages["file"]), drawn, n, int(old_seed))
    add = canonical(add_raw, list(pages["file"]))[0]
    if not add:
        info(f"Every page image is already in {name}.")
        return False
    if not same:
        say(("!", f"The page images are not the same as when {name} was drawn: the pages already drawn are kept, and the added "
                  "ones come from the same seeded draw over the current images."))
    cbk = book["cb"]
    r = as_number(a.get("rows"))
    rows = None if math.isnan(r) else int(r)
    units = {} if prefill is None else prefill_units(prefill, add, cbk, rows)
    add = ordered(add)
    book["lines"] = append_lines(book["lines"], lines_chr(units, cbk, add))
    book["pages"] = pd.concat([book["pages"], pd.DataFrame({"file": add, "done": initial_done(e, add, units), "notes": [None] * len(add)})],
                              ignore_index=True)
    a["n"] = str(len(drawn) + len(add_raw))
    a["drawn"] = join_drawn(drawn + add_raw)
    if units:
        a["prefilled"] = "yes"
    book["about"] = a
    bk = backup(file)
    book_write(book)
    if prefill is not None:
        prefill_messages(units, add)
    say(("v", f"Added {plural(len(add), 'page')} to {name}: {listing(add)}."),
        ("i", f"What you had typed is kept; a copy of the previous file is in {os.path.basename(bk)}."),
        (">", f"Fill in the new pages, mark them done, save, close {_program(e)}, and run the same line again."))
    return True


def append_lines(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New template lines under the existing ones, matched by column name."""
    if not len(old):
        return new
    cols = match_columns(old.columns, list(new.columns))
    add = pd.DataFrame({c: [None] * len(new) for c in old.columns}, dtype=object)
    for want, j in cols.items():
        if j is not None:
            add.iloc[:, j] = new[want].tolist()
    old2 = old.astype(object)
    out = pd.concat([old2, add], ignore_index=True)
    out.columns = old.columns
    return out


def units_chr(lines: pd.DataFrame, cb, files) -> dict:
    """Text page units from lines in the sheet layout: page values = the first
    non-empty value of the page's lines; rows = lines with a row number or
    any row value, in row-number order when every row has one."""
    pv = [v.variable for v in cb.vars if v.level == "page"]
    rv = [v.variable for v in cb.vars if v.level == "row"]
    cols = match_columns(lines.columns, ["file", "row"] + pv + rv)

    def get(nm, idx):
        j = cols[nm]
        if j is None:
            return [""] * len(idx)
        col = lines.iloc[:, j].tolist()
        return ["" if is_missing(col[i]) else str(col[i]) for i in idx]

    all_idx = list(range(len(lines)))
    fk = [page_key(f) for f in filled_files(lines, cols, rv + pv)]
    out = {}
    for f in files:
        idx = [i for i in all_idx if fk[i] == page_key(f)]
        page = {}
        for v in pv:
            x = [s for s in get(v, idx) if s.strip()]
            page[v] = x[0] if x else ""
        if rv:
            rows = {v: get(v, idx) for v in rv}
            rnum = [s.strip() for s in get("row", idx)]
            is_row = [bool(rnum[k]) or any(rows[v][k].strip() for v in rv) for k in range(len(idx))]
            sel = [k for k in range(len(idx)) if is_row[k]]
            nums = [as_number(rnum[k]) for k in sel]
            if nums and not any(math.isnan(x) for x in nums):
                sel = [sel[i] for i in sorted(range(len(sel)), key=lambda i: nums[i])]
            rows = {v: [rows[v][k] for k in sel] for v in rv}
        else:
            rows = {}
        out[f] = {"page": page, "rows": rows}
    return out


def filled_files(lines: pd.DataFrame, cols: dict, names) -> list[str]:
    """The page of every line: its file cell, or, for a line with values (or a
    row number) and no file name, the file of the line above, as the reader
    takes it (added lines often leave the file out). The editor followed the
    file cell alone, and so showed and kept fewer rows than were read."""
    def col(nm):
        j = cols.get(nm)
        return [""] * len(lines) if j is None else ["" if is_missing(v) else str(v).strip() for v in lines.iloc[:, j]]

    files = col("file")
    rows = col("row")
    vals = [col(n) for n in names]
    out, last = [], ""
    for i, f in enumerate(files):
        if f:
            last = f
        elif rows[i] or any(v[i] for v in vals):
            f = last
        out.append(f)
    return out


def merge_lines(old: pd.DataFrame, new: pd.DataFrame, cb, files) -> pd.DataFrame:
    """The editor's lines (``new``, in the layout of gold_columns) with what the
    file held besides them: the values of columns the user added stay on the
    same line of the same page (by position: the editor adds and deletes rows
    only at the end of a page; a deleted line that carried such a value is
    kept with the value alone), and the lines of pages the editor does not
    show are kept as they were."""
    if old is None or not len(old.columns) or not len(old):
        return new
    cols_new = list(new.columns)
    m = match_columns(old.columns, cols_new)
    own = {j for j in m.values() if j is not None}

    def blank(v):
        return is_missing(v) or not str(v).strip()

    extra = [j for j in range(len(old.columns)) if j not in own
             and (str(old.columns[j]).strip() or not all(blank(v) for v in old.iloc[:, j]))]
    names = [v.variable for v in cb.vars]
    old_files = filled_files(old, match_columns(old.columns, ["file", "row"] + names), names)
    shown = {page_key(f) for f in files}

    by_old, by_new = {}, {}
    for i, f in enumerate(old_files):
        by_old.setdefault(page_key(f), []).append(i)
    for i, f in enumerate(new["file"]):
        by_new.setdefault(page_key(f), []).append(i)
    rows = []
    for f in files:
        k = page_key(f)
        olds, news = by_old.get(k, []), by_new.get(k, [])
        for pos, ni in enumerate(news):
            ex = [old.iloc[olds[pos], j] if pos < len(olds) else None for j in extra]
            rows.append(list(new.iloc[ni]) + ex)
        for pos in range(len(news), len(olds)):
            ex = [old.iloc[olds[pos], j] for j in extra]
            if not all(blank(v) for v in ex):
                rows.append([f if c == "file" else None for c in cols_new] + ex)
    for i, f in enumerate(old_files):
        if page_key(f) in shown or all(blank(v) for v in old.iloc[i]):
            continue
        rows.append([old.iloc[i, m[c]] if m.get(c) is not None else None for c in cols_new] + [old.iloc[i, j] for j in extra])
    return pd.DataFrame(rows, columns=cols_new + [str(old.columns[j]) for j in extra], dtype=object)


def lines_chr(units: dict, cb, files) -> pd.DataFrame:
    """Lines in the sheet layout from text page units: rows numbered 1..k,
    page values on the first line, one line with an empty row for a page with
    no rows. Rows left entirely empty are dropped."""
    pv = [v.variable for v in cb.vars if v.level == "page"]
    rv = [v.variable for v in cb.vars if v.level == "row"]
    cols = gold_columns(cb)
    data = {c: [] for c in cols}
    for f in files:
        u = units.get(f) or {"page": {}, "rows": {}}
        rows = u.get("rows") or {}
        n_all = len(next(iter(rows.values()))) if rows else 0
        keep = [k for k in range(n_all) if any(str(rows.get(v, [""] * n_all)[k] or "").strip() for v in rv)] if rv else []
        n = len(keep)
        m = max(1, n)
        block = {c: [None] * m for c in cols}
        block["file"] = [f] * m
        if n:
            block["row"] = [str(k + 1) for k in range(n)]
        for v in pv:
            val = (u.get("page") or {}).get(v, "")
            if val is not None and str(val).strip():
                block[v][0] = val
        for v in rv:
            if not n or v not in rows:
                continue
            vals = [rows[v][k] for k in keep]
            block[v] = [None if (x is None or not str(x).strip()) else x for x in vals]
        for c in cols:
            data[c] += block[c]
    return pd.DataFrame(data, columns=cols, dtype=object)


# ------- writing the Excel file ------------------------------------------------------------------

def _trunc(x, n):
    x = re.sub(r"\s+", " ", x or "")
    return x[: n - 3] + "..." if len(x) > n else x


def how_to(cb, rows, prefilled, folder) -> list[str]:
    pv = [v.variable for v in cb.vars if v.level == "page"]
    has_rows = any(v.level == "row" for v in cb.vars)
    out = ["This file is your benchmark: your own transcription of a few pages, which score() compares with the model's output to measure its accuracy."]
    if has_rows:
        out.append("Sheet 'benchmark': one line per row of the page's table. The first line of each page is there already; type the page's "
                   "rows on it and on new lines below it, copying the file name onto every line.")
        if pv:
            out.append(f"Page variables ({', '.join(pv)}) need only be filled on the first line of each page.")
        out.append("The row column may stay empty: rows are numbered in the order of the lines.")
        out.append("A page with no rows: leave the row variables empty on its line.")
    else:
        out.append("Sheet 'benchmark': one line per page, already there; type the page's values on it.")
    if rows is not None:
        out.append(f"Type only the first {rows} rows of each page; score() compares only those.")
    out += ["Type exactly what is written, following the same rules as the instructions you give the model: spelling, abbreviations "
            "and accents as written; a doubtful reading ending in ?; an empty cell for what is illegible or not on the page.",
            "Do it before you look at the model's output for these pages.",
            "Yes/no variables: yes or no. Category variables: one of the values in the drop-down list (also in the codebook sheet).",
            "Every column is formatted as Text, so Excel keeps what you type (007, 1-2, 3/12) as typed instead of turning it into a number "
            "or a date. If a value still changes, type an apostrophe first ('007)."]
    if prefilled:
        out.append("The lines hold the model's own output: check every value against the page image and correct it before marking the page done.")
    out.append("When a page is finished, write yes in the done column of the 'pages' sheet. Only pages marked yes are used.")
    if folder:
        out.append(f"The page images are in {folder}.")
    out += ["Keep the sheets 'pages', 'about' and 'codebook': they record the pages, how they were drawn, and the codebook.",
            "When you have finished, save, close Excel, and run the same benchmark() line again."]
    return out


def _dv(kind, rng, values=None, title=None, message=None, error_title=None, error=None):
    args = {"allow_blank": True}
    if kind == "list":
        args.update(type="list", formula1='"' + ",".join(values) + '"', showErrorMessage=True, errorStyle="warning",
                    errorTitle=error_title, error=error)
    if message:
        args.update(showInputMessage=True, promptTitle=(title or "")[:32], prompt=message[:255])
    return {"range": rng, "args": args}


def write_gold_xlsx(book) -> None:
    a = book["about"]
    cb = book["cb"]
    r = as_number(a.get("rows"))
    rows = None if math.isnan(r) else int(r)
    lines = book["lines"].astype(object)
    vmatch = match_columns(lines.columns, [v.variable for v in cb.vars])
    var_of = {j: k for k, j in enumerate(vmatch.values()) if j is not None}
    last = max(len(lines) + 1, 2000)
    widths, checks = [], []
    for j, nm in enumerate(lines.columns):
        k = var_of.get(j)
        v = cb.vars[k] if k is not None else None
        t = v.type if v is not None else "integer" if str(nm).lower() == "row" else "text"
        widths.append(26 if str(nm).lower() == "file" else 6 if str(nm).lower() == "row" else 24 if t == "text" else 14)
        if v is None:
            continue
        rng = f"{col_letter(j + 1)}2:{col_letter(j + 1)}{last}"
        msg = _trunc(v.description, 250)
        if t == "category" and v.values and len(",".join(v.values)) <= 255 and not any("," in x for x in v.values):
            checks.append(_dv("list", rng, v.values, v.variable, msg or None, "Not in the list",
                              "This is not one of the codebook's values. Keep it only if it is right."))
        elif t == "boolean":
            checks.append(_dv("list", rng, ["yes", "no"], v.variable, msg or None, "yes or no", "Write yes or no (or leave the cell empty)."))
        elif msg:
            checks.append(_dv("any", rng, None, v.variable, msg))
    pages = book["pages"]
    pages = pd.DataFrame({"file": [None if is_missing(x) else str(x) for x in pages["file"]],
                          "done": [None if is_missing(x) else str(x) for x in pages["done"]],
                          "notes": [None if is_missing(x) else str(x) for x in pages["notes"]]}, dtype=object)
    plast = max(len(pages) + 1, 500)
    about = pd.DataFrame({"item": list(a), "value": [None if is_missing(v) else str(v) for v in a.values()]}, dtype=object)
    sheets = {
        "benchmark": SheetSpec(lines, widths=widths, freeze="B2", text_columns=True, validations=checks),
        "pages": SheetSpec(pages, widths=[26, 8, 50], text_columns=True,
                           validations=[_dv("list", f"B2:B{plast}", ["yes", "no"], "Done?",
                                            "yes when the page is completely transcribed and checked; only pages marked yes are used.",
                                            "yes or no", "Write yes or no.")]),
        "about": SheetSpec(about, widths=[18, 90]),
        "codebook": SheetSpec(codebook_table(cb), widths=[18, 10, 8, 70, 30]),
        "how_to": SheetSpec(pd.DataFrame({"how_to": how_to(cb, rows, a.get("prefilled") == "yes", a.get("images"))}), widths=[130]),
    }
    # sheets the user added to the file are kept
    write_xlsx(sheets, book["file"], keep_other_sheets=True, replaces=[book.get("bench_sheet")])


# ------- writing a text benchmark --------------------------------------------------------------

def text_intro(prefilled, folder) -> list[str]:
    out = ["Benchmark: your own transcription of the pages below, which score() compares with the model's output.",
           "Type the text of each page under its === line, exactly as written, following the same rules as the instructions you give the model.",
           "Leave a page empty until you have transcribed it: empty pages are left out. For a page with no writing at all, type [no text]."]
    if prefilled:
        out.append("The text under each line is the model's own output: correct it against the image, then change \"not done\" to \"done\" on its === line.")
    if folder:
        out.append(f"The page images are in {folder}")
    out += ["When you have finished, save, close the file, and run the same benchmark() line again.",
            "Keep the === lines and the record of the draw below as they are."]
    return out


def text_lines(book) -> list[str]:
    """The text benchmark: the package's notes, the record of the draw, the
    user's own lines from above the first === line (kept), then one === line
    per page and its text."""
    a = book["about"]
    rec = [f"{k}: {'' if is_missing(v) else v}" for k, v in a.items()]
    known = set(text_intro(True, a.get("images"))) | set(text_intro(False, a.get("images")))
    own = []
    for line in book.get("preamble") or []:
        m = re.match(r"^([a-z_]+):\s?(.*)$", line)
        if (m and m.group(1) in ABOUT_ITEMS) or line in known:
            continue
        own.append(line)
    while own and not own[0].strip():
        own.pop(0)
    while own and not own[-1].strip():
        own.pop()
    lines = book["lines"]
    text_of = {page_key(f): t for f, t in zip(lines["file"], lines["text"])} if len(lines) else {}
    body = []
    for f, d in zip(book["pages"]["file"], book["pages"]["done"]):
        dn = parse_done(d)
        status = "" if dn is None else (" done" if dn else " not done")
        txt = text_of.get(page_key(f))
        txt = "" if is_missing(txt) else str(txt)
        if not txt.strip() and dn is True:
            txt = "[no text]"
        body += [f"=== {f} ==={status}"] + (txt.split("\n") if txt else []) + [""]
    return text_intro(a.get("prefilled") == "yes", a.get("images")) + [""] + rec + [""] + (own + [""] if own else []) + body


# ------- reading: each kind of file as raw lines ------------------------------------------------------
#
# Every reader returns the same "raw" description, which build() turns into
# the benchmark: lines (text, the file's columns), loc(i, var, j) (where line
# i's value is, in words), pages (file, done, notes, line) or None, done_blank
# (what a blank done cell means: False, or None = "has values"), about,
# codebook, label, kind, date_cells, dec.

def raw_xlsx(file) -> dict:
    sheets = sheet_names(file)
    low = [s.strip().lower() for s in sheets]
    bench = next((s for s, l in zip(sheets, low) if l == "benchmark"), None) or next((s for s, l in zip(sheets, low) if l not in META_SHEETS), None)
    if bench is None:
        raise InputError(f"{os.path.basename(file)} has no sheet with the benchmark (expected a sheet called 'benchmark').")
    cells = read_cells(file, bench)
    grid = cells["grid"]
    head = ["" if h is None else str(h) for h in grid[0]] if grid else []
    body, dates, numbers = grid[1:], cells["dates"][1:], [[f is not None for f in r] for r in cells["formats"][1:]]
    lines = pd.DataFrame(body, columns=_unique(head), dtype=object) if head else pd.DataFrame()
    pages = None
    ps = next((s for s, l in zip(sheets, low) if l == "pages"), None)
    if ps is not None:
        ph, pb, _ = read_table(file, ps)
        p = pd.DataFrame(pb, columns=_unique(ph), dtype=object) if ph else pd.DataFrame()
        pc = match_columns(p.columns, ["file", "done", "notes"])
        if pc["done"] is None:
            pc["done"] = match_columns(p.columns, ["checked"])["checked"]
        if pc["file"] is None and ph:
            # without it every page with values was used, pages marked "no" included
            raise InputError(f"The pages sheet of {os.path.basename(file)} has no `file` column: its first row reads "
                             f"{', '.join(repr(str(h)) for h in ph if str(h).strip())}.",
                             "Its first row must be the header file, done, notes (a translated or sorted header is not "
                             "recognised): put it back, save, close Excel, and run the line again.")
        if pc["file"] is not None:
            def get(nm):
                j = pc[nm]
                return [None] * len(p) if j is None else [None if is_missing(v) else str(v) for v in p.iloc[:, j]]
            pages = pd.DataFrame({"file": [None if v is None else v.strip() for v in get("file")],
                                  "done": [None if v is None else v.strip() for v in get("done")], "notes": get("notes"),
                                  "_line": list(range(2, len(p) + 2))})
            pages = pages[[f is not None and f != "" for f in pages["file"]]].reset_index(drop=True)
    about = {}
    ab_s = next((s for s, l in zip(sheets, low) if l == "about"), None)
    if ab_s is not None:
        ah, ab, _ = read_table(file, ab_s)
        if len(ah) >= 2:
            for row in ab:
                it = row[0]
                if it is not None and str(it).strip():
                    about[str(it).strip()] = None if row[1] is None else str(row[1]).strip()
    cbk = None
    cs = next((s for s, l in zip(sheets, low) if l == "codebook"), None)
    if cs is not None:
        cbk = codebook_from_sheet(file, cs, about.get("codebook_source"), about.get("free_text"))

    def loc(i, var, j=None):
        return f"sheet {bench}, row {i + 2} ({var})" if j is None else f"sheet {bench}, cell {col_letter(j + 1)}{i + 2} ({var})"

    return {"lines": lines, "date_cells": dates, "number_cells": numbers, "pages": pages, "done_blank": False, "about": about,
            "codebook": cbk, "label": os.path.basename(file), "kind": "xlsx", "loc": loc,
            "page_loc": lambda line: f"sheet pages, row {line}", "bench_sheet": bench}


def _unique(head):
    out = []
    for h in head:
        name, k = h, 2
        while name in out and name:
            name = f"{h}.{k}"
            k += 1
        out.append(name)
    return out


def codebook_from_sheet(file, sheet, source=None, free_text=None):
    """The codebook kept in the file. A table codebook made with
    free_text=True lists the page's other text as a page variable; it is
    scored apart, so it is recognised (the about sheet records the choice;
    older files by the text variable's own description)."""
    with quietly():
        tab = read_codebook_sheet(file, sheet)
        if not len(tab):
            return None
        cbk = codebook_from_table(tab, source=source or file, other=False)
    v = cbk.vars
    if len(v) == 1 and v[0].variable == "text" and v[0].level == "page" and v[0].type == "text":
        return codebook_text_only()
    is_other = [x.variable == "text" and x.level == "page" and x.type == "text" for x in v]
    said = parse_yes_no(free_text)
    desc = next((x.description for x, o in zip(v, is_other) if o), None)
    if any(is_other) and len(v) > 1 and (said is True or (said is None and (desc or "").strip() == TEXT_DESCRIPTION_WITH_RECORDS)):
        vc = match_columns(tab.columns, ["variable"])["variable"]
        keep = [True] * len(tab) if vc is None else [is_missing(x) or str(x).strip() != "text" for x in tab.iloc[:, vc]]
        sub = tab[keep].reset_index(drop=True)
        sub.attrs.update(tab.attrs)
        with quietly():
            cbk = codebook_from_table(sub, source=source or file, other=False, free_text=True)
    if source:
        cbk = replace(cbk, source=source)
    return cbk


def raw_table(df: pd.DataFrame, file, kind, loc=None) -> dict:
    def text(x):
        if is_missing(x):
            return None
        if hasattr(x, "isoformat") and not isinstance(x, str):
            return x.isoformat()[:10] if hasattr(x, "hour") is False else str(x)
        return chr_value(x)

    lines = pd.DataFrame({str(c): [text(v) for v in df[c].tolist()] for c in df.columns}, dtype=object)
    label = os.path.basename(str(file))
    return {"lines": lines, "date_cells": None, "pages": None, "done_blank": False, "about": {}, "codebook": None,
            "label": label, "kind": kind, "loc": loc or (lambda i, var, j=None: f"row {i + 1} ({var})"), "page_loc": None}


def raw_csv(file) -> dict:
    label = os.path.basename(file)
    tab = read_delim_text(file)
    raw = raw_table(tab, file, "csv", loc=lambda i, var, j=None: f"line {i + 2} of {label} ({var})")
    raw["dec"] = "," if tab.attrs.get("sep") == ";" else "."
    return raw


def raw_dta(file) -> dict:
    df = pd.read_stata(file, convert_categoricals=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object)
    return raw_table(df, file, "dta")


def raw_json(file, cb) -> dict:
    try:
        x = read_json(file)
    except ValueError as e:
        raise InputError(f"{os.path.basename(file)} is not valid JSON.", ("x", str(e))) from None
    entries = []
    if isinstance(x, list) and x:
        for el in x:
            if not isinstance(el, dict) or el.get("file") is None:
                raise InputError(f"{os.path.basename(file)} is a list whose elements do not all have a `file`.",
                                 'Expected one object per page, as output() writes: [{"file": "p001.jpg", "data": {...}}, ...].')
            entries.append({"file": str(el["file"]), "data": el.get("data")})
    elif isinstance(x, dict) and x:
        names = [v.variable for v in cb.vars]
        for nm, v in x.items():
            d = v["data"] if isinstance(v, dict) and "data" in v and "data" not in names else v
            entries.append({"file": nm, "data": d})
    else:
        raise InputError(f"{os.path.basename(file)} holds no pages.")
    pv = [v.variable for v in cb.vars if v.level == "page"]
    rv = [v.variable for v in cb.vars if v.level == "row"]
    path = records_path(cb)
    cols = gold_columns(cb)

    def as_text(v):
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return to_json(v)
        if isinstance(v, bool):
            return "yes" if v else "no"
        s = chr_value(v)
        return "" if s is None else s

    data = {c: [] for c in cols}
    where, done_ = [], []
    for e in entries:
        done_.append(e["data"] is not None)
        if e["data"] is None:
            continue
        scal = object_scalars(e["data"], skip=tuple(path or ()))
        rr = [] if path is None else page_rows(e["data"], path, row_columns(cb))
        m = max(1, len(rr))
        block = {c: [""] * m for c in cols}
        block["file"] = [e["file"]] * m
        if rr:
            block["row"] = [str(k + 1) for k in range(len(rr))]
        for v in pv:
            block[v][0] = as_text(scal.get(v))
        if rr:
            for v in rv:
                block[v] = [as_text(r.get(v)) for r in rr]
        for c in cols:
            data[c] += block[c]
        where += [f"row {k + 1}" for k in range(len(rr))] if rr else ["page values"]
    lines = pd.DataFrame(data, columns=cols, dtype=object)
    files = [e["file"] for e in entries]
    label = os.path.basename(file)
    return {"lines": lines, "date_cells": None, "done_blank": False, "about": {}, "codebook": None, "label": label, "kind": "json",
            "pages": pd.DataFrame({"file": files, "done": ["yes" if d else "no" for d in done_], "notes": [None] * len(files),
                                   "_line": list(range(1, len(files) + 1))}),
            "loc": lambda i, var, j=None: f"{lines['file'].iloc[i]}, {where[i]} ({var})",
            "page_loc": lambda line: f"page {line} of {label}"}


_HEAD_RE = re.compile(r"^\s*={3,}\s*(.*?)\s*={3,}\s*(.*?)\s*$")


def raw_text(text: str, file) -> dict:
    """Free text: each page starts with a line "=== <file name> ===",
    optionally followed by "done" or "not done". Lines before the first page
    are the instructions and the record of the draw ("seed: 1930")."""
    label = os.path.basename(str(file))
    lines = text.split("\n")
    heads = [i for i, l in enumerate(lines) if _HEAD_RE.match(l) and _HEAD_RE.match(l).group(1).strip()]
    pre = lines[: heads[0]] if heads else lines
    about = {}
    for l in pre:
        m = re.match(r"^([a-z_]+):\s?(.*)$", l)
        if m and m.group(1) in ABOUT_ITEMS:
            about[m.group(1)] = m.group(2).strip() or None
    files = [_HEAD_RE.match(lines[i]).group(1).strip() for i in heads]
    # output(res, "x.txt") writes "=== file (source p. N) ===": drop the parenthesis
    files = [re.sub(r"\s*\([^()]*\bp\. ?[0-9]+\)$", "", f) for f in files]
    status = [_HEAD_RE.match(lines[i]).group(2).strip().lower() for i in heads]
    ends = [h - 1 for h in heads[1:]] + [len(lines) - 1]
    texts = []
    for k, h in enumerate(heads):
        body = [re.sub(r"[ \t]+$", "", l) for l in lines[h + 1: ends[k] + 1]]
        nz = [i for i, l in enumerate(body) if l.strip()]
        texts.append("\n".join(body[nz[0]: nz[-1] + 1]) if nz else "")
    no_text = [t.strip().lower() == "[no text]" for t in texts]
    texts = ["" if nt else t for t, nt in zip(texts, no_text)]
    done_ = []
    for st, nt in zip(status, no_text):
        if nt:
            done_.append("yes")
        elif st == "":
            done_.append(None)
        else:
            # a word that is neither done nor not done is kept as it is, so that
            # it is refused ("to do" once counted as blank)
            d = parse_done(st)
            done_.append("yes" if d is True else "no" if d is False else st)
    seen, dup, keep = set(), [], []
    for i, f in enumerate(files):
        k = page_key(f)
        if k in seen:
            dup.append(f)
        else:
            seen.add(k)
            keep.append(i)
    if dup:
        warn(f"{label} has more than one === line for {listing(list(dict.fromkeys(dup)))}; only the first is used.")
    files = [files[i] for i in keep]
    texts = [texts[i] for i in keep]
    done_ = [done_[i] for i in keep]
    return {"lines": pd.DataFrame({"file": files, "text": texts}, dtype=object),
            "pages": pd.DataFrame({"file": files, "done": done_, "notes": [None] * len(files), "_line": list(range(1, len(files) + 1))}),
            "done_blank": None, "about": about, "codebook": None, "date_cells": None, "label": label, "kind": "text", "preamble": pre,
            "loc": lambda i, var, j=None: f"the page {files[i]} in {label}",
            "page_loc": lambda line: f"the === line of {files[line - 1]}"}


# ------- reading: raw lines into the benchmark ------------------------------------------------------

def gold_read(file, cb=None, pages=None, rows=None, fallback=None):
    e = ext(file)
    if e in ("pkl", "pickle"):
        obj = load_pickle(file)
        if isinstance(obj, Gold) and obj.gold is not None:
            return obj
        if isinstance(obj, Result) and obj.info is not None:
            cb = cb or obj.info.codebook
            obj = pd.DataFrame(obj)
        if not isinstance(obj, pd.DataFrame):
            raise InputError(f"{os.path.basename(file)} does not hold a table (it holds a {type(obj).__name__}).")
        raw = raw_table(obj, file, "pkl")
    elif e == "json":
        cb = cb or fallback
        if cb is None:
            _need_codebook(file)
        raw = raw_json(file, cb)
    elif e in XLSX_EXTS:
        raw = raw_xlsx(file)
    elif e in ("csv", "tsv"):
        raw = raw_csv(file)
    elif e == "dta":
        raw = raw_dta(file)
    elif e in ("txt", "md"):
        raw = raw_text(read_text(file), file)
    elif e == "docx":
        raw = raw_text(read_docx_text(file), file)
    elif e == "rds":
        raise InputError(".rds is R's own format and cannot be read here.", "Save the benchmark as .xlsx or .csv from R, or read it with benchmark() in R.")
    else:
        raise InputError(f"{os.path.basename(file)} cannot be read as a benchmark.")
    if cb is None:
        cb = raw["codebook"]
    if cb is None and raw["kind"] == "text":
        cb = codebook_text_only()
    if cb is None:
        cb = fallback
    if cb is None:
        _need_codebook(file)
    if raw["kind"] == "text" and [v.variable for v in cb.vars] != ["text"]:
        other = [v.variable for v in cb.vars if v.variable != "text"]
        raise InputError(f"{os.path.basename(file)} holds only the text of each page, but the codebook also has {', '.join(other)}.",
                         "Keep a table benchmark in an Excel file, or read this one with codebook=codebook(free_text=True).")
    return build(raw, cb, image_files=None if pages is None else list(pages["file"]), rows=rows, source=show_path(file))


def _need_codebook(file):
    raise InputError(f"To read {os.path.basename(file)}, give the codebook it follows.",
                     f"For example benchmark({os.path.basename(file)!r}, codebook=cb).")


def as_gold(x, codebook=None) -> Gold:
    """For score(): a benchmark, or anything benchmark() can read. The codebook
    given is a fallback for files that carry none."""
    if isinstance(x, Gold) and x.gold is not None:
        return x
    cb = None if codebook is None else gold_as_codebook(codebook)
    x = user_path(x)
    if isinstance(x, (str, os.PathLike)):
        if not os.path.exists(x):
            raise NotFoundError(f"Cannot find the benchmark {show_path(x)}.",
                                f"Make it with benchmark(); relative paths start from your working folder, {os.getcwd()}.")
        return gold_read(str(x), fallback=cb)
    if isinstance(x, pd.DataFrame):
        if cb is None:
            raise InputError("A benchmark given as a DataFrame needs `codebook`.")
        raw = raw_table(pd.DataFrame(x), "data frame", "data frame")
        raw["label"] = "the DataFrame"
        return build(raw, cb, source=None)
    raise InputError("The benchmark must come from benchmark(), or be a benchmark file.")


def build(raw: dict, cb, image_files=None, rows=None, source=None) -> Gold:
    label = raw["label"]
    program = _program("xlsx" if raw["kind"] == "xlsx" else "txt" if raw["kind"] == "text" else raw["kind"])
    pv = [v.variable for v in cb.vars if v.level == "page"]
    rv = [v.variable for v in cb.vars if v.level == "row"]
    all_v = pv + rv
    types = {v.variable: v.type for v in cb.vars}
    cats = {v.variable: list(v.values) for v in cb.vars}
    has_rows = bool(rv)
    lines = raw["lines"]
    n_lines = len(lines)

    # 1. columns
    cols = match_columns(lines.columns, ["file", "row"] + all_v)
    if cols["file"] is None:
        raise InputError(f"{label} has no `file` column.",
                         f"Its columns are {', '.join(repr(str(c)) for c in lines.columns)}. The file column names the page image each line comes from.")
    miss = [v for v in all_v if cols[v] is None]
    if miss:
        raise InputError(f"{label} has no column for the codebook variable{'s' if len(miss) > 1 else ''} {', '.join(repr(m) for m in miss)}.",
                         f"Its columns are {', '.join(repr(str(c)) for c in lines.columns)}.",
                         "If the codebook changed after the benchmark was made, read it without `codebook` (an Excel benchmark keeps its "
                         "own), or add the missing columns.")
    used = {j for j in cols.values() if j is not None}
    extra = [str(c) for j, c in enumerate(lines.columns) if j not in used and str(c).strip() and str(c).strip().lower() not in QUIET_COLS]
    if extra:
        info(f"Ignoring column{'s' if len(extra) > 1 else ''} {', '.join(repr(e) for e in extra)} of {label} (not in the codebook).")

    def get(nm):
        j = cols[nm]
        if j is None:
            return [""] * n_lines
        return ["" if is_missing(v) else str(v).strip() for v in lines.iloc[:, j].tolist()]

    M = {v: get(v) for v in all_v}
    file_v = [nfc(f) for f in get("file")]
    row_v = get("row") if has_rows else [""] * n_lines
    any_val = [any(M[v][i] for v in all_v) for i in range(n_lines)]
    row_val = [any(M[v][i] for v in rv) for i in range(n_lines)]
    keep = [bool(file_v[i]) or bool(row_v[i]) or any_val[i] for i in range(n_lines)]

    def loc(i, var):
        return raw["loc"](i, var, cols.get(var))

    # 2. a file name on every line: added lines often leave it out
    last, filled = "", 0
    for i in range(n_lines):
        if not keep[i]:
            continue
        if file_v[i]:
            last = file_v[i]
        elif last:
            file_v[i] = last
            filled += 1
        else:
            raise InputError(f"{label}: {loc(i, 'file')} has values but no file name.",
                             "Write the page image's file name on every line, then run the line again.")
    if filled:
        info(f"{plural(filled, 'line')} of {label} had no file name; took the file of the line above.")
    drawn_names = split_drawn(raw["about"].get("drawn"))
    known = image_files or list(dict.fromkeys((list(raw["pages"]["file"]) if raw["pages"] is not None else []) + drawn_names)) or None
    file_v, unknown = canonical(file_v, known)
    if image_files is None:
        unknown = []

    # 3. which pages are done
    keys = [page_key(f) for f in file_v]
    shown = {}
    for i in range(n_lines):
        if keep[i] and keys[i] not in shown:
            shown[keys[i]] = file_v[i]
    has_val = {k: any(any_val[i] for i in range(n_lines) if keep[i] and keys[i] == k) for k in shown}
    prefilled = parse_yes_no(raw["about"].get("prefilled")) is True
    ps = raw["pages"]
    listed, explicit = [], False
    if ps is not None and len(ps):
        pfiles, punk = canonical(list(ps["file"]), known)
        if image_files is not None:
            unknown = list(dict.fromkeys(unknown + punk))
        dvals = list(ps["done"])
        parsed = [parse_done(d) for d in dvals]
        blank = [is_missing(d) or not str(d).strip() for d in dvals]
        check_done_words(raw)
        explicit = any(not b for b in blank)
        pk = [page_key(f) for f in pfiles]
        for k, f in zip(pk, pfiles):
            shown.setdefault(k, f)
        done_map = {}
        for i, k in enumerate(pk):
            if k in done_map:
                continue
            if blank[i]:
                done_map[k] = has_val.get(k, False) if raw["done_blank"] is None else bool(raw["done_blank"])
            else:
                done_map[k] = parsed[i]
        listed = list(done_map)
        outside = [k for k, hv in has_val.items() if hv and k not in done_map]
        if outside:
            say(("!", f"{plural(len(outside), 'page')} in the benchmark sheet of {label} {'is' if len(outside) == 1 else 'are'} not listed in "
                      f"its pages sheet and {'is' if len(outside) == 1 else 'are'} left out: {listing([shown[k] for k in outside])}."),
                ("i", "Add them to the pages sheet with done = yes to use them."))
        for k in shown:
            done_map.setdefault(k, False)
    else:
        done_map = dict(has_val)
    if not any(done_map.values()):
        pool = [k for k, hv in has_val.items() if hv and (not listed or k in listed)]
        if pool and prefilled:
            raise GoldEmptyError(f"No page of {label} is marked done yet.",
                                 "The benchmark was prefilled with the model's output, so pages count only once you have corrected them and marked them done.",
                                 "Change \"not done\" to \"done\" on the === line of each corrected page." if raw["kind"] == "text" else
                                 "Write yes in the done column of the pages sheet for each corrected page.")
        if pool and explicit:
            raise GoldEmptyError(f"No page of {label} is marked done yet.",
                                 "Only pages marked done are used, so that a page you have not finished typing does not count against the model.",
                                 ("Change \"not done\" to \"done\" on the === line of each finished page" if raw["kind"] == "text" else
                                  "Write yes in the done column of the pages sheet for each finished page") +
                                 f" (pages with values: {listing([shown[k] for k in pool])}), save, close {program}, and run the same line again.")
        if pool:
            for k in pool:
                done_map[k] = True
            say(("!", f"No page of {label} is marked done; using the {plural(len(pool), 'page')} with values: {listing([shown[k] for k in pool])}."),
                ("i", "Add \"done\" after a page's === line to choose pages yourself." if raw["kind"] == "text" else
                      "Write yes in the done column of the pages sheet to choose them yourself."))
        else:
            todo = [shown[k] for k in (listed or shown)]
            raise GoldEmptyError(f"{label} has no values yet" + (f" for {listing(todo)}" if todo else "") + ".",
                                 f"Fill them in, save, close {program}, and run the same line again.")
    elif listed:
        waiting = [k for k, hv in has_val.items() if hv and not done_map.get(k) and k in listed]
        if waiting:
            info(f"Using the {plural(sum(done_map.values()), 'page')} marked done; {plural(len(waiting), 'other page')} with values "
                 f"{'is' if len(waiting) == 1 else 'are'} left out until marked done: {listing([shown[k] for k in waiting])}.")
    done_keys = [k for k, d in done_map.items() if d]
    done_keys = [done_keys[i] for i in gold_order([shown[k] for k in done_keys])]

    # 4. for each done page: the line holding each page value, and its rows
    dates = raw.get("date_cells")
    row_j = cols.get("row")
    row_dates = [bool(dates[i][row_j]) if (has_rows and dates is not None and row_j is not None and i < len(dates) and row_j < len(dates[i])) else False
                 for i in range(n_lines)]
    errors, conflicts, renumbered, empty_done = [], [], [], []
    page_cell = {v: [None] * len(done_keys) for v in pv}
    row_line, row_num, row_page = [], [], []
    for p, dk in enumerate(done_keys):
        li = [i for i in range(n_lines) if keep[i] and keys[i] == dk]
        if not li or not any(any_val[i] for i in li):
            empty_done.append(shown[dk])
        for v in pv:
            nz = [i for i in li if M[v][i]]
            if not nz:
                continue
            page_cell[v][p] = nz[0]
            differ = [i for i in nz[1:] if M[v][i] != M[v][nz[0]]]
            if differ:
                conflicts.append(f"{shown[dk]}, {v}: \"{M[v][nz[0]]}\" ({loc(nz[0], v)}) but \"{M[v][differ[0]]}\" ({loc(differ[0], v)})")
        if not has_rows:
            continue
        rl = [i for i in li if row_v[i] or row_val[i]]
        if not rl:
            continue
        given = [bool(row_v[i]) for i in rl]
        num = [as_number(row_v[i]) for i in rl]
        stamped = [g and row_dates[i] for g, i in zip(given, rl)]
        bad = [g and not s and (math.isnan(x) or x < 1 or x != int(x)) for g, s, x in zip(given, stamped, num)]
        for k, i in enumerate(rl):
            if stamped[k]:
                errors.append(f"{loc(i, 'row')}: Excel turned what was typed into a date (it reads \"{row_v[i]}\"); type the row number again")
            elif bad[k]:
                errors.append(f"{loc(i, 'row')}: \"{row_v[i]}\" is not a row number (1, 2, 3, ...)")
        badall = [b or s for b, s in zip(bad, stamped)]
        if all(given) and not any(badall):
            seen_n = {}
            for k, x in enumerate(num):
                seen_n.setdefault(x, []).append(rl[k])
            for x, where in seen_n.items():
                if len(where) > 1:
                    errors.append(f"{shown[dk]}: row number {int(x)} is used on more than one line ({'; '.join(loc(i, 'row') for i in where)})")
            order = sorted(range(len(rl)), key=lambda k: num[k])
            rl = [rl[k] for k in order]
            nums = [int(num[k]) for k in order]
        else:
            if any(given):
                renumbered.append(shown[dk])
            nums = list(range(1, len(rl) + 1))
        row_line += rl
        row_num += nums
        row_page += [p] * len(rl)

    # 5. values converted to the codebook's types, every failure with its cell
    page_vals, row_vals = {}, {}
    case_fixed, off_list, thousands = [], [], []
    damage = {"serial": [], "sci": [], "zeros": []}
    dec = raw.get("dec") or "."
    # "1.250" is 1.25 with a decimal point and 1250 in Spanish or Catalan
    # notation: read as the first, and said
    thousands_re = re.compile(r"[-+]?[0-9]{1,3}" + re.escape("." if dec == "." else ",") + r"[0-9]{3}")
    numbers = raw.get("number_cells")
    for v in all_v:
        idx = page_cell[v] if v in pv else row_line
        vals = ["" if i is None else M[v][i] for i in idx]
        locs = ["" if i is None else loc(i, v) for i in idx]
        j = cols[v]
        dts = [bool(dates[i][j]) if (dates is not None and i is not None and i < len(dates) and j < len(dates[i])) else False for i in idx]
        nums = [bool(numbers[i][j]) if (numbers is not None and i is not None and i < len(numbers) and j < len(numbers[i])) else False
                for i in idx]
        conv = gold_convert(vals, types[v], cats[v], dec=dec)
        if types[v] == "number":
            thousands += [f"{locs[k]} reads \"{vals[k]}\"" for k in range(len(vals))
                          if vals[k] and thousands_re.fullmatch(vals[k].strip()) and conv["problem"][k] is None]
        stamped = [types[v] in ("number", "integer", "boolean") and d for d in dts]
        for k in range(len(idx)):
            if stamped[k]:
                errors.append(f"{locs[k]}: Excel turned what was typed into a date (it reads \"{vals[k]}\"); type the value again as "
                              f"{'yes or no' if types[v] == 'boolean' else 'a number'}")
            elif conv["problem"][k] is not None:
                errors.append(f"{locs[k]}: \"{vals[k]}\" {conv['problem'][k]}")
            if conv["case_fixed"][k]:
                case_fixed.append(f"\"{vals[k]}\" -> \"{conv['value'][k]}\"")
            if conv["unknown"][k]:
                off_list.append(f"{locs[k]}: \"{vals[k]}\" is not one of {listing(cats[v], 8)}")
        if types[v] == "text" and raw["kind"] in ("xlsx", "csv"):
            dmg = excel_damage(vals, dts, nums)
            for kind in damage:
                damage[kind] += [f"{locs[k]} reads \"{vals[k]}\"" for k in range(len(vals)) if dmg[kind][k]]
        (page_vals if v in pv else row_vals)[v] = conv["value"]
    if errors:
        raise InputError(f"{plural(len(errors), 'value')} in {label} cannot be read as the codebook says:",
                         *[("x", e) for e in errors[:10]],
                         *([("x", f"... and {len(errors) - 10} more.")] if len(errors) > 10 else []),
                         ("i", f"Correct {'it' if len(errors) == 1 else 'them'}, save, close {program}, and run the same line again."),
                         ("i", "Numbers: digits, with a point for decimals (1234.5). Yes/no: yes or no. Leave a cell empty when the page has nothing there."))
    read_warnings(label, unknown, image_files, conflicts, renumbered, case_fixed, off_list, damage,
                  empty_done if raw["kind"] == "xlsx" else [], thousands, dec)

    # 6. the table, in the results layout: page values repeated on every row
    out_file, out_row, out_page, out_line = [], [], [], []
    for p, dk in enumerate(done_keys):
        here = [k for k, pp in enumerate(row_page) if pp == p]
        if has_rows and here:
            out_file += [shown[dk]] * len(here)
            out_row += [row_num[k] for k in here]
            out_page += [p] * len(here)
            out_line += here
        else:
            out_file.append(shown[dk])
            out_row.append(None)
            out_page.append(p)
            out_line.append(None)
    tab = {"file": pd.Series(out_file, dtype=object)}
    if has_rows:
        tab["row"] = pd.Series(pd.array(out_row, dtype="Int64"))
    for v in pv:
        tab[v] = _typed([page_vals[v][p] for p in out_page], types[v])
    for v in rv:
        tab[v] = _typed([None if k is None else row_vals[v][k] for k in out_line], types[v])
    tab = pd.DataFrame(tab)
    all_keys = list(done_map)
    pages_df = pd.DataFrame({"file": [shown[k] for k in all_keys], "done": [bool(done_map[k]) for k in all_keys]})
    nrows = []
    for k in all_keys:
        if not done_map[k]:
            nrows.append(None)
        else:
            p = done_keys.index(k)
            nrows.append(sum(1 for pp in row_page if pp == p) if has_rows else 1)
    pages_df["n_rows"] = pd.array(nrows, dtype="Int64")
    pages_df = pages_df.iloc[gold_order(list(pages_df["file"]))].reset_index(drop=True)
    a = raw["about"]
    d = None
    if a.get("drawn"):
        drawn = split_drawn(a["drawn"])
        sd = as_number(a.get("seed"))
        nn = as_number(a.get("n"))
        d = {"seed": None if math.isnan(sd) else sd, "files": drawn, "n": len(drawn) if math.isnan(nn) else int(nn),
             "date": a.get("made"), "method": a.get("method")}
    rf = as_number(a.get("rows"))
    rows_file = None if math.isnan(rf) else int(rf)
    if rows is None and rows_file is not None:
        rows = rows_file
    if rows is not None and rows_file is not None and rows != rows_file:
        info(f"{label} records that only the first {rows_file} rows of each page were to be typed; using rows={rows} as you asked.")
    return new_gold(tab, cb, pages_df, draw=d, rows=rows, prefilled=prefilled, source=source)


def _typed(vals, t):
    if t == "integer":
        if all(v is None or (isinstance(v, (int, float)) and float(v) == int(v) and abs(v) < 2 ** 63) for v in vals):
            return pd.Series(pd.array([None if v is None else int(v) for v in vals], dtype="Int64"))
        return pd.Series([math.nan if v is None else v for v in vals], dtype="float64")
    if t == "number":
        return pd.Series([math.nan if v is None else v for v in vals], dtype="float64")
    if t == "boolean":
        return pd.Series(pd.array(vals, dtype="boolean"))
    return pd.Series(vals, dtype=object)


def gold_convert(vals, t, cats=(), dec="."):
    """The text of one variable's cells as its codebook type, the problem of
    each cell (None when fine), and category values matched ignoring case and
    accents, or matching nothing."""
    n = len(vals)
    empty = [is_missing(v) or v == "" for v in vals]
    problem = [None] * n
    case_fixed = [False] * n
    unknown = [False] * n
    if t in ("number", "integer"):
        # a whole number typed in an integer column is read exactly, at any
        # length (through float, 16-digit numbers came out equal)
        x = [math.nan if empty[i] else int(str(vals[i]).strip()) if t == "integer" and re.fullmatch(r"\s*[-+]?[0-9]+\s*", str(vals[i]))
             else as_number(vals[i], dec) for i in range(n)]
        for i in range(n):
            if empty[i] or isinstance(x[i], int):
                continue
            if not math.isfinite(x[i]):
                problem[i] = "is not a whole number" if t == "integer" else "is not a number"
                x[i] = math.nan
            elif t == "integer" and x[i] != int(x[i]):
                problem[i] = "is not a whole number"
                x[i] = math.nan
        value = [v if isinstance(v, int) else None if math.isnan(v) else (int(v) if t == "integer" else v) for v in x]
    elif t == "boolean":
        value = [None if empty[i] else parse_yes_no(vals[i]) for i in range(n)]
        for i in range(n):
            if not empty[i] and value[i] is None:
                problem[i] = "is not yes or no"
    else:
        value = [None if empty[i] else nfc(vals[i]) for i in range(n)]
        if t == "category" and cats:
            folded = [fold_accents(c.strip()).lower() for c in cats]
            for i in range(n):
                if empty[i] or value[i] in cats:
                    continue
                f = fold_accents(value[i].strip()).lower()
                if f in folded:
                    case_fixed[i] = True
                    value[i] = cats[folded.index(f)]
                else:
                    unknown[i] = True
    return {"value": value, "problem": problem, "case_fixed": case_fixed, "unknown": unknown}


def excel_damage(vals, dates=None, numbers=None) -> dict:
    """Signs that Excel changed what was typed in a text column: a date serial,
    a number in scientific notation or one Excel stored as a number with more
    than the 15 digits it keeps (a long identifier: the rest came back as
    zeros), digit strings shorter than the zero-padded ones beside them (007
    typed, 7 kept)."""
    n = len(vals)
    nz = [not is_missing(v) and v != "" for v in vals]
    serial = [nz[i] and re.fullmatch(r"[0-9]{5}", vals[i]) is not None and 20000 <= float(vals[i]) <= 60000 for i in range(n)]
    if dates is not None:
        serial = [serial[i] or (nz[i] and bool(dates[i])) for i in range(n)]
    sci = [nz[i] and re.fullmatch(r"[-+]?[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+", vals[i]) is not None for i in range(n)]
    if numbers is not None:
        sci = [sci[i] or (nz[i] and bool(numbers[i]) and len(re.sub(r"[^0-9]", "", vals[i])) > 15) for i in range(n)]
    digits = [nz[i] and re.fullmatch(r"[0-9]+", vals[i]) is not None for i in range(n)]
    padded = [digits[i] and vals[i].startswith("0") and len(vals[i]) > 1 for i in range(n)]
    zeros = [False] * n
    if any(padded):
        longest = max(len(vals[i]) for i in range(n) if padded[i])
        zeros = [digits[i] and not vals[i].startswith("0") and len(vals[i]) < longest for i in range(n)]
    return {"serial": serial, "sci": sci, "zeros": [z and not s for z, s in zip(zeros, serial)]}


def read_warnings(label, unknown, image_files, conflicts, renumbered, case_fixed, off_list, damage, empty_done,
                  thousands=(), dec=".") -> None:
    if thousands:
        k = len(thousands)
        mark = "point" if dec == "." else "comma"
        warn(f"{plural(k, 'number')} in {label} could be thousands written with a {mark} (1{dec}250 for 1250): "
             f"{listing(thousands, 3)}.",
             f"{'It is' if k == 1 else 'They are'} read with the {mark} as the decimal mark (1{dec}250 as 1.25). If "
             f"{'it means' if k == 1 else 'they mean'} thousands, type the number without the {mark} (1250).")
    if unknown and image_files is not None:
        warn(f"{plural(len(unknown), 'file name')} in {label} {'is' if len(unknown) == 1 else 'are'} not among the page images: {listing(unknown)}.",
             f"The file column must hold the page image names (e.g. {image_files[0]}); check the spelling.")
    if conflicts:
        warn("Some page variables have different values on different lines of the same page; the first is used:",
             *[("*", c) for c in conflicts[:5]], *([("*", f"... and {len(conflicts) - 5} more.")] if len(conflicts) > 5 else []))
    if renumbered:
        info(f"The row column was incomplete on {listing(renumbered)}; rows were numbered in the order of the lines.")
    if case_fixed:
        info(f"Matched {plural(len(case_fixed), 'category value')} to the codebook's spelling ({listing(list(dict.fromkeys(case_fixed)), 3)}).")
    if off_list:
        k = len(off_list)
        warn(f"{plural(k, 'category value')} in {label} {'is' if k == 1 else 'are'} not in the codebook's list; kept as typed (the model "
             f"cannot give {'it' if k == 1 else 'them'}, so {'it counts' if k == 1 else 'they count'} as a difference):",
             *[("*", o) for o in off_list[:5]], *([("*", f"... and {k - 5} more.")] if k > 5 else []))
    what = {"serial": "look like dates Excel made out of what was typed (e.g. 1-2 or 3/4)",
            "sci": "are long numbers Excel shows in scientific notation, which loses digits",
            "zeros": "look like numbers that lost their leading zeros (others in the column start with 0)"}
    msg = [("*", f"{plural(len(damage[k]), 'value')} {what[k]}: {listing(damage[k], 3)}.") for k in what if damage[k]]
    if msg:
        warn(f"Excel may have changed some text values in {label}:", *msg,
             "Retype them with an apostrophe first ('1-2, '007) or format the column as Text, then run the line again.")
    if empty_done:
        k = len(empty_done)
        warn(f"{plural(k, 'page')} marked done in {label} {'has' if k == 1 else 'have'} no values at all: {listing(empty_done)}.",
             "They count as pages with nothing written on them. If that is wrong, fill them in or mark them done = no.")


# ------- the editor's way back in --------------------------------------------------------------

def after_app(file, cb, pages):
    """Run the editor on a benchmark file, then read what was saved. Nothing
    marked done yet is not an error here: the typing is kept for next time."""
    from ._editor import run_editor

    run_editor(file, cb, pages)
    try:
        return gold_read(file, cb, pages)
    except GoldEmptyError:
        info(f"No page is marked done yet. What you typed is saved in {show_path(file)}; run the same line again to continue.")
        return None

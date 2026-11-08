"""score(): how close the model's output is to your own transcription (the
benchmark), in three rates:

    rows        rows whose every compared row variable matches / row pairs
                (extra rows in the output enlarge the denominator)
    values      matching values / values compared (row variables in every
                pair, page variables once per page)
    characters  1 - sum(edits) / sum(max(len(gold), len(output))), text
                variables only, edits counted only for wrong values, empty
                counted as ""

Judgment calls (the R package's, each argued where it is made):

- Strict by default: text is compared exactly after Unicode NFC, with ""
  counted as empty. Lenient rates (``normalize=True``) are reported beside
  the strict ones, never instead of them.
- A benchmark page with no record in the output is "not run" and left out of
  every rate; a page that failed counts as entirely wrong, so the hardest
  page never drops out silently and flatters the model.
- A row present on one side only is paired with a MISSING sentinel, never
  with a row of empty values (which would match every empty gold cell).
- Rows are paired by position when both sides have the same number of rows
  and by content when they differ (``align="auto"``): one skipped row must
  not turn every later row of the page into a mismatch.
- A benchmark of the first N rows only says nothing about where the page
  ends: a page whose benchmark reaches N rows is always paired by content,
  and the output's rows after the one paired with the benchmark's last row
  are left out (they lie beyond what was typed).
- An empty denominator gives None, never 100%.
"""

from __future__ import annotations

import math
import os
import re
import string
import sys
import unicodedata

import numpy as np
import pandas as pd

from ._codebook import subset
from ._errors import InputError, NotFoundError
from ._gold import Gold, as_gold, match_columns, parse_yes_no
from ._result import Result, as_result, result_from_folder, result_units
from ._say import plural, say
from ._text import as_number, chr_value, fmt_int, fold_accents, is_missing, natural_order, nfc, page_key, show_path, user_path

try:  # optional speed-up for long texts; the results are the same
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:  # pragma: no cover
    _RF = None


class Score:
    """The comparison: ``summary``, ``by_variable``, ``by_page``,
    ``differences`` (every wrong value), ``not_run`` and ``notes``. Print it
    for a short summary in plain words."""

    def __init__(self, summary, by_variable, by_page, differences, not_run, notes, cells, about):
        self.summary = summary
        self.by_variable = by_variable
        self.by_page = by_page
        self.differences = differences
        self.not_run = not_run
        self.notes = notes
        self.cells = cells
        self.about = about

    def __getitem__(self, name):
        return getattr(self, name)

    def __repr__(self) -> str:
        return format_score(self)

    __str__ = __repr__


def score(benchmark, output, *, align: str = "auto", normalize: bool = False, report=None) -> Score:
    """Measure how accurate the extraction is.

    Compares the model's output with your benchmark (your own transcription
    of a few pages, from :func:`benchmark`) and says how much of it is right:
    rows with every value right, values right, and characters right (text
    variables), plus every difference. Pages are paired by file name; on each
    page the rows are paired by position when both sides have as many rows,
    by content otherwise (``align``), so one row the model skipped counts as
    one missing row. Values are compared after converting both sides to the
    codebook's type. For free text the figures are characters and words right.

    Give two results to compare two runs (the first is the reference): where
    they disagree shows where to look, not which is right.

    By default the report (``accuracy.html``, or ``agreement.html`` for two
    runs) is saved in the output's results folder; ``report=False`` writes
    none, and a file name writes it there.

    Parameters
    ----------
    benchmark : Gold, path, Result or DataFrame
        Your benchmark (the result of benchmark(), or its file), another
        result to compare with, or a table in the results layout.
    output : Result or results folder
        The results to check.
    align : str
        "auto" (default), "position" or "content".
    normalize : bool
        Also report lenient figures that ignore case, accents, punctuation,
        spacing and doubt marks ("?"). Numbers are never changed.
    report : path or False, optional
        Where to write the HTML report.
    """
    benchmark, output, report = user_path(benchmark), user_path(output), user_path(report)
    if align not in ("auto", "position", "content"):
        raise InputError("`align` must be one of 'auto', 'position', 'content'.", ("x", f"You gave {align!r}."))
    if not isinstance(normalize, bool):
        raise InputError("`normalize` must be True or False.")
    if report is not None and report is not False:
        if not isinstance(report, (str, os.PathLike)) or os.path.splitext(str(report))[1].lower() not in (".html", ".htm"):
            raise InputError("The report must be an HTML file.", "For example report=\"accuracy.html\".")
    if isinstance(output, Gold):
        raise InputError("Your benchmark goes first and the results second.", "For example score(gold, res).")
    out = output if isinstance(output, Result) and output.info is not None else as_result(output)
    out_info = out.info
    # score() compares whole pages from the run's records, so a selection of
    # rows was scored as the whole run, without a word
    if len(out) != out_info.n_rows:
        raise InputError(f"score() compares whole pages, and this is a selection of {fmt_int(len(out))} of the result's "
                         f"{fmt_int(out_info.n_rows)} rows.",
                         "Give the whole result. To leave pages out, select them in the benchmark instead, e.g. "
                         "score(gold[gold.file != 'p002.jpg'], res).")
    out_units = result_units(out)
    ref = score_reference(benchmark, out_info.codebook)
    # the page's other text is scored apart; an older benchmark file may have
    # lost the choice, and the results' codebook still knows
    rcb = ref["codebook"]
    if not rcb.free_text and out_info.codebook.free_text and rcb.kind == "table" and "text" in rcb.names("page"):
        from dataclasses import replace
        ref["codebook"] = replace(rcb, free_text=True)
    variables, vnotes = score_variables(ref["codebook"], out_info.codebook, ref["labels"])
    s = score_units(ref["units"], out_units, ref["codebook"], align=align, normalize=normalize, rows=ref["rows"],
                    variables=variables, labels=ref["labels"])
    a = s.about
    a.update(reference=ref["kind"], model=out_info.settings.get("model"), reference_model=ref.get("model"),
             source=ref.get("source"), prefilled=ref.get("prefilled"), draw=ref.get("draw"), output_folder=out_info.folder)
    scored = [f for f, st in zip(s.by_page["file"], s.by_page["status"]) if st != "not run"]
    sk = {page_key(f) for f in scored}
    dates = sorted({str(r.get("date"))[:10] for r in out_info.records.values() if page_key(r["file"]) in sk and r.get("date")})
    a["output_dates"] = [dates[0], dates[-1]] if dates else None
    first, rest = s.notes[: a["n_first"]], s.notes[a["n_first"]:]
    notes = ([("Your benchmark was pre-filled with the model's own output, so these figures are optimistic: whoever checks a page "
               "tends to keep what is already written.")] if ref.get("prefilled") else []) + first + ref["notes"] + vnotes
    others = {page_key(k) for k in out_units} - {page_key(k) for k in ref["units"]}
    if others:
        notes.append(f"The results hold {plural(len(others), 'other page')} that {'are' if len(others) != 1 else 'is'} not in "
                     f"{ref['labels']['ref']}; only the {plural(len(scored), 'page')} in both {'are' if len(scored) != 1 else 'is'} scored.")
    s.notes = notes + rest
    if report is None and out_info.folder is not None:
        report = os.path.join(out_info.folder, "agreement.html" if ref["kind"] == "run" else "accuracy.html")
    if report:
        from ._report import write_score_report
        write_score_report(s, str(report))
        a["report"] = str(report)
    return s


# ------- the reference side --------------------------------------------------------

BENCH_LABELS = {"ref": "your benchmark", "out": "the output", "ref_col": "benchmark", "out_col": "output"}
RUN_LABELS = {"ref": "the first run", "out": "the second run", "ref_col": "first run", "out_col": "second run"}


def score_reference(x, out_cb) -> dict:
    if isinstance(x, (str, os.PathLike)) and os.path.isdir(x):
        x = result_from_folder(x)
    if isinstance(x, Result) and x.info is not None:
        if len(x) != x.info.n_rows:
            raise InputError(f"score() compares whole pages, and the first result is a selection of {fmt_int(len(x))} of its "
                             f"{fmt_int(x.info.n_rows)} rows.", "Give the whole result.")
        units = result_units(x)
        ok = {k: u for k, u in units.items() if u["status"] in ("ok", "check")}
        notes = []
        bad = [k for k in units if k not in ok]
        if bad:
            notes.append(f"The first run has no data for {plural(len(bad), 'page')} (failed): {_list(bad)}. "
                         f"{'It is' if len(bad) == 1 else 'They are'} left out.")
        if not ok:
            raise InputError("The first result has no pages with data to compare against.",
                             "Give a benchmark from benchmark(), or a result whose pages were extracted.")
        return {"units": ok, "codebook": x.info.codebook, "rows": None, "prefilled": False, "kind": "run",
                "model": x.info.settings.get("model"), "source": x.info.folder, "draw": None, "labels": RUN_LABELS, "notes": notes}
    if isinstance(x, (str, os.PathLike)):
        if not os.path.exists(x):
            raise NotFoundError(f"Cannot find the benchmark file {show_path(x)}.",
                                f"Make one with benchmark({os.path.basename(str(x))!r}, images=pages, codebook=cb), fill it in, then score again.",
                                f"Relative paths start from your working folder, {os.getcwd()}.")
        x = as_gold(str(x), codebook=out_cb)
    if isinstance(x, Gold) and x.gold is None:
        x = pd.DataFrame(x)
    if isinstance(x, pd.DataFrame) and not isinstance(x, Gold):
        x = score_table_as_gold(x, out_cb)
    if isinstance(x, Gold):
        g = x.gold
        pages = g["pages"]
        tab = pd.DataFrame(x)
        # only the variables the table still has a column for
        cb = subset(g["codebook"], list(tab.columns))
        done = [f for f, d in zip(pages["file"], pages["done"]) if d]
        if not done:
            raise InputError("No page of your benchmark is marked as done.",
                             "Mark the pages you have finished with yes in the done column of the pages sheet, save, and score again.")
        present = {page_key(f) for f in tab["file"].unique()} if "file" in tab.columns else set()
        gone = [f for f in done if page_key(f) not in present]
        done = [f for f in done if page_key(f) in present]
        if not done:
            raise InputError("None of the pages marked done is in the benchmark table you gave.",
                             f"Pages marked done: {', '.join(gone[:4])}{' ...' if len(gone) > 4 else ''}",
                             "Score the whole benchmark, or keep at least one of those pages in the table.")
        from ._flatten import units_from_table
        units = units_from_table(tab, cb)
        uk = {page_key(k): k for k in units}
        out_units = {f: units[uk[page_key(f)]] for f in done}
        notes = []
        n_left = int((~pages["done"].astype(bool)).sum())
        if n_left:
            notes.append(f"{plural(n_left, 'page')} of your benchmark {'is' if n_left == 1 else 'are'} not marked as done and "
                         f"{'was' if n_left == 1 else 'were'} left out.")
        if gone:
            notes.append(f"{plural(len(gone), 'page')} marked done in your benchmark {'has' if len(gone) == 1 else 'have'} no lines in the table "
                         f"you gave and {'was' if len(gone) == 1 else 'were'} left out: {_list(gone)}.")
        if any(v.level == "row" for v in cb.vars):
            pk = {page_key(f): n for f, n in zip(pages["file"], pages["n_rows"])}
            cut = []
            for f in done:
                now = out_units[f]["n"]
                then = pk.get(page_key(f))
                if then is not None and not is_missing(then) and now < int(then):
                    cut.append(f"{f} ({now} of {int(then)})")
            if cut:
                notes.append(f"The benchmark table you gave has fewer rows than your benchmark on {_list(cut)}; the output's other rows "
                             "there count as rows not in your benchmark.")
        if g.get("rows"):
            notes.append(f"Your benchmark covers the first {g['rows']} rows of each page: where the page goes on, the output was compared "
                         "up to the row that pairs with your last one, and its later rows were left out.")
        return {"units": out_units, "codebook": cb, "rows": g.get("rows"), "prefilled": bool(g.get("prefilled")), "kind": "benchmark",
                "model": None, "source": g.get("source"), "draw": g.get("draw"), "labels": BENCH_LABELS, "notes": notes}
    raise InputError("`benchmark` must be your benchmark: the result of benchmark() or the file it wrote.",
                     "For example score('gold.xlsx', res). To compare two runs, give two results: score(res1, res2).")


def score_table_as_gold(x: pd.DataFrame, out_cb) -> Gold:
    """A table given as the benchmark goes through benchmark()'s own reader."""
    if match_columns(x.columns, ["file"])["file"] is None:
        raise InputError("The benchmark table needs a `file` column naming the page image of each row.",
                         f"Its columns are: {', '.join(map(str, x.columns))}.")
    cols = match_columns(x.columns, [v.variable for v in out_cb.vars])
    have = [k for k, j in cols.items() if j is not None]
    if not have:
        raise InputError("The benchmark table has no column for any variable of the results.",
                         f"Its columns are: {', '.join(map(str, x.columns))}.",
                         f"Result variables: {', '.join(v.variable for v in out_cb.vars)}.")
    return as_gold(x, codebook=subset(out_cb, have))


def score_variables(ref_cb, out_cb, labels):
    rv = ref_cb.vars
    ov = {v.variable: v for v in out_cb.vars}
    notes = []
    only_ref = [v.variable for v in rv if v.variable not in ov]
    only_out = [n for n in ov if n not in {v.variable for v in rv}]
    if only_ref:
        notes.append(f"Not compared (only in {labels['ref']}): {', '.join(only_ref)}.")
    if only_out:
        notes.append(f"Not compared (only in {labels['out']}): {', '.join(only_out)}.")
    other_level = [v.variable for v in rv if v.variable in ov and ov[v.variable].level != v.level]
    if other_level:
        notes.append(f"Not compared (a page value on one side and a row value on the other): {', '.join(other_level)}.")
    keep = [v for v in rv if v.variable in ov and ov[v.variable].level == v.level]
    if not keep:
        raise InputError("Your benchmark and the results have no variables in common.",
                         f"Benchmark variables: {', '.join(v.variable for v in rv)}.",
                         f"Result variables: {', '.join(ov)}.", "Score with the codebook the results were extracted with.")
    differ = [v for v in keep if v.type != ov[v.variable].type]
    if differ:
        notes.append(f"Compared with the type in {labels['ref']}'s codebook: " +
                     ", ".join(f"{v.variable} ({v.type}, {ov[v.variable].type} in {labels['out']})" for v in differ) + ".")
    return [v.variable for v in keep], notes


# ------- values ----------------------------------------------------------------------

_PUNCT_RE = None
_ASCII_SPACE = "[ \t\n\r\f\v]"


def _punct_re():
    global _PUNCT_RE
    if _PUNCT_RE is None:
        chars = set(string.punctuation)
        for cp in range(sys.maxunicode + 1):
            if unicodedata.category(chr(cp))[0] == "P":
                chars.add(chr(cp))
        cls = "".join(re.escape(c) for c in sorted(chars))
        _PUNCT_RE = re.compile(f"(?<![0-9])[{cls}]+|[{cls}]+(?![0-9])")
    return _PUNCT_RE


def lenient(x):
    """Capital letters and accents folded, punctuation turned into a space (a
    doubt mark "?" is punctuation), spacing collapsed. Punctuation between two
    digits is kept, so "1.5" never matches "15"."""
    if x is None:
        return None
    y = unicodedata.normalize("NFC", x.casefold())
    y = fold_accents(y)
    y = "".join(ch for ch in unicodedata.normalize("NFD", y) if unicodedata.category(ch)[0] != "M")
    y = unicodedata.normalize("NFC", y).replace(" ", " ")
    y = _punct_re().sub(" ", y)
    y = re.sub(f"{_ASCII_SPACE}+", " ", y).strip(" \t\r\n")
    return y or None


def lenient_key(x):
    """Lenient forms compared without their spaces, except a space between two
    digits: punctuation inside a word costs nothing ("Ul·ldemolins" matches
    "Ulldemolins"), while "1 5" and "15" still differ."""
    if x is None:
        return None
    return re.sub(r"(?<![0-9]) | (?![0-9])", "", x)


def _empty_none(s):
    return None if s is None or s == "" else s


_INT_TEXT_RE = re.compile(r"[-+]?[0-9]+")


def number_key(x) -> str:
    """A number as compared: whole numbers exactly, digit for digit (through
    float, 1234567890123456 and 1234567890123457 were the same); other
    numbers to 15 significant digits."""
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return str(int(x))
    x = float(x)
    if math.isfinite(x) and x.is_integer():
        return str(int(x))
    return chr_value(x)


def score_values(values, t: str):
    """Both sides as the codebook's type: (keys compared, texts shown, lenient)."""
    keys = []
    for x in values:
        if is_missing(x):
            keys.append(None)
            continue
        if t in ("number", "integer"):
            if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)):
                keys.append(number_key(x))
            else:
                raw = _empty_none(nfc(str(x)).strip())
                if raw is None:
                    keys.append(None)
                    continue
                if _INT_TEXT_RE.fullmatch(raw):
                    keys.append(str(int(raw)))
                    continue
                num = as_number(raw)
                keys.append(raw if math.isnan(num) else number_key(num))
        elif t == "boolean":
            if isinstance(x, (bool, np.bool_)):
                keys.append("TRUE" if x else "FALSE")
            else:
                raw = _empty_none(nfc(str(x)).strip())
                if raw is None:
                    keys.append(None)
                    continue
                y = parse_yes_no(raw)
                keys.append(raw if y is None else ("TRUE" if y else "FALSE"))
        else:
            if isinstance(x, (bool, np.bool_)):
                s = "true" if x else "false"
            elif isinstance(x, (int, float)):
                s = chr_value(x)
            else:
                s = str(x)
            keys.append(_empty_none(nfc(s)))
    if t in ("number", "integer", "boolean"):
        return keys, keys, keys
    return keys, keys, [lenient(k) for k in keys]


# ------- distances --------------------------------------------------------------------

def lev(a, b) -> int:
    """Levenshtein distance between two strings (or two lists of words)."""
    if _RF is not None:
        return int(_RF.distance(a, b))
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def words(x) -> list[str]:
    if x is None:
        return []
    x = x.replace(" ", " ").strip(" \t\r\n")
    return re.split(f"{_ASCII_SPACE}+", x) if x else []


def word_totals(gold, output, match):
    edits = total = 0
    for g, o, m in zip(gold, output, match):
        wg, wo = words(g), words(o)
        total += max(len(wg), len(wo))
        if not m:
            edits += lev(wg, wo)
    return edits, total


# ------- pairing rows ------------------------------------------------------------------

def align_position(n_g: int, n_o: int):
    k = max(n_g, n_o)
    return [(i if i <= n_g else None, i if i <= n_o else None) for i in range(1, k + 1)]


def align_content(gk, ok, rv, n_g, n_o, open_end=False):
    """Monotone alignment by content: the order of rows is kept, gaps are
    allowed, and the sum of similarities of the pairs made is as large as
    possible. Similarity of two rows = the average over the variables filled
    on at least one side of 1 when equal (lenient) else, for text, 1 - the
    normalised edit distance. Pairs below 0.5 are not made."""
    if not n_g or not n_o:
        if open_end:
            n_o = 0
        return [(i, None) for i in range(1, n_g + 1)] + [(None, j) for j in range(1, n_o + 1)]
    num = np.zeros((n_g, n_o))
    den = np.zeros((n_g, n_o))
    for v in rv:
        a = [lenient_key(x) for x in gk[v.variable][2]]
        b = [lenient_key(x) for x in ok[v.variable][2]]
        fa = np.array([x is not None for x in a])
        fb = np.array([x is not None for x in b])
        filled = fa[:, None] | fb[None, :]
        if v.type == "text":
            ta = ["" if x is None else x for x in a]
            tb = ["" if x is None else x for x in b]
            s = np.ones((n_g, n_o))
            for i in range(n_g):
                for j in range(n_o):
                    L = max(len(ta[i]), len(tb[j]))
                    if L > 0:
                        s[i, j] = 1 - lev(ta[i], tb[j]) / L
        else:
            s = np.array([[1.0 if (x is not None and y is not None and x == y) else 0.0 for y in b] for x in a])
        num += s * filled
        den += filled
    sim = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
    if open_end:
        return dp_align(sim, sim >= 0.5, gap=0.5, open_end=True)
    return dp_align(sim, sim >= 0.5)


def dp_align(sim, allowed, gap: float = 0.0, open_end: bool = False):
    """Best monotone alignment of a similarity matrix (rows gold, columns
    output), pairing only where allowed; each unpaired row costs ``gap``. With
    open_end the alignment may stop at any output row (the best total wins; on
    a tie, the later row). Traceback prefers a pair, then an extra output row,
    then a missing gold row."""
    n, m = sim.shape
    w = np.where(allowed, sim, -np.inf)
    steps = gap * np.arange(m + 1)
    D = np.zeros((n + 1, m + 1))
    D[0, :] = -steps
    D[:, 0] = -gap * np.arange(n + 1)
    for i in range(n):
        a = np.empty(m + 1)
        a[0] = D[i, 0] - gap
        a[1:] = np.maximum(D[i, 1:] - gap, D[i, :-1] + w[i, :])
        D[i + 1, :] = np.maximum.accumulate(a + steps) - steps
    tol = 1e-9
    i, j = n, m
    if open_end:
        last = D[n, :]
        j = int(np.max(np.nonzero(last > np.max(last) - tol)[0]))
    out = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and allowed[i - 1, j - 1] and abs(D[i, j] - (D[i - 1, j - 1] + w[i - 1, j - 1])) < tol:
            out.append((i, j))
            i, j = i - 1, j - 1
        elif j > 0 and abs(D[i, j] - (D[i, j - 1] - gap)) < tol:
            out.append((None, j))
            j -= 1
        else:
            out.append((i, None))
            i -= 1
    return out[::-1]


# ------- scoring units ------------------------------------------------------------------

def _col(rows: dict, v: str, n: int):
    return rows.get(v) if rows is not None and v in rows else [None] * n


def score_page(file, g, o, vars_, align, rows_limit):
    failed = o.get("status") not in ("ok", "check")
    pv = [v for v in vars_ if v.level == "page"]
    rv = [v for v in vars_ if v.level == "row"]
    n_g = g["n"]
    n_o = 0 if failed else o["n"]
    n_o_all = n_o
    partial = rows_limit is not None and n_g >= rows_limit
    if partial:
        n_g = int(rows_limit)
    cells = []
    for v in pv:
        gv = score_values([g["page"].get(v.variable)], v.type)
        ov = score_values([None if failed else o["page"].get(v.variable)], v.type)
        cells.append(_cell(file, "page", 0, None, None, v, gv, ov, 0, 0, False, failed, failed))
    method = None
    missing_rows = extra_rows = None
    if rv:
        method = "none" if failed else (("position" if not partial and n_g == n_o else "content") if align == "auto" else align)
        if partial and method != "content":
            n_o = min(n_o, n_g)
        gk = {v.variable: score_values(_col(g["rows"], v.variable, g["n"])[:n_g], v.type) for v in rv}
        okk = {v.variable: score_values(([] if failed else _col(o["rows"], v.variable, o["n"]))[:n_o], v.type) for v in rv}
        pairs = align_content(gk, okk, rv, n_g, n_o, open_end=partial) if method == "content" else align_position(n_g, n_o)
        if partial and method == "content":
            n_o = max([0] + [oi for _, oi in pairs if oi is not None])
        missing_rows = sum(1 for gi, oi in pairs if gi is not None and oi is None)
        extra_rows = sum(1 for gi, oi in pairs if gi is None and oi is not None)
        for k, (gi, oi) in enumerate(pairs, 1):
            for v in rv:
                cells.append(_cell(file, "row", k, gi, oi, v, gk[v.variable], okk[v.variable],
                                   None if gi is None else gi - 1, None if oi is None else oi - 1,
                                   gi is None, oi is None, failed))
    else:
        n_g = n_o = None
    page = {"file": file, "status": "failed" if failed else o.get("status"), "alignment": method,
            "gold_rows": n_g, "output_rows": n_o, "missing_rows": missing_rows, "extra_rows": extra_rows, "differences": None}
    return page, cells, partial and bool(rv), n_o_all


def _cell(file, level, pair, row_gold, row_output, v, gv, ov, gi, oi, g_missing, o_missing, failed):
    def pick(vals, i, miss):
        return None if miss or i is None or i >= len(vals[0]) else (vals[0][i], vals[1][i], vals[2][i])

    gp = pick(gv, gi, g_missing) or (None, None, None)
    op = pick(ov, oi, o_missing) or (None, None, None)
    return {"file": file, "level": level, "pair": pair, "row_gold": row_gold, "row_output": row_output,
            "variable": v.variable, "type": v.type, "gold": gp[1], "output": op[1], "gold_key": gp[0], "output_key": op[0],
            "gold_lenient": gp[2], "output_lenient": op[2], "g_missing": bool(g_missing), "o_missing": bool(o_missing),
            "failed": bool(failed)}


def _strip_doubt(x):
    return None if x is None else re.sub(r"\s*\?+$", "", x)


def measure_cells(cells: list) -> list:
    for c in cells:
        present = not c["g_missing"] and not c["o_missing"]
        c["match"] = present and c["gold_key"] == c["output_key"]
        gl, ol = lenient_key(c["gold_lenient"]), lenient_key(c["output_lenient"])
        c["lenient_match"] = present and gl == ol
        is_text = c["type"] == "text"
        gt, ot = c["gold"] or "", c["output"] or ""
        c["length"] = max(len(gt), len(ot)) if is_text else None
        c["edits"] = (0 if c["match"] else lev(gt, ot)) if is_text else None
        gl, ol = gl or "", ol or ""
        c["lenient_length"] = max(len(gl), len(ol)) if is_text else None
        c["lenient_edits"] = (0 if c["lenient_match"] else lev(gl, ol)) if is_text else None
        # a wrong value that would be right without a final "?"
        c["doubt_only"] = (not c["match"] and present and c["type"] in ("text", "category") and c["gold"] is not None
                           and c["output"] is not None and _strip_doubt(c["gold"]) == _strip_doubt(c["output"]))
    return cells


def score_units(gold_units, out_units, cb, align="auto", normalize=False, rows=None, variables=None, labels=None) -> Score:
    labels = labels or BENCH_LABELS
    vars_ = [v for v in cb.vars if variables is None or v.variable in variables]
    free_text = len(vars_) == 1 and vars_[0].variable == "text" and vars_[0].level == "page" and vars_[0].type == "text"
    separate = {"text"} if (not free_text and cb.free_text and "text" in [v.variable for v in vars_ if v.level == "page"]) else set()
    gold_files = list(gold_units)
    gold_files = [gold_files[i] for i in natural_order(gold_files)]
    out_keys = {page_key(k): k for k in out_units}
    if not any(page_key(f) in out_keys for f in gold_files):
        raise InputError(f"No page of {labels['ref']} is in the results.",
                         f"Pages in {labels['ref']}: {', '.join(gold_files[:4])}{' ...' if len(gold_files) > 4 else ''}",
                         f"Pages in the results: {', '.join(list(out_units)[:4])}{' ...' if len(out_units) > 4 else ''}",
                         "Extract the benchmark pages first: transcribe(pages, instructions, codebook, only=gold).")
    page_rows_, cells, notes = [], [], []
    for f in gold_files:
        k = out_keys.get(page_key(f))
        if k is None:
            page_rows_.append({"file": f, "status": "not run", "alignment": None, "gold_rows": None, "output_rows": None,
                               "missing_rows": None, "extra_rows": None, "differences": None})
            continue
        page, pc, partial, out_all = score_page(f, gold_units[f], out_units[k], vars_, align, rows)
        page_rows_.append(page)
        cells += pc
        p = page
        if p["alignment"] == "content" and ((p["missing_rows"] or 0) + (p["extra_rows"] or 0) > 0 or p["gold_rows"] != p["output_rows"]):
            tally = f"{p['missing_rows']} missing from {labels['out']}, {p['extra_rows']} not in {labels['ref']})"
            if partial:
                notes.append(f"{f}: {labels['ref']} has the first {plural(p['gold_rows'], 'row')} of the page, paired by content with "
                             f"the first {fmt_int(p['output_rows'])} of the {fmt_int(out_all)} in {labels['out']} ({tally}.")
            else:
                notes.append(f"{f}: {labels['ref']} has {plural(p['gold_rows'], 'row')}, {labels['out']} {fmt_int(p['output_rows'])}; "
                             f"rows paired by content ({tally}.")
    for c in cells:
        c["separate"] = c["variable"] in separate
    measure_cells(cells)
    diffs = {}
    for c in cells:
        if not c["match"]:
            diffs[c["file"]] = diffs.get(c["file"], 0) + 1
    for p in page_rows_:
        if p["status"] != "not run":
            p["differences"] = diffs.get(p["file"], 0)
    by_page = pd.DataFrame(page_rows_, columns=["file", "status", "alignment", "gold_rows", "output_rows", "missing_rows", "extra_rows", "differences"])
    for c in ("gold_rows", "output_rows", "missing_rows", "extra_rows", "differences"):
        by_page[c] = pd.array([None if is_missing(v) else int(v) for v in by_page[c]], dtype="Int64")
    not_run = [p["file"] for p in page_rows_ if p["status"] == "not run"]
    failed = [p["file"] for p in page_rows_ if p["status"] == "failed"]
    if failed:
        notes.insert(0, f"{plural(len(failed), 'page')} failed in {labels['out']} and {'counts' if len(failed) == 1 else 'count'} as "
                        f"entirely wrong: {_list(failed)}. Run the extraction again (with redo=\"failed\" if need be) to retry "
                        f"{'it' if len(failed) == 1 else 'them'}.")
    if not_run:
        notes.insert(0, f"{plural(len(not_run), 'page')} of {labels['ref']} {'is' if len(not_run) == 1 else 'are'} not in the results yet "
                        f"and {'was' if len(not_run) == 1 else 'were'} left out: {_list(not_run)}. Extract "
                        f"{'it' if len(not_run) == 1 else 'them'} with transcribe(pages, instructions, codebook, only=gold).")
    if separate:
        notes.append("The page's other text (variable text) is scored on its own, as characters and words right, not as a value.")
    mode = "text" if free_text else "table"
    cdf = pd.DataFrame(cells) if cells else pd.DataFrame(columns=["file", "level", "pair", "row_gold", "row_output", "variable", "type",
                                                                   "gold", "output", "gold_key", "output_key", "gold_lenient",
                                                                   "output_lenient", "g_missing", "o_missing", "failed", "separate",
                                                                   "match", "lenient_match", "length", "edits", "lenient_length",
                                                                   "lenient_edits", "doubt_only"])
    run = [p for p in page_rows_ if p["status"] != "not run"]
    about = {"mode": mode, "separate": sorted(separate), "normalize": normalize, "align": align, "rows_limit": rows,
             "labels": labels, "n_first": int(bool(not_run)) + int(bool(failed)), "n_pages": len(run),
             "gold_rows": sum((p["gold_rows"] or 0) for p in run), "n_variables": len(vars_),
             "row_variables": sum(1 for v in vars_ if v.level == "row"),
             "page_variables": sum(1 for v in vars_ if v.level == "page" and v.variable not in separate), "variables": vars_}
    return Score(summary=score_summary(cells, mode, normalize), by_variable=score_by_variable(cells, vars_, normalize),
                 by_page=by_page, differences=score_differences(cells, normalize, labels), not_run=not_run, notes=notes,
                 cells=cdf, about=about)


# ------- the tables ------------------------------------------------------------------------

def _measure(measure, matched, compared):
    return {"measure": measure, "value": (matched / compared) if compared > 0 else None,
            "matched": float(matched), "compared": float(compared)}


def score_summary(cells, mode, normalize) -> pd.DataFrame:
    out = []

    def chars(sub, len_=False):
        L = sum(c["lenient_length" if len_ else "length"] or 0 for c in sub if c["type"] == "text")
        E = sum(c["lenient_edits" if len_ else "edits"] or 0 for c in sub if c["type"] == "text")
        return L - E, L

    def wordsum(sub, len_=False):
        sub = [c for c in sub if c["type"] == "text"]
        if len_:
            e, t = word_totals([c["gold_lenient"] or "" for c in sub], [c["output_lenient"] or "" for c in sub], [c["lenient_match"] for c in sub])
        else:
            e, t = word_totals([c["gold"] or "" for c in sub], [c["output"] or "" for c in sub], [c["match"] for c in sub])
        return t - e, t

    if mode == "text":
        out.append(_measure("characters", *chars(cells)))
        out.append(_measure("words", *wordsum(cells)))
        out.append(_measure("pages", sum(c["match"] for c in cells), len(cells)))
        if normalize:
            out.append(_measure("characters (lenient)", *chars(cells, True)))
            out.append(_measure("words (lenient)", *wordsum(cells, True)))
        return pd.DataFrame(out, columns=["measure", "value", "matched", "compared"])
    main = [c for c in cells if not c["separate"]]
    rc = [c for c in main if c["level"] == "row"]

    def row_counts(key):
        groups = {}
        for c in rc:
            groups.setdefault((c["file"], c["pair"]), []).append(c[key])
        return sum(1 for v in groups.values() if all(v)), len(groups)

    out.append(_measure("rows", *row_counts("match")))
    out.append(_measure("values", sum(c["match"] for c in main), len(main)))
    out.append(_measure("characters", *chars(main)))
    pc = [c for c in main if c["level"] == "page"]
    if pc:
        groups = {}
        for c in pc:
            groups.setdefault(c["file"], []).append(c["match"])
        out.append(_measure("page values", sum(1 for v in groups.values() if all(v)), len(groups)))
    out.append(_measure("doubt marks", sum(c["doubt_only"] for c in main), sum(not c["match"] for c in main)))
    sep = [c for c in cells if c["separate"]]
    if sep:
        out.append(_measure("other text: characters", *chars(sep)))
        out.append(_measure("other text: words", *wordsum(sep)))
    if normalize:
        out.append(_measure("rows (lenient)", *row_counts("lenient_match")))
        out.append(_measure("values (lenient)", sum(c["lenient_match"] for c in main), len(main)))
        out.append(_measure("characters (lenient)", *chars(main, True)))
        if sep:
            out.append(_measure("other text: characters (lenient)", *chars(sep, True)))
            out.append(_measure("other text: words (lenient)", *wordsum(sep, True)))
    return pd.DataFrame(out, columns=["measure", "value", "matched", "compared"])


def score_by_variable(cells, vars_, normalize) -> pd.DataFrame:
    rows = []
    for v in vars_:
        sub = [c for c in cells if c["variable"] == v.variable]
        n = len(sub)
        d = sum(not c["match"] for c in sub)
        is_text = v.type == "text"
        L = sum(c["length"] or 0 for c in sub)
        E = sum(c["edits"] or 0 for c in sub)
        row = {"variable": v.variable, "type": v.type, "level": v.level, "compared": n, "differences": d,
               "variable_match": (1 - d / n) if n else None,
               "character_match": (1 - E / L) if is_text and L > 0 else None, "edit_distance": float(E) if is_text else None}
        if normalize:
            ld = sum(not c["lenient_match"] for c in sub)
            LL = sum(c["lenient_length"] or 0 for c in sub)
            LE = sum(c["lenient_edits"] or 0 for c in sub)
            row.update(lenient_differences=ld, lenient_variable_match=(1 - ld / n) if n else None,
                       lenient_character_match=(1 - LE / LL) if is_text and LL > 0 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def score_differences(cells, normalize, labels) -> pd.DataFrame:
    rows = []
    for c in cells:
        if c["match"]:
            continue
        note = None
        if normalize and c["lenient_match"]:
            note = "only case, accents, punctuation or spacing"
        if c["doubt_only"]:
            note = "only a final ?"
        if c["o_missing"] and not c["g_missing"] and c["level"] == "row":
            note = f"row missing from the {labels['out_col']}"
        if c["g_missing"]:
            note = f"row not in the {labels['ref_col']}"
        if c["failed"]:
            note = "page failed"
        rows.append({"file": c["file"], "row_gold": c["row_gold"], "row_output": c["row_output"], "variable": c["variable"],
                     "gold": c["gold"], "output": c["output"], "edits": c["edits"], "note": note})
    df = pd.DataFrame(rows, columns=["file", "row_gold", "row_output", "variable", "gold", "output", "edits", "note"])
    for c in ("row_gold", "row_output"):
        df[c] = pd.array([None if is_missing(v) else int(v) for v in df[c]], dtype="Int64")
    return df


# ------- printing -----------------------------------------------------------------------------

def _list(files, n=5) -> str:
    files = [str(f) for f in files]
    return ", ".join(files[:n]) + (f" and {len(files) - n} more" if len(files) > n else "")


def pct1(v) -> str:
    """One decimal, but never "100.0%" below 100% (or "0.0%" above 0)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    s = f"{100 * v:.1f}%"
    if v < 1 and s == "100.0%":
        return ">99.9%"
    if v > 0 and s == "0.0%":
        return "<0.1%"
    return s


def doubt_sentence(dm, agree) -> str:
    return (f"{fmt_int(dm['matched'])} of the {fmt_int(dm['compared'])} " +
            ("differences are only a final \"?\"" if agree else "wrong values differ only by a final \"?\""))


def score_subtitle(a) -> str:
    if a["mode"] == "text":
        parts = [f"{plural(a['n_pages'], 'page')} of free text"]
    else:
        parts = [plural(a["n_pages"], "page")]
        if a["row_variables"]:
            parts.append(plural(a["gold_rows"], "row"))
        parts.append(plural(a["n_variables"], "variable"))
    model = None
    if a.get("reference") == "run":
        if a.get("model") and a.get("reference_model"):
            model = f"{a['model']} against {a['reference_model']}"
    elif a.get("model"):
        model = a["model"]
    return ", ".join(parts) + (f"; {model}" if model else "")


def score_title(a) -> str:
    head = "Agreement with the first run" if a.get("reference") == "run" else "Accuracy against your benchmark"
    return f"{head} ({score_subtitle(a)})"


def format_score(s: Score) -> str:
    import textwrap

    a = s.about
    sm = s.summary
    agree = a.get("reference") == "run"
    right = "the same" if agree else "right"

    def get(m):
        r = sm[sm["measure"] == m]
        return None if not len(r) else r.iloc[0].to_dict()

    out = [score_title(a)]

    def block(len_):
        rows = []
        sfx = " (lenient)" if len_ else ""

        def line(label, r, counts=True):
            if r is None:
                return
            cnt = f"{fmt_int(r['matched'])} of {fmt_int(r['compared'])}" if counts else ""
            rows.append((label, cnt, pct1(r["value"])))

        if a["mode"] == "text":
            line(f"Characters {right}", get("characters" + sfx), counts=False)
            line(f"Words {right}", get("words" + sfx), counts=False)
            if not len_:
                line("Pages identical" if agree else "Pages entirely right", get("pages"))
            return rows
        if a["row_variables"]:
            line(f"Rows with every value {right}", get("rows" + sfx))
        line(f"Values {right}", get("values" + sfx))
        ch = get("characters" + sfx)
        if ch is not None and ch["compared"] > 0:
            line(f"Characters {right} (text)", ch, counts=False)
        if not len_ and a["page_variables"] and a["row_variables"]:
            line(f"Pages with every page value {right}", get("page values"))
        if a["separate"]:
            line(f"Other text: characters {right}", get("other text: characters" + sfx), counts=False)
            if not len_:
                line(f"Other text: words {right}", get("other text: words"), counts=False)
        return rows

    def show(rows, indent):
        if not rows:
            return
        w1 = max(len(r[0]) for r in rows)
        w2 = max(len(r[1]) for r in rows)
        for r in rows:
            out.append(f"{indent}{r[0]:<{w1}}  {r[1]:>{w2}}  {r[2]:>6}")

    show(block(False), "  ")
    if a["normalize"]:
        out.append('  Lenient (ignoring capital letters, accents, punctuation, spacing and "?"):')
        show(block(True), "    ")
    dm = get("doubt marks")
    if dm is not None and dm["matched"] > 0:
        out.append("  " + doubt_sentence(dm, agree))
    bv = s.by_variable
    if len(bv) and a["mode"] != "text":
        bv = bv[(bv["differences"] > 0) & (~bv["variable"].isin(a["separate"]))].copy()
        if len(bv):
            order = list(s.by_variable["variable"])
            bv["_o"] = [order.index(v) for v in bv["variable"]]
            bv = bv.sort_values(["variable_match", "differences", "_o"], ascending=[True, False, True]).head(2)
            items = []
            for _, r in bv.iterrows():
                d = int(r["differences"])
                what = (" wrong" if not agree else (" differs" if d == 1 else " differ"))
                items.append(f"{r['variable']} {pct1(r['variable_match'])} ({fmt_int(d)}{what})")
            out.append(f"  {'Most disagreement' if agree else 'Weakest'}: {', '.join(items)}")
    for n in s.notes:
        out.append(textwrap.fill(n, width=78, initial_indent="- ", subsequent_indent="  "))
    if a.get("report"):
        out.append(f"Next: look at s.differences; the report is in {a['report']}")
    else:
        out.append("Next: look at s.differences, or score(gold, res, report=\"accuracy.html\")" if not agree else
                   "Next: look at s.differences, or score(run1, run2, report=\"agreement.html\")")
    return "\n".join(out)

"""output(): the results as a table, or saved as files. Nothing here calls the
model or changes a result; it only re-presents what transcribe() saved, so
it can be run as often as wanted, on a result in memory or on its results
folder in a later session.

Judgment calls (the R package's):

- The records written are the rows given: a filtered or reordered result is
  written as it is. The pages, codebook and about sheets always describe the
  whole run, so a selection of rows still carries its provenance.
- Pages to check (their reply did not match the codebook) stay out unless
  ``include="all"``, and then every row carries a page_status column.
- Nothing is overwritten without ``overwrite=True``. Every file of a call is
  checked before the first is written, so a call writes all of them or none.
- Each file is written to a temporary file first and then moved into place.
- Formats that cannot hold everything say so: Excel cells hold at most 32,767
  characters, Stata names are short and plain ASCII. The JSON file is the
  lossless one: it keeps the model's reply for every page.

Python's own format for the whole result object is a pickle (``.pkl``), where
the R package writes ``.rds``.
"""

from __future__ import annotations

import os
import pickle
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

from ._codebook import array_paths, codebook_table, records_path
from ._docx import cell_text, control_chars, write_docx
from ._errors import InputError
from ._flatten import as_series, convert, guess_type, page_rows, page_unit, counter_cols
from ._images import refuse_library_path
from ._models import MODELS_CHECKED, PRACTICE_MODELS
from ._result import Result, as_result, load_pickle, result_from_folder, result_table
from ._say import attention, done, info, plural, say, warn
from ._text import (chr_value, ext, fold_accents, fmt_int, is_missing, page_key, show_path, stem, to_json, utc_now,
                    user_path, write_bytes_atomic, write_csv, write_text)
from ._version import __version__
from ._xlsx import EXCEL_MAX_CHARS, EXCEL_MAX_ROWS, SheetSpec, auto_widths, write_xlsx

FORMATS = ("xlsx", "csv", "dta", "pkl", "pickle", "json", "txt", "text", "md", "docx")
_HINTS = {"csv": False}


def output(x, file=None, *, table: str = "records", include: str = "ok", overwrite: bool = False, bom: bool = True,
           sep: str = ",", dec: str | None = None, contract: str | None = None):
    """Show or save the results.

    Without ``file``, returns the table as a plain DataFrame. With ``file``
    (one name or a list), saves it in the format of the extension:

    - ``.xlsx``: sheets ``records`` (``text`` for free text), ``pages`` (every
      page with its status, tokens and cost), ``codebook`` and ``about`` (the
      model, settings, instructions, date and cost of the run); a JSON-schema
      result gets a sheet for each further list.
    - ``.csv``: one table, UTF-8, with a byte-order mark for Windows Excel
      (``bom=False`` leaves it out); ``sep=";"`` for Excel in Spanish,
      Catalan, French or German, which also writes decimal commas.
    - ``.dta``: Stata 14; names made valid (the changes are listed),
      descriptions as variable labels, yes/no as 0/1 with value labels.
    - ``.pkl``: the whole result object (read it back with ``output()`` or
      ``transcribe(continue_from=...)``).
    - ``.json``: one entry per page with the model's reply as returned.
    - ``.txt``, ``.md``, ``.docx``: a document with one section per page.

    Parameters
    ----------
    x : Result, results folder or .pkl file
        The result of :func:`transcribe` (or a selection of its rows), its
        results folder, or a pickle written by output().
    file : path or list of paths, optional
        Where to save. Leave it out to get the table back.
    table : str
        "records" (default), "pages" or "codebook", for one-table formats.
    include : str
        "ok" (pages read correctly) or "all" (with the pages to check,
        marked in a page_status column).
    overwrite : bool
        Replace files that already exist. Default False.
    bom, sep, dec
        For CSV: byte-order mark, column separator, decimal mark ("," by
        default with ``sep=";"``).
    contract : str, optional
        For a results folder with results of several instructions/codebooks:
        which to use.
    """
    if isinstance(file, Result) and isinstance(x, (str, os.PathLike)):
        raise InputError("The result comes first, then the file name.", f"Write output(res, {str(x)!r}).")
    if table not in ("records", "pages", "codebook"):
        raise InputError("`table` must be one of 'records', 'pages', 'codebook'.", ("x", f"You gave {table!r}."))
    if include not in ("ok", "all"):
        raise InputError("`include` must be 'ok' or 'all'.", ("x", f"You gave {include!r}."))
    for nm, v in (("overwrite", overwrite), ("bom", bom)):
        if not isinstance(v, bool):
            raise InputError(f"`{nm}` must be True or False.")
    if not isinstance(sep, str) or len(sep) != 1 or sep in ('"', "\n", "\r"):
        raise InputError("`sep` must be a single character such as ',' or ';'.", ("x", f"You gave {sep!r}."))
    if dec is None:
        dec = "," if sep == ";" else "."
    if dec not in (".", ","):
        raise InputError("`dec` must be '.' or ','.", ("x", f"You gave {dec!r}."))
    if dec == sep:
        raise InputError(f"`dec` and `sep` cannot both be {sep!r}: numbers would then look like two columns.",
                         "Use sep=';' with a decimal comma, or sep=',' with a decimal point.")
    x = user_path(x)
    files = None
    if file is not None:
        files = [str(user_path(f)) for f in file] if isinstance(file, (list, tuple)) else [str(user_path(file))]
        if not files or any(not f for f in files):
            raise InputError("`file` must be one or more file names, e.g. 'results.xlsx'.")
        files = list(dict.fromkeys(files))
        check_output_files(files, overwrite)
    res = output_source(x, contract)
    info_ = res.info
    if files is None:
        return output_table(res, table, include)
    if table == "records":
        check_page_column(res, files)
    if table == "records" and include == "ok" and not (info_.pages["status"] == "ok").any():
        say(("!", "The table has no rows: no page has been read correctly yet."),
            ("i", "res.pages shows what happened to each page."))
    written = []
    for f in files:
        write_output(res, f, table, include, bom, sep, dec)
        written.append(Path(f))
    if table == "records" and include == "ok" and any(ext(f) not in ("pkl", "pickle") for f in files):
        n = int((info_.pages["status"] == "check").sum())
        if n:
            info(f"{plural(n, 'page')} to check {'is' if n == 1 else 'are'} not in the file (the reply did not match the codebook); "
                 f"add {'it' if n == 1 else 'them'} with include=\"all\".")
    return written


# ------- arguments -------------------------------------------------------------------

def check_output_files(files, overwrite: bool) -> None:
    for f in files:
        e = ext(f)
        if e == "xls":
            raise InputError("Old .xls files cannot be written; use .xlsx, which every version of Excel opens.",
                             f"For example output(res, {stem(f) + '.xlsx'!r}).")
        if e == "rds":
            raise InputError(".rds is R's own format; in Python the whole result is saved as a pickle.",
                             f"For example output(res, {stem(f) + '.pkl'!r}); to share the table with R, use .csv, .xlsx or .dta.")
        if os.path.isdir(f):
            raise InputError(f"{show_path(f)} is a folder.", f"Give a file name inside it, e.g. '{f}/results.xlsx'.")
        if e not in FORMATS:
            raise InputError(f"{os.path.basename(f)} has the extension {e!r}, which output() cannot write." if e else
                             f"{os.path.basename(f)} has no extension, so output() cannot tell which format to write.",
                             "Use .xlsx (Excel), .csv, .dta (Stata), .docx (Word), .txt, .md, .json or .pkl (Python).")
        # every .json file in a results folder is read as a page's result
        if e == "json" and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(f)), "contract.json")):
            parent = os.path.dirname(os.path.dirname(os.path.abspath(f)))
            raise InputError(f"{show_path(os.path.dirname(os.path.abspath(f)))} is a results folder: a .json file saved there would be "
                             "taken for a page's result.", f"Save it next to the folder instead, e.g. {os.path.join(parent, os.path.basename(f))!r}.")
        refuse_library_path(f, arg="file")
    there = [f for f in files if os.path.exists(f)]
    if there and not overwrite:
        raise InputError(f"{', '.join(show_path(t) for t in there)} already exist{'s' if len(there) == 1 else ''}.",
                         f"Choose another name, or add overwrite=True to replace {'it' if len(there) == 1 else 'them'}.")


def check_page_column(res: Result, files) -> None:
    """Documents and JSON are arranged by page and find a row's page through its
    ``file`` column; a selection without it would give every page "(no rows on
    this page)" under a "Wrote" message, so the column is asked for first."""
    if "file" in res.columns:
        return
    cb = res.info.codebook
    main = records_path(cb)
    extra = cb.kind == "json" and any(p != main for p in array_paths(cb.schema))
    by_page = [f for f in files if ext(f) in ("json", "txt", "text", "md", "docx") or (ext(f) == "xlsx" and extra)]
    if not by_page:
        return
    plain = "A .csv file holds" if extra else "An .xlsx or .csv file holds"
    raise InputError(f"{', '.join(os.path.basename(f) for f in by_page)} {'is' if len(by_page) == 1 else 'are'} arranged by page, and this "
                     "selection has no `file` column to say which page each row comes from.",
                     "Keep `file` in the selection, e.g. res.loc[rows, ['file', 'surname', 'age']].",
                     f"{plain} the selection exactly as it is.")


def output_source(x, contract=None) -> Result:
    if isinstance(x, Result) and x.info is not None:
        if contract is not None:
            info("`contract` is only needed with a results folder; ignored for a result already in Python.")
        return x
    if isinstance(x, (str, os.PathLike)) and os.path.isdir(x):
        return result_from_folder(x, contract=contract)
    if isinstance(x, pd.DataFrame):
        raise InputError("This table is no longer linked to its extraction run (that happens after pd.merge(), pd.concat(), "
                         "pd.DataFrame(res) and some other operations).",
                         "Give output() the result of transcribe() (or a selection of it, res[rows]), or its results folder.",
                         "To save a plain table, use df.to_csv() or df.to_excel().")
    return as_result(x)


# ------- tables -----------------------------------------------------------------------------

def output_records(res: Result, include: str) -> pd.DataFrame:
    tab = pd.DataFrame(res).reset_index(drop=True)
    if include == "ok":
        return tab
    if len(res) != res.info.n_rows:
        raise InputError("include=\"all\" rebuilds the table from the whole run, so it cannot be used on a selection of rows.",
                         "Use output(res, include=\"all\") on the complete result, then select rows in the table it returns.")
    full = result_table(res.info, "all")
    if "page_status" not in full.columns:
        full["page_status"] = pd.Series([None] * len(full), dtype=object)
    extra = [c for c in tab.columns if c not in full.columns]
    if extra:
        raise InputError(f"include=\"all\" rebuilds the table from the run, and column{'s' if len(extra) > 1 else ''} "
                         f"{', '.join(extra)} {'is' if len(extra) == 1 else 'are'} not part of it.",
                         "Save the table without them, or without include=\"all\".")
    cols = [c for c in tab.columns if c != "page_status"]
    if "source_page" in cols:
        pos = cols.index("source_page") + 1
        cols = cols[:pos] + ["page_status"] + cols[pos:]
    else:
        cols = ["page_status"] + cols
    return full[cols].reset_index(drop=True)


def output_table(res: Result, table: str, include: str) -> pd.DataFrame:
    if table == "records":
        return output_records(res, include)
    if table == "pages":
        return res.info.pages.copy()
    return codebook_table(res.info.codebook)


def output_pages(res: Result, include: str, tab: pd.DataFrame) -> list:
    """The page records behind the rows being written, in reading order."""
    keep = ("ok", "check") if include == "all" else ("ok",)
    info_ = res.info
    in_scope = [r for r in info_.records.values() if r.get("status") in keep]
    keys = list(dict.fromkeys(page_key(f) for f in tab["file"]))
    if len(res) == info_.n_rows:
        full = result_table(info_, "all" if include == "all" else "ok")
        if all(page_key(f) in keys for f in full["file"].unique()):
            return in_scope
    return [info_.records[k] for k in keys if k in info_.records]


# ------- writing ---------------------------------------------------------------------------

def write_output(res, f, table, include, bom, sep, dec) -> None:
    e = ext(f)
    folder = os.path.dirname(os.path.abspath(f))
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        info(f"Created the folder {show_path(folder)}.")
    if e == "xlsx":
        output_xlsx(res, f, table, include)
    elif e == "csv":
        output_csv(res, f, table, include, bom, sep, dec)
    elif e == "dta":
        output_dta(res, f, table, include)
    elif e in ("pkl", "pickle"):
        output_pkl(res, f, table, include)
    elif e == "json":
        output_json(res, f, table, include)
    elif e in ("txt", "text", "md"):
        output_text(res, f, table, include, md=e == "md")
    elif e == "docx":
        output_docx(res, f, table, include)


def wrote(f, what: str) -> None:
    done(f"Wrote {show_path(f)} ({what}).")


def _rows(n) -> str:
    return plural(n, "row")


def _free_text_fit(res, table, what) -> None:
    if table == "records" and res.info.codebook.kind == "text":
        info(f"Free text: {what} has one row per page, with the page's text in the column text.")


_FORMULA_RE = re.compile(r"^[=+@-]")
_NEG_NUMBER_RE = re.compile(r"^-[0-9]+([.,][0-9]+)?$")


def formula_like(df: pd.DataFrame) -> list[str]:
    """Text columns with a value Excel would run as a formula when it opens a CSV."""
    out = []
    for c in df.columns:
        if df[c].dtype != object:
            continue
        if any(isinstance(v, str) and _FORMULA_RE.match(v) and not _NEG_NUMBER_RE.match(v) for v in df[c]):
            out.append(str(c))
    return out


def output_csv(res, f, table, include, bom, sep, dec) -> None:
    df = output_table(res, table, include)
    _free_text_fit(res, table, "the CSV file")
    write_csv(df, f, bom=bom, sep=sep, dec=dec)
    wrote(f, _rows(len(df)))
    if dec == ",":
        info("Numbers with decimals are written with a decimal comma (2,5), as Excel in Spanish, Catalan, French or German expects; "
             "dec=\".\" keeps the point.")
    if table == "records":
        info("The pages table (status, tokens and cost of every page) can be written with table=\"pages\".")
    fl = formula_like(df)
    if fl:
        say(("!", f"The column{'s' if len(fl) > 1 else ''} {', '.join(fl)} {'has' if len(fl) == 1 else 'have'} values starting with "
                  "=, +, - or @, which Excel runs as formulas when it opens a CSV file (the file itself keeps them as written)."),
            ("i", "To look at the results in Excel, save an .xlsx file: output(res, \"results.xlsx\")."))
    if not _HINTS["csv"]:
        _HINTS["csv"] = True
        say(("i", "To work in Excel, an .xlsx file is safer: Excel changes some values when it opens a CSV file "
                  "(1-2 becomes a date, 007 becomes 7)."),
            ("i", "If Excel shows everything in one column (Excel in Spanish, Catalan, French, German...), write the file with "
                  "sep=\";\", which also writes decimals with a comma.") if sep == "," else None)


def output_pkl(res, f, table, include) -> None:
    if table != "records" or include != "ok":
        info("A .pkl file holds the whole result object; `table` and `include` do not apply to it.")
    write_bytes_atomic(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL), f)
    wrote(f, _rows(len(res)))
    info(f"output({str(f)!r}) gets the result back, with its pages table and codebook (only open pickles you made yourself).")


def output_json(res, f, table, include) -> None:
    info_ = res.info
    if table == "codebook":
        rows = [{"variable": v.variable, "type": v.type, "level": v.level, "description": v.description,
                 "values": list(v.values)} for v in info_.codebook.vars]
        write_text(to_json(rows, pretty=True) + "\n", f)
        wrote(f, plural(len(rows), "variable"))
        return
    if table == "pages":
        df = info_.pages
        rows = [{c: (None if is_missing(v) else v) for c, v in zip(df.columns, r)} for r in df.itertuples(index=False)]
        write_text(to_json(rows, pretty=True) + "\n", f)
        wrote(f, plural(len(rows), "page"))
        return
    tab = output_records(res, include)
    recs = output_pages(res, include, tab)
    pages = []
    for r in recs:
        obj = {"file": r["file"], "source": r.get("source") or r["file"], "source_page": r.get("source_page"), "status": r.get("status")}
        if r.get("status") == "check":
            obj["message"] = r.get("message")
        obj["data"] = r.get("data") if r.get("data") is not None else {}
        pages.append(obj)
    write_text(to_json(pages, pretty=True) + "\n", f)
    wrote(f, plural(len(pages), "page"))
    if len(res) != info_.n_rows:
        info(f"The JSON file holds whole pages (the model's complete reply for each): the {plural(len(pages), 'page')} your rows come from.")


# ------- Stata -------------------------------------------------------------------------------

STATA_RESERVED = {"_all", "_b", "byte", "_coef", "_cons", "double", "float", "if", "in", "int", "long", "_n", "_N",
                  "_pi", "_pred", "_rc", "_skip", "strL", "using", "with"}


def stata_names(names) -> list[str]:
    """Valid Stata names: letters, digits and _, starting with a letter, at most
    32 characters, not reserved, all different. Accents are folded and other
    characters become _ (ASCII only, so the file opens in every Stata from 14).
    A name already valid keeps it: when a changed name comes out the same as
    one of those, the changed one takes the _2."""
    orig = [str(n) for n in names]
    x = []
    for o in orig:
        s = fold_accents(o)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii") if not s.isascii() else s
        s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
        if not o.endswith("_") and s.endswith("_"):
            s = s.rstrip("_")
        s = s or "v"
        if not re.match(r"^[A-Za-z]", s):
            s = "v" + s
        s = s[:32]
        if s in STATA_RESERVED or re.fullmatch(r"str[0-9]+", s):
            s = s[:31] + "_"
        x.append(s)
    kept = [x[i] == orig[i] for i in range(len(x))]
    seen = set()
    for i in range(len(x)):
        if kept[i]:
            if x[i] in seen:
                kept[i] = False
            seen.add(x[i])
    taken = [x[i] for i in range(len(x)) if kept[i]]
    for i in range(len(x)):
        if kept[i]:
            continue
        if x[i] not in taken:
            taken.append(x[i])
            continue
        later = [x[j] for j in range(i + 1, len(x)) if not kept[j]]
        k = 2
        while True:
            suffix = f"_{k}"
            cand = x[i][: 32 - len(suffix)] + suffix
            if cand not in taken and cand not in later:
                break
            k += 1
        x[i] = cand
        taken.append(cand)
    return x


PROVENANCE_LABELS = {"file": "Page image file", "source": "Source document the page comes from",
                     "source_page": "Page number within the source document (not a folio)",
                     "row": "Row on the page, in reading order",
                     "page_status": "ok, or check (the reply did not match the codebook)"}

PAGES_LABELS = {"status": "ok, check, failed or not run", "problem": "Kind of problem, if any",
                "n_rows": "Rows the model returned for the page", "model": "Model asked",
                "model_version": "Model version that answered", "engine": "sync, batch or flex",
                "finish_reason": "Why the model stopped", "input_tokens": "Input tokens",
                "cached_tokens": "Cached input tokens", "image_tokens": "Image tokens", "thinking_tokens": "Thinking tokens",
                "output_tokens": "Output tokens", "cost_usd": "Estimated cost in US dollars",
                "total_cost_usd": "Estimated cost of every paid attempt of the page, US dollars", "attempts": "Attempts",
                "notes": "Notes worth a look", "message": "What went wrong, in words", "date": "Date of the reply (UTC)"}


def output_dta(res, f, table, include) -> None:
    df = output_table(res, table, include)
    _free_text_fit(res, table, "the Stata file")
    cb = res.info.codebook
    desc = {v.variable: v.description for v in cb.vars}
    known = {**PROVENANCE_LABELS, **desc} if table == "records" else {**PROVENANCE_LABELS, **PAGES_LABELS} if table == "pages" else {}
    labels, long_ = {}, 0
    out = pd.DataFrame(index=range(len(df)))
    value_labels = {}
    strl = []
    new = stata_names(df.columns)
    changed = [(o, n) for o, n in zip(df.columns, new) if o != n]
    for (col, name) in zip(df.columns, new):
        v = df[col]
        if str(v.dtype) == "boolean" or (v.dtype == object and all(isinstance(z, bool) or is_missing(z) for z in v) and v.notna().any()):
            vals = [None if is_missing(z) else int(bool(z)) for z in v]
            if any(z is None for z in vals):
                out[name] = pd.Series([float("nan") if z is None else float(z) for z in vals], dtype="float64")
            else:
                out[name] = pd.Series(vals, dtype="int8")
            value_labels[name] = {0: "no", 1: "yes"}
        elif str(v.dtype) in ("Int64", "Int32", "Int8"):
            if v.isna().any():
                out[name] = v.astype("float64")
            else:
                out[name] = v.astype("int64")
        elif v.dtype == object or str(v.dtype).startswith("str"):
            vals = ["" if is_missing(z) else str(z) for z in v]
            out[name] = pd.Series(vals, dtype=object)
            if any(len(z.encode("utf-8")) > 2045 for z in vals):
                strl.append(name)
        else:
            out[name] = v.reset_index(drop=True)
        lab = known.get(col)
        if lab:
            if len(lab) > 80:
                lab = lab[:77] + "..."
                long_ += 1
            labels[name] = lab
    if changed:
        say(("i", f"Stata needs short plain names; {plural(len(changed), 'column')} renamed:"),
            *[(" ", f"{o} -> {n}") for o, n in changed])
    sources = list(dict.fromkeys(res.info.pages["source"]))
    label = "amanuensis: " + ", ".join(map(str, sources[:3])) + (" ..." if len(sources) > 3 else "")
    label = label[:77] + "..." if len(label) > 80 else label
    import io
    buf = io.BytesIO()
    try:
        out.to_stata(buf, write_index=False, version=118, data_label=label, variable_labels=labels,
                     convert_strl=strl or None, value_labels=value_labels or None)
    except Exception as e:
        raise InputError(f"Could not write {f}: {e}") from None
    write_bytes_atomic(buf.getvalue(), f)
    wrote(f, _rows(len(df)))
    if long_:
        info(f"{plural(long_, 'variable label')} shortened to Stata's limit of 80 characters (the full descriptions are in the codebook).")


# ------- Excel ---------------------------------------------------------------------------------

def excel_fit(df: pd.DataFrame):
    """Cells longer than Excel allows are cut, marked with "..." at the end."""
    cut = 0
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        vals = out[c].tolist()
        changed = False
        for i, v in enumerate(vals):
            if isinstance(v, str) and len(v) > EXCEL_MAX_CHARS:
                vals[i] = v[: EXCEL_MAX_CHARS - 3] + "..."
                cut += 1
                changed = True
        if changed:
            out[c] = pd.Series(vals, dtype=object, index=out.index)
    return out, cut


def sheet_name(name: str, taken) -> str:
    base = re.sub(r"[\[\]\\/?*:]", "_", name)[:31]
    out, k = base, 2
    low = {t.lower() for t in taken}
    while out.lower() in low:
        suffix = f"_{k}"
        out = base[: 31 - len(suffix)] + suffix
        k += 1
    return out


def output_xlsx(res, f, table, include) -> None:
    if table != "records":
        info(f"An Excel file holds all the tables (records, pages, codebook, about); table=\"{table}\" only matters for one-table formats.")
    info_ = res.info
    cb = info_.codebook
    recs_tab = output_records(res, include)
    main = "text" if cb.kind == "text" else "records"
    _free_text_fit(res, "records", "the records sheet")
    sheets = {main: recs_tab}
    if cb.kind == "json":
        extra = extra_tables(output_pages(res, include, recs_tab), cb)
        for nm, t in extra.items():
            sheets[sheet_name(nm, list(sheets) + ["pages", "codebook", "about"])] = t
    sheets["pages"] = info_.pages
    sheets["codebook"] = codebook_table(cb)
    sheets["about"] = about_table(res, include, len(recs_tab))
    for nm, t in sheets.items():
        if len(t) > EXCEL_MAX_ROWS:
            raise InputError(f"The sheet {nm!r} has more rows than Excel can hold (1,048,575).", "Write a .csv file instead.")
    cut = 0
    specs = {}
    for nm, t in sheets.items():
        t, c = excel_fit(t)
        cut += c
        specs[nm] = SheetSpec(t, widths=auto_widths(t))
    if cut:
        warn(f"{plural(cut, 'cell')} held more than 32,767 characters, the most an Excel cell can hold, and "
             f"{'was' if cut == 1 else 'were'} cut short in the Excel file.",
             "The full text is in a .txt, .docx or .json file: output(res, \"results.txt\").")
    dropped = write_xlsx(specs, f)
    others = [n for n in specs if n != main]
    wrote(f, f"{_rows(len(recs_tab))} in sheet {main}; sheets {', '.join(others)}")
    if dropped:
        info(f"{plural(dropped, 'invisible control character')} {'was' if dropped == 1 else 'were'} left out of the Excel file "
             "(Excel refuses them); the .json output keeps them.")


def about_table(res, include, n_rows) -> pd.DataFrame:
    """What the about sheet records: enough to cite and to repeat the run."""
    info_ = res.info
    p = info_.pages
    cb = info_.codebook
    ins = info_.instructions
    s = info_.settings
    counts = {st: int((p["status"] == st).sum()) for st in ("ok", "check", "failed", "not run")}

    def groups(col):
        v = [x for x in p[col] if not is_missing(x) and x != ""]
        if not v:
            return "not recorded"
        c = pd.Series(v).value_counts().sort_index()
        return "; ".join(f"{k} ({plural(int(n), 'page')})" for k, n in c.items())

    priced = [r for r in info_.records.values() if r.get("price")]
    price_dates = sorted({r["price"].get("date") for r in priced if r["price"].get("date")})
    if priced:
        pr = priced[0]["price"]
        price_text = (f"input ${chr_value(pr.get('input'))}, cached input ${chr_value(pr.get('cached', pr.get('input')))}, "
                      f"output ${chr_value(pr.get('output'))} per million tokens" + (" (half price: batch or flex)" if pr.get("discount") else ""))
    else:
        price_text = "not recorded"
    dates = sorted(d for d in p["date"] if not is_missing(d))
    practice = (s.get("model") or "") in PRACTICE_MODELS
    text = ins.full or ins.text or ""
    if len(text) > EXCEL_MAX_CHARS:
        text = text[: EXCEL_MAX_CHARS - 3] + "..."
    kinds = {"table": "table plus free text" if cb.free_text else "table", "text": "free text", "json": "JSON schema"}
    total_col = "total_cost_usd" if "total_cost_usd" in p.columns else "cost_usd"
    import platform
    items = {
        "package": f"amanuensis {__version__} (Python)",
        "written": utc_now(),
        "sources": "; ".join(map(str, dict.fromkeys(p["source"]))),
        "pages": f"{len(p)} ({counts['ok']} ok, {counts['check']} to check, {counts['failed']} failed, {counts['not run']} not run)",
        "rows_written": f"{n_rows}" + (f" (a selection of the {info_.n_rows} rows)" if len(res) != info_.n_rows else ""),
        "included": ("pages read correctly and pages to check (see the column page_status)" if include == "all"
                     else "pages read correctly only (status ok)"),
        "model": s.get("model"), "model_versions": groups("model_version"), "engine": groups("engine"),
        "resolution": s.get("resolution"), "thinking": s.get("thinking"), "max_tokens": str(s.get("max_tokens")),
        "temperature": "not sent" if s.get("temperature") is None else chr_value(s.get("temperature")),
        "api_args": to_json(s["api_args"]) if s.get("api_args") else "none",
        "extracted": " to ".join(dict.fromkeys([dates[0], dates[-1]])) if dates else "not recorded",
        "contract": info_.contract, "instructions_source": ins.source, "rules": ins.rules_id or "none",
        "codebook_source": cb.source, "codebook_kind": kinds.get(cb.kind, cb.kind),
        # every paid attempt of a page, not only the reply kept; a result
        # pickled by 0.2.0 has no total_cost_usd column
        "cost_usd": chr_value(round(float(p[total_col].sum(skipna=True)), 6)),
        "cost_other_attempts_usd": chr_value(round(float(p[total_col].sum(skipna=True))
                                                   - float(p["cost_usd"].sum(skipna=True)), 6)),
        "cost_note": ("practice model: nothing was charged; shown at gemini-3.8-flash prices" if practice
                      else "estimate at Google's paid prices, every paid attempt of each page included (retries and "
                           "replaced results: cost_other_attempts_usd); keys without billing are not charged"),
        "prices": price_text, "prices_date": "; ".join(price_dates) if price_dates else "not recorded",
        "results_folder": info_.folder or "kept in memory only",
        "python_version": platform.python_version(),
        "instructions_text": text,
    }
    return pd.DataFrame({"item": list(items), "value": [None if v is None else str(v) for v in items.values()]})


def extra_tables(recs, cb) -> dict:
    """For a JSON schema, every further list of objects as a table of its own:
    file, source, page and row, then its fields; lists of plain values joined
    with "; "."""
    main = records_path(cb)
    out = {}
    for path in [p for p in array_paths(cb.schema) if p != main]:
        rows = []
        for r in recs:
            if r.get("data") is None:
                continue
            prov = {"file": r["file"], "source": r.get("source") or r["file"], "source_page": r.get("source_page")}
            for k, row in enumerate(page_rows(r["data"], path), 1):
                vals = {}
                for c, v in row.items():
                    if isinstance(v, list):
                        vals[c] = "; ".join("" if e is None else (chr_value(e) if isinstance(e, (int, float)) else str(e)) for e in v)
                    else:
                        vals[c] = v
                rows.append({**prov, "row": k, **vals})
        cols = ["file", "source", "source_page", "row"]
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
        counters = counter_cols(path)
        data = {}
        for c in cols:
            vals = [r.get(c) for r in rows]
            t = "integer" if c in ("source_page", "row") or c in counters else "text" if c in ("file", "source") else guess_type(vals)
            data[c] = as_series(vals, t)
        out[".".join(path)] = pd.DataFrame(data, columns=cols)
    return out


# ------- documents (txt, md, docx) -----------------------------------------------------------

def page_sections(tab: pd.DataFrame, recs, cb) -> list[dict]:
    prov = ("file", "source", "source_page", "row", "page_status")
    has_rows = records_path(cb) is not None
    page_vars = [v.variable for v in cb.vars if v.level == "page"]
    page_cols = [c for c in page_vars if c in tab.columns] if has_rows else [c for c in tab.columns if c not in prov]
    row_cols = [c for c in tab.columns if c not in prov and c not in page_cols]
    keys = [page_key(f) for f in tab["file"]]
    out = []
    for r in recs:
        k = page_key(r["file"])
        sub = tab[[kk == k for kk in keys]]
        if len(sub):
            vals = {c: sub[c].iloc[0] for c in page_cols}
        elif r.get("data") is not None and page_cols:
            u = page_unit(r["data"], cb)["page"]
            vals = {c: u[c] for c in page_cols if c in u}
        else:
            vals = {}
        text = vals.pop("text") if "text" in vals else None
        rows = sub[[c for c in ["row"] + row_cols if c in sub.columns]].reset_index(drop=True) if has_rows else None
        out.append({"file": r["file"], "source": r.get("source") or r["file"], "source_page": r.get("source_page"),
                    "status": r.get("status"), "message": r.get("message"), "values": vals,
                    "text": None if is_missing(text) else text, "has_text_key": "text" in page_cols, "rows": rows})
    return out


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 0 if unicodedata.combining(ch) else 1 for ch in s)


def _pad(s: str, w: int, right: bool = False) -> str:
    gap = " " * max(0, w - _width(s))
    return gap + s if right else s + gap


def _flat(x) -> str:
    s = cell_text(x)
    s = re.sub(r"\r?\n", " / ", s)
    return re.sub(r"[\x00-\x1f\x7f]", " ", s)


def _is_num(col) -> bool:
    return str(col.dtype) in ("Int64", "Int32", "int64", "float64", "Float64") and str(col.dtype) != "boolean"


def txt_table(df: pd.DataFrame) -> list[str]:
    if not len(df.columns):
        return []
    cells = [[_flat(v) for v in df[c].tolist()] for c in df.columns]
    heads = [str(c) for c in df.columns]
    widths = [max([_width(h)] + [_width(v) for v in col]) for h, col in zip(heads, cells)]
    num = [_is_num(df[c]) for c in df.columns]

    def line(vals):
        return "  ".join(_pad(v, widths[j], num[j]) for j, v in enumerate(vals)).rstrip()

    out = [line(heads), "  ".join("-" * w for w in widths)]
    for i in range(len(df)):
        out.append(line([cells[j][i] for j in range(len(heads))]))
    return out


def md_escape(x: str) -> str:
    x = x.replace("\\", "\\\\")
    x = re.sub(r"([*_`|<>\[\]])", r"\\\1", x)
    x = re.sub(r"(^|\n)([#>+-])", r"\1\\\2", x)
    return re.sub(r"(^|\n)([0-9]+)\.", r"\1\2\\.", x)


def md_table(df: pd.DataFrame) -> list[str]:
    if not len(df.columns):
        return []
    cells = [[re.sub(r"\r?\n", "<br>", md_escape(cell_text(v))) for v in df[c].tolist()] for c in df.columns]
    num = [_is_num(df[c]) for c in df.columns]

    def row(vals):
        return "| " + " | ".join(vals) + " |"

    out = [row([md_escape(str(c)) for c in df.columns]), "|" + "|".join("---:" if n else "---" for n in num) + "|"]
    for i in range(len(df)):
        out.append(row([cells[j][i] for j in range(len(df.columns))]))
    return out


def value_lines(vals: dict, md: bool = False) -> list[str]:
    if not vals:
        return []
    names = list(vals)
    texts = [cell_text(vals[n]) for n in names]
    if md:
        return [f"- **{md_escape(n)}:** " + re.sub(r"\r?\n", "  \n  ", md_escape(t)) for n, t in zip(names, texts)]
    w = max(_width(n) for n in names) + 1
    return [f"{_pad(n + ':', w)} " + re.sub(r"\r?\n", " / ", t) for n, t in zip(names, texts)]


def document_title(info_) -> tuple[str, str]:
    sources = list(dict.fromkeys(info_.pages["source"]))
    src = ", ".join(map(str, sources[:3])) + (f" and {len(sources) - 3} more" if len(sources) > 3 else "")
    sub = (f"Read by {info_.settings.get('model') or 'the model'} with amanuensis {__version__} (Python); written on "
           f"{date.today().isoformat()}. Values are as the model returned them; empty means illegible or not on the page.")
    return f"{src}: results", sub


def section_heading(s: dict) -> str:
    where = f" ({s['source']} p. {s['source_page']})" if s["source"] != s["file"] else ""
    return f"=== {s['file']}{where} ==="


def _document_fit(info_) -> None:
    if info_.codebook.kind == "text":
        return
    info("A document has one section per page, with the page's values" +
         (" and its rows as a table." if records_path(info_.codebook) is not None else "."))


def output_text(res, f, table, include, md: bool) -> None:
    info_ = res.info
    title, sub = document_title(info_)
    lines = [f"# {md_escape(title)}", "", md_escape(sub), ""] if md else [title, sub, ""]
    if table != "records":
        df = output_table(res, table, include)
        name = "Pages" if table == "pages" else "Codebook"
        lines += ([f"## {name}", ""] + md_table(df) if md else [f"=== {name} ===", ""] + txt_table(df)) + [""]
        write_text(lines, f)
        wrote(f, _rows(len(df)))
        return
    tab = output_records(res, include)
    recs = output_pages(res, include, tab)
    sections = page_sections(tab, recs, info_.codebook)
    _document_fit(info_)
    for s in sections:
        if md:
            lines += [f"## {md_escape(s['file'])}", ""]
            if s["source"] != s["file"]:
                lines += [f"*{md_escape(str(s['source']))}, page {s['source_page']}*", ""]
        else:
            lines += [section_heading(s), ""]
        if s["status"] == "check":
            note = f"To check: {s['message'] or 'the reply did not match the codebook.'}"
            lines += [f"**{md_escape(note)}**" if md else note, ""]
        if s["values"]:
            lines += value_lines(s["values"], md) + [""]
        if s["text"] is not None or info_.codebook.kind == "text" or s["has_text_key"]:
            txt = "(no text)" if s["text"] is None else str(s["text"])
            if md:
                txt = md_escape(txt).replace("\n", "  \n")
            lines += [txt, ""]
        if s["rows"] is not None:
            rows = s["rows"]
            lines += (md_table(rows) if md else txt_table(rows)) if len(rows) else ["(no rows on this page)"]
            lines += [""]
    write_text(lines, f)
    wrote(f, plural(len(sections), "page"))


def output_docx(res, f, table, include) -> None:
    info_ = res.info
    title, sub = document_title(info_)
    blocks = [{"type": "title", "text": title}, {"type": "para", "text": sub}]
    if table != "records":
        df = output_table(res, table, include)
        blocks += [{"type": "heading", "text": "Pages" if table == "pages" else "Codebook"}, {"type": "table", "table": df}]
        write_docx(blocks, f)
        wrote(f, _rows(len(df)))
        return
    tab = output_records(res, include)
    recs = output_pages(res, include, tab)
    sections = page_sections(tab, recs, info_.codebook)
    _document_fit(info_)
    dropped = 0
    for s in sections:
        where = f" ({s['source']} p. {s['source_page']})" if s["source"] != s["file"] else ""
        blocks.append({"type": "heading", "text": f"{s['file']}{where}"})
        if s["status"] == "check":
            blocks.append({"type": "para", "text": f"To check: {s['message'] or 'the reply did not match the codebook.'}"})
        if s["values"]:
            v = [cell_text(z) for z in s["values"].values()]
            blocks.append({"type": "para", "text": "\n".join(f"{n}: {t}" for n, t in zip(s["values"], v))})
            dropped += control_chars(v)
        if s["text"] is not None or info_.codebook.kind == "text" or s["has_text_key"]:
            txt = "(no text)" if s["text"] is None else str(s["text"])
            blocks.append({"type": "para", "text": txt})
            dropped += control_chars([txt])
        if s["rows"] is not None:
            rows = s["rows"]
            blocks.append({"type": "table", "table": rows} if len(rows) else {"type": "para", "text": "(no rows on this page)"})
            for c in rows.columns:
                dropped += control_chars([chr_value(z) for z in rows[c]])
    write_docx(blocks, f)
    wrote(f, plural(len(sections), "page"))
    if dropped:
        info(f"{plural(dropped, 'invisible control character')} {'was' if dropped == 1 else 'were'} left out of the Word file "
             "(Word refuses them); the .json output keeps them.")

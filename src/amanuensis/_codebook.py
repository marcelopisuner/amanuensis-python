"""The codebook: the variables wanted from every page, which becomes the JSON
schema Google enforces on every reply.

Historians write it as a spreadsheet, one row per variable; this turns it
into a draft-07 schema following these rules: every field required and
nullable (null = illegible or absent, never an omitted key), page-level fields
first, then one array of records.

Whether the model returns a table or free text is decided here
(``free_text=True``), not in ``output()``: it changes what the model is
asked, so it is part of the contract hashed into every result. The package's
own wording that goes into the schema (descriptions of the text field, the
type hints appended to descriptions) is the R package's, word for word, so a
codebook hashes the same in both; like the standard rules, it is never edited
in a released version.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from ._errors import InputError, NotFoundError
from ._say import done, info, plural, say, warn
from ._text import (chr_value, delim_frame, ext, fold_accents, is_missing, nfc, norm_header, parse_json, read_delim_rows,
                    read_json, sha12, show_path, to_json, user_path)
from ._validate import resolve_ref, unchecked_keywords

RESERVED_NAMES = ("file", "source", "source_page", "row", "image", "page_status")
PROVENANCE_COLS = ("file", "source", "source_page", "row")

# Accepted spellings of the spreadsheet's headers, after lower-casing and
# folding accents: English, Spanish and Catalan. The exact name always wins,
# wherever the column sits; then these synonyms, first in sheet order.
HEADERS = {
    "variable": ["variable", "variables", "variable name", "name", "field", "field name", "var",
                 "nombre", "nombre de la variable", "nombre de variable", "nombre variable", "campo",
                 "nom", "nom de la variable", "nom de variable", "nom variable", "camp"],
    "type": ["type", "data type", "tipo", "tipus"],
    "description": ["description", "descripcion", "descripcio", "definition",
                    "instructions", "instruction", "what to write"],
    "values": ["values", "allowed values", "categories", "category values", "options", "levels",
               "valores", "categorias", "valors", "categories permeses", "opciones"],
    "level": ["level", "nivel", "nivell"],
}

# Loose synonyms, used only when the codebook has no column of the kind above
# and (for type and level) only when every value in the column is a type or a
# level: a measurement "unit" (pesetas) is not a level, a date "format" is not
# a type, and private "notes" must not replace the descriptions sent to Google.
HEADERS_LOOSE = {"variable": [], "type": ["kind", "format"], "description": ["label", "notes", "nota", "notas"],
                 "values": [], "level": ["scope", "unit"]}

HEADER_WORDS = {"variable": "the variable names", "type": "the types", "description": "the descriptions",
                "values": "the category values", "level": "the levels"}

TYPE_SYNONYMS = {
    "text": ["", "text", "string", "character", "char", "str", "texto", "words", "free text", "verbatim"],
    "number": ["number", "numeric", "num", "double", "float", "decimal", "real", "numero", "nombre", "cifra"],
    "integer": ["integer", "int", "whole number", "whole", "count", "entero", "numero entero", "enter", "sencer"],
    "boolean": ["yes/no", "yes-no", "yes no", "yesno", "boolean", "bool", "logical", "true/false",
                "si/no", "si-no", "binary", "dummy"],
    "category": ["category", "categories", "categorical", "factor", "enum", "choice", "list",
                 "categoria", "categorica"],
    "date": ["date", "fecha", "data", "year", "ano", "any"],
}

LEVEL_SYNONYMS = {
    "row": ["", "record", "records", "row", "rows", "line", "entry", "item", "person",
            "registro", "fila", "linea", "registre", "linia"],
    "page": ["page", "pages", "image", "document", "sheet", "header", "card",
             "pagina", "imagen", "documento", "hoja", "full", "fitxa", "ficha"],
}

TEXT_DESCRIPTION_ALONE = " ".join([
    "Everything written on the image, transcribed exactly as written, in reading order.",
    "Start a new line only for a new paragraph, list item, line of verse or table row; join the lines",
    "of a paragraph with spaces and rejoin words split by a hyphen at the end of a line.",
    "Write a table as one line per row, with the cells separated by ' | '.",
    "Mark doubtful readings with '?' and illegible passages with [...].",
    "Return null if the image has no writing.",
])

TEXT_DESCRIPTION_WITH_RECORDS = " ".join([
    "All other writing on the image that is not part of the records (for example headings, notes,",
    "signatures), transcribed exactly as written, in reading order. Null if there is none.",
])

TYPE_SUFFIX = {
    "number": "Give a number. If unsure, give your best reading; null only if illegible or absent.",
    "integer": "Give a whole number. If unsure, give your best reading; null only if illegible or absent.",
    "boolean": "Answer true or false. Null if it cannot be told.",
    "category": "Choose one of the listed values. If what is written is not in the list, choose 'other' if it is listed; null only if illegible or absent.",
}

JSON_TYPE = {"text": "string", "number": "number", "integer": "integer", "boolean": "boolean", "category": "string"}

OTHER_VALUES = ("other", "others", "otro", "otra", "otros", "altre", "altra", "altres", "autre", "other value")


@dataclass(frozen=True)
class Variable:
    """One variable of the codebook (read-only: make the codebook again to change it)."""

    variable: str
    type: str = "text"
    level: str = "row"
    description: str = ""
    values: list = field(default_factory=list)


# Codebook, Variable and Instructions are frozen: their sha12 is taken when
# they are made, so an attribute changed afterwards (cb.vars[0].description,
# ins.full) would change what is sent, or what the table is built from,
# without changing the contract that records it.
# transcribe() also checks the hashes again (_store.contract), for a change
# made inside the schema dict.
@dataclass(frozen=True)
class Codebook:
    """The variables wanted from every page, and the JSON schema made of them.

    ``kind`` is "table" (a codebook of variables), "text" (free text only)
    or "json" (a JSON schema you wrote); ``sha12`` identifies it in results.
    It is read-only: to change it, change the file and call codebook() again.
    """

    vars: list
    schema: dict
    kind: str = "table"
    free_text: bool = False
    record: str | None = None
    source: str = "data frame"
    sha12: str = ""

    @property
    def variables(self) -> pd.DataFrame:
        """The variables as a table (variable, type, level, description, values)."""
        return pd.DataFrame({
            "variable": [v.variable for v in self.vars],
            "type": [v.type for v in self.vars],
            "level": [v.level for v in self.vars],
            "description": [v.description for v in self.vars],
            "values": [list(v.values) for v in self.vars],
        }, columns=["variable", "type", "level", "description", "values"])

    def names(self, level: str | None = None) -> list[str]:
        return [v.variable for v in self.vars if level is None or v.level == level]

    def types(self) -> dict:
        return {v.variable: v.type for v in self.vars}

    def get(self, name: str):
        for v in self.vars:
            if v.variable == name:
                return v
        return None

    def __repr__(self) -> str:
        if self.kind == "text":
            return "Codebook: free text - the model transcribes each page as text."
        src = self.source if self.source in ("data frame", "json", "free text") else os.path.basename(str(self.source))
        n_page = len(self.names("page"))
        n_rec = len(self.names("row"))
        head = f"Codebook ({src}): {plural(len(self.vars), 'variable')}"
        if n_rec:
            head += f" - {n_page} per page, {n_rec} per row"
        if self.free_text:
            head += ", plus the page's other text"
        lines = [head]
        if self.kind == "json":
            main = main_array_path(self.schema)
            if main:
                lines.append(f"From a JSON schema: one row per entry of '{'.'.join(main[0])}'; other lists get sheets "
                             "of their own in Excel output and are all kept in JSON output.")
            else:
                lines.append("From a JSON schema: one row per page; lists are kept in Excel and JSON output.")

        def short(s, n):
            return s[: n - 3] + "..." if len(s) > n else s

        cols = {
            "variable": [v.variable for v in self.vars],
            "type": [v.type for v in self.vars],
            "level": [v.level for v in self.vars],
            "values": [short("; ".join(map(str, v.values)), 30) for v in self.vars],
            "description": [short(v.description, 50) for v in self.vars],
        }
        if self.vars:
            widths = {k: max([len(k)] + [len(x) for x in vals]) for k, vals in cols.items()}
            lines.append("  ".join(k.ljust(widths[k]) for k in cols).rstrip())
            for i in range(len(self.vars)):
                lines.append("  ".join(cols[k][i].ljust(widths[k]) for k in cols).rstrip())
        return "\n".join(lines)

    __str__ = __repr__


def codebook(file=None, json: str | None = None, free_text: bool = False, record: str | None = None,
             other: bool = True, as_written: bool = False, sheet: str | None = None) -> Codebook:
    """Describe the variables you want: the codebook.

    One row per variable, in a spreadsheet (``.xlsx``, ``.csv``) or a
    DataFrame, with the columns ``variable`` (required), ``type`` (text,
    number, integer, yes/no or category; text by default), ``description``
    (what the model should write: what to copy, an example as printed, what
    must never happen), ``values`` (the categories, separated by ``;`` or on
    separate lines, never by commas) and ``level`` (row, the default: one
    value per row of the page's table; page: one value for the whole page).
    Headers in Spanish and Catalan are understood too.

    Every variable may be empty: the model returns null when something is
    illegible or not on the page. A category gets a catch-all ``other``
    unless one is listed (``other=False`` turns this off); ``as_written=True``
    adds a text twin ``<name>_as_written`` before each number. ``free_text``
    asks for the page's text: alone (no file) or beside the table. A JSON
    schema (a ``.json`` file, or the text in ``json``) is sent to Google as
    it is.

    Parameters
    ----------
    file : path or DataFrame, optional
        A spreadsheet, a JSON schema file, or a DataFrame with the columns above.
    json : str, optional
        A JSON schema as text, instead of a file.
    free_text : bool
        True for a transcription of each page as text.
    record : str, optional
        What one row of the table is, e.g. ``"person listed on the page"``.
    other : bool
        Add a catch-all ``other`` value to categories. Default True.
    as_written : bool
        Also ask for each number exactly as written. Default False.
    sheet : str, optional
        For Excel files: the sheet with the codebook (default: one called
        "codebook", else the first).
    """
    file = user_path(file)
    if file is not None and json is not None:
        raise InputError("Give either `file` or `json`, not both.")
    if record is not None and not isinstance(record, str):
        raise InputError("`record` must be a single piece of text.")
    if isinstance(file, str) and re.match(r"^\s*\{", file):
        json, file = file, None
    if json is not None:
        if not isinstance(json, str):
            raise InputError("`json` must be a JSON schema as text.")
        try:
            schema = parse_json(json)
        except ValueError as e:
            raise InputError("`json` is not valid JSON.", ("x", str(e))) from None
        return codebook_from_schema(schema, source="json", free_text=free_text)
    if file is None:
        if not free_text:
            raise InputError("Give a codebook file (or a DataFrame), or use free_text=True.",
                             "For a table: codebook('codebook.xlsx'); codebook_template() writes an example to start from.",
                             "For a plain transcription of each page: codebook(free_text=True).")
        return codebook_text_only()
    opts = dict(free_text=free_text, record=record, other=other, as_written=as_written)
    if isinstance(file, pd.DataFrame):
        return codebook_from_table(_frame_as_text(file), source="data frame", **opts)
    if not isinstance(file, (str, os.PathLike)):
        raise InputError("`file` must be a codebook file (.xlsx, .csv, .json) or a DataFrame.")
    file = str(file)
    if not os.path.exists(file):
        raise NotFoundError(f"Cannot find the codebook {show_path(file)}.",
                            f"Check the spelling. Relative paths start from your working folder, {os.getcwd()}.")
    e = ext(file)
    if e == "json":
        try:
            schema = read_json(file)
        except ValueError as err:
            raise InputError(f"{os.path.basename(file)} is not valid JSON.", ("x", str(err))) from None
        return codebook_from_schema(schema, source=file, free_text=free_text)
    if e in ("xlsx", "xlsm", "xls"):
        tab = read_codebook_sheet(file, sheet)
    elif e in ("csv", "tsv", "txt"):
        tab = read_codebook_delim(file)
    else:
        raise InputError("The codebook must be an Excel file (.xlsx), a .csv file, or a .json schema.",
                         ("x", f"{os.path.basename(file)} has the extension {e!r}."))
    return codebook_from_table(tab, source=file, **opts)


def _frame_as_text(df: pd.DataFrame) -> pd.DataFrame:
    """A codebook given as a DataFrame, as text read the way a sheet is read:
    numbers without a decimal point when whole (codes typed one per row come
    back from pd.read_excel as 1.0, 2.0), pandas' missing values (pd.NA, NaT)
    as empty, list cells (cb.variables) joined with "; ", and dates marked so
    that they stop like a date cell of a sheet. The same codebook then hashes
    the same as a file or as a DataFrame."""
    import datetime as _dt

    import numpy as np

    cols, dates = {}, []
    for j, c in enumerate(df.columns):
        vals = []
        for i, v in enumerate(df.iloc[:, j].tolist()):
            if isinstance(v, (list, tuple, np.ndarray)):
                parts = [chr_value(x) for x in v]
                parts = [p for p in parts if p is not None and p.strip()]
                vals.append("; ".join(parts) if parts else None)
            elif isinstance(v, (_dt.datetime, _dt.date, _dt.time)) and not is_missing(v):
                dates.append((i, j))
                vals.append(str(v))
            else:
                vals.append(chr_value(v))
        cols[str(c)] = vals
    out = pd.DataFrame(cols, dtype=object)
    out.attrs["header_row"] = 1
    out.attrs["date_cells"] = dates
    return out


def read_codebook_delim(path) -> pd.DataFrame:
    """A CSV codebook, its header row found as in a sheet: a title line above
    the table is allowed, as in .xlsx."""
    rows, sep = read_delim_rows(path)
    if not rows:
        df = pd.DataFrame()
        df.attrs["header_row"] = 1
        return df
    width = max(len(r) for r in rows)
    first = _header_row([[c or None for c in r] + [None] * (width - len(r)) for r in rows])
    df = delim_frame(rows[first - 1:], sep)
    df.attrs["header_row"] = first
    return df


def read_codebook_sheet(path, sheet=None) -> pd.DataFrame:
    from ._xlsx import read_cells, sheet_names

    sheets = sheet_names(path)
    chosen = sheet is None
    if sheet is None:
        pick = [s for s in sheets if s.lower() in ("codebook", "variables")]
        # a hidden sheet (lookup lists) is never the codebook unless it is
        # called "codebook" or "variables"
        visible = sheet_names(path, visible=True)
        sheet = pick[0] if pick else (visible or sheets)[0]
    elif sheet not in sheets:
        raise InputError(f"{os.path.basename(str(path))} has no sheet called {sheet!r}.",
                         f"Its sheets are {', '.join(repr(s) for s in sheets)}.")
    cells = read_cells(path, sheet)
    grid = cells["grid"]
    where = {"sheet": sheet, "sheets": sheets, "chosen": chosen}
    if not grid:
        df = pd.DataFrame()
        df.attrs.update(header_row=1, **where)
        return df
    first = _header_row(grid)
    head = ["" if h is None else str(h) for h in grid[first - 1]]
    body = grid[first:]
    names = []
    for h in head:
        name, k = h, 2
        while name in names and name:
            name = f"{h}.{k}"
            k += 1
        names.append(name)
    df = pd.DataFrame([list(r) for r in body], columns=names, dtype=object) if body else pd.DataFrame(columns=names, dtype=object)
    df.attrs["header_row"] = first
    df.attrs["header"] = head
    df.attrs.update(where)
    # where Excel changed what was typed, by body position (row i, column j)
    df.attrs["date_cells"] = [(i, j) for i, r in enumerate(cells["dates"][first:]) for j, d in enumerate(r) if d]
    df.attrs["formats"] = {(i, j): f for i, r in enumerate(cells["formats"][first:]) for j, f in enumerate(r)
                           if f is not None and str(f) not in ("General", "@", "0")}
    df.attrs["hidden_rows"] = [h - first - 1 for h in cells["hidden"] if h > first]
    return df


def _header_row(grid) -> int:
    """The header row, counted from 1 as in Excel.

    The first row that names known columns of two kinds or more is the header
    (a title row above the table is common; "Variables" alone names one
    kind), whether or not one of them is the variable column: a header whose
    variable column is spelt in a way not known here must stop with "needs a
    column called 'variable'", never let a row below it be taken for the
    header. When no row names two kinds, a row with a column called exactly
    "variable" is the header, else the first non-empty row if it names a
    variable column, else row 1. A data row whose variable happens to be
    called nom, nombre, name or campo is never taken for the header."""
    first_nonempty, exact = None, None
    for i, row in enumerate(grid):
        h = [norm_header(c) for c in row]
        if not any(h):
            continue
        if first_nonempty is None:
            first_nonempty = i
        kinds = sum(any(x in HEADERS[k] + HEADERS_LOOSE[k] for x in h) for k in HEADERS)
        if kinds >= 2:
            return i + 1
        if exact is None and "variable" in h:
            exact = i
    if exact is not None:
        return exact + 1
    if first_nonempty is not None and any(norm_header(c) in HEADERS["variable"] for c in grid[first_nonempty]):
        return first_nonempty + 1
    return 1


def match_synonym(x, synonyms: dict):
    key = fold_accents("" if x is None else str(x).strip()).lower()
    for s, words in synonyms.items():
        if key in words:
            return s
    return None


def _codebook_columns(tab: pd.DataFrame) -> dict:
    names = list(tab.columns)
    heads = [norm_header(n) for n in names]
    idx = {k: None for k in HEADERS}
    for key in HEADERS:
        hit = [j for j, h in enumerate(heads) if h == key] + [j for j, h in enumerate(heads) if h in HEADERS[key]]
        if hit:
            idx[key] = hit[0]
    loose_used = {}
    for key in [k for k in HEADERS if idx[k] is None]:
        for j, h in enumerate(heads):
            if h not in HEADERS_LOOSE[key] or j in idx.values():
                continue
            vals = [str(v).strip() for v in tab.iloc[:, j].tolist() if v is not None and str(v).strip()]
            if key == "type":
                fits = bool(vals) and all(match_synonym(v, TYPE_SYNONYMS) for v in vals)
            elif key == "level":
                fits = bool(vals) and all(match_synonym(v, LEVEL_SYNONYMS) for v in vals)
            else:
                fits = bool(vals)
            if fits:
                idx[key] = j
                loose_used[key] = names[j]
                break
    # "Levels" is R's word for a factor's values, but a column of page/row
    # words under it is the level column: read as values, it made every
    # page variable a row variable
    j = idx["values"]
    if j is not None and heads[j] == "levels" and idx["level"] is None:
        vals = [str(v).strip() for v in tab.iloc[:, j].tolist() if v is not None and str(v).strip()]
        if vals and all(match_synonym(v, LEVEL_SYNONYMS) for v in vals):
            idx["level"], idx["values"] = j, None
            loose_used["level"] = names[j]
    if idx["variable"] is None:
        raise InputError("The codebook needs a column called 'variable' with the variable names.",
                         f"Its columns are: {', '.join(repr(str(n)) for n in names)}.",
                         _sheet_hint(tab),
                         "Expected columns: variable, type, description, values, level. codebook_template() writes an example.")
    for key, used in loose_used.items():
        info(f"Reading {HEADER_WORDS[key]} from column {used!r}: the codebook has no {key!r} column.")
    for key in HEADERS:
        if idx[key] is None:
            continue
        also = [j for j, h in enumerate(heads) if h in HEADERS[key] and j != idx[key]]
        if also:
            also_names = ", ".join(repr(names[j]) for j in also)
            info(f"Ignoring codebook column{'s' if len(also) > 1 else ''} {also_names}: "
                 f"{HEADER_WORDS[key]} are read from column {names[idx[key]]!r}.")
    known = {h for words in HEADERS.values() for h in words}
    unknown = [names[j] for j, h in enumerate(heads) if j not in idx.values() and h not in known and str(names[j]).strip()]
    if unknown:
        info(f"Ignoring codebook column{'s' if len(unknown) > 1 else ''} {', '.join(repr(str(u)) for u in unknown)} "
             "(not one of variable, type, description, values, level).")
    return idx


def split_values(x: str) -> list[str]:
    """Category values, separated by line breaks, ";" or "|" - never by commas
    (a value such as "Barcelona, ciutat" must survive)."""
    if not x:
        return []
    out = []
    for v in re.split(r"\n|;|\|", x):
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out


def _sheet_hint(tab):
    """For a sheet the package chose itself: which sheet it read, and sheet=."""
    a = tab.attrs
    if not a.get("chosen") or len(a.get("sheets") or []) < 2:
        return None
    return (f"This was read from the sheet {a['sheet']!r}; the file's sheets are {', '.join(repr(s) for s in a['sheets'])}. "
            "Choose the codebook's own with sheet=, e.g. codebook(file, sheet=" + repr([s for s in a["sheets"] if s != a["sheet"]][0]) + ").")


def _parse_type(x, row: int, catalan: bool = False) -> str:
    """The type of one row. "nombre" is a number in Catalan and a name in
    Spanish: a number only under a Catalan header (tipus); anywhere else it
    stops, as either reading would be a guess."""
    if fold_accents(str(x or "").strip()).lower() == "nombre" and not catalan:
        raise InputError(f"Row {row} of the codebook has type {x!r}, which is a number in Catalan and a name in Spanish.",
                         "Write 'número' (or 'number') for a number, or 'text' for a name.")
    t = match_synonym(x, TYPE_SYNONYMS)
    if t is None:
        raise InputError(f"Row {row} of the codebook has type {x!r}, which is not one of the known types.",
                         "Use text, number, integer, yes/no or category.")
    return t


def _parse_level(x, row: int) -> str:
    l = match_synonym(x, LEVEL_SYNONYMS)
    if l is None:
        raise InputError(f"Row {row} of the codebook has level {x!r}.",
                         "Use 'row' (one value per row of the table; the default) or 'page' (one value per page).")
    return l


def codebook_from_table(tab: pd.DataFrame, source, free_text=False, record=None, other=True, as_written=False) -> Codebook:
    header_row = tab.attrs.get("header_row", 1)
    if not len(tab):
        raise InputError("The codebook has no rows. Add one row per variable.", _sheet_hint(tab))
    idx = _codebook_columns(tab)
    used = {j: key for key, j in idx.items() if j is not None}
    if record is not None:
        record = nfc(record)

    # Excel stores "5-9" or "1/2" typed into a General cell as a date, and the
    # codebook would send its serial number (46151) as an allowed value
    dated = [(i, j) for i, j in tab.attrs.get("date_cells") or [] if j in used]
    if dated:
        raise InputError("Excel (or pandas) turned some cells of the codebook into dates, so they cannot be read as typed:",
                         *[("x", f"row {i + 1 + header_row}, column {str(tab.columns[j])!r}: it reads {tab.iloc[i, j]}")
                           for i, j in dated[:5]],
                         *([("x", f"... and {len(dated) - 5} more")] if len(dated) > 5 else []),
                         ("i", "Format the column as Text (Home > Number > Text) and type those cells again, or type an "
                               "apostrophe first ('5-9); then save and run the line again."))
    shown = [(i, j, f) for (i, j), f in sorted((tab.attrs.get("formats") or {}).items()) if j in used]
    if shown:
        i, j, f = shown[0]
        v = tab.iloc[i, j]
        padded = f" (Excel shows {str(v).zfill(len(str(f)))})" if re.fullmatch(r"0+", str(f)) and str(v).isdigit() else ""
        warn(f"{plural(len(shown), 'number')} in the codebook {'is' if len(shown) == 1 else 'are'} shown in a number format of "
             f"{'its' if len(shown) == 1 else 'their'} own, e.g. row {i + 1 + header_row}, column {str(tab.columns[j])!r}: "
             f"format {f!r}{padded}. What is sent is the number the cell holds, {v}.",
             "If the zeros or the formatting matter, format the column as Text and type the values again.")

    def get(key):
        if idx[key] is None:
            return [""] * len(tab)
        # NFC, as instructions are: the same codebook typed or synced in NFD
        # hashes the same and lists the same values
        return ["" if v is None else nfc(str(v).strip()) for v in tab.iloc[:, idx[key]].tolist()]

    var, desc_all, type_all, values_all, level_all = get("variable"), get("description"), get("type"), get("values"), get("level")
    n = len(var)
    hidden = [i for i in tab.attrs.get("hidden_rows") or [] if 0 <= i < n and var[i]]
    if hidden:
        warn(f"{plural(len(hidden), 'row')} of the codebook {'is' if len(hidden) == 1 else 'are'} hidden in Excel and "
             f"{'is' if len(hidden) == 1 else 'are'} read like any other: "
             + ", ".join(f"row {i + 1 + header_row} ({var[i]!r})" for i in hidden[:5]) + ".",
             "Delete the rows of variables that should not be sent.")
    # A row with nothing but values, right below a variable (or below another
    # such row), holds more values of that variable: one category per row, or
    # the variable's name merged down over its values. Anywhere else it is an
    # error: dropping it would send Google a shorter list, and every person
    # with a dropped value would come back as "other" for the whole paid run.
    no_name = [not var[i] and not desc_all[i] and not type_all[i] and not level_all[i] for i in range(n)]
    only_values = [no_name[i] and bool(values_all[i]) for i in range(n)]
    owner = [None] * n
    current = None
    for i in range(n):
        if var[i]:
            current = i
        elif only_values[i]:
            owner[i] = current
        else:
            current = None
    is_category = [not type_all[i] or match_synonym(type_all[i], TYPE_SYNONYMS) == "category" for i in range(n)]
    orphan = [only_values[i] and (owner[i] is None or not is_category[owner[i]]) for i in range(n)]
    if any(orphan):
        rows = ", ".join(str(i + 1 + header_row) for i in range(n) if orphan[i])
        k = sum(orphan)
        raise InputError(f"Row{'s' if k > 1 else ''} {rows} of the codebook ha{'ve' if k > 1 else 's'} category values but no variable.",
                         "Write the values in the 'values' cell of their variable, separated by ';', "
                         "or on the rows directly below a category variable.")
    if any(only_values):
        took = []
        for i in range(n):
            if only_values[i]:
                values_all[owner[i]] = values_all[owner[i]] + "\n" + values_all[i]
                if var[owner[i]] not in took:
                    took.append(var[owner[i]])
        info(f"Read the values on the rows below {', '.join(repr(t) for t in took)} as more category values of "
             f"{'that variable' if len(took) == 1 else 'those variables'}.")
    keep = [i for i in range(n) if not no_name[i]]
    sheet_row = [i + 1 + header_row for i in keep]
    names = [var[i] for i in keep]
    if any(not v for v in names):
        bad = [str(sheet_row[k]) for k, v in enumerate(names) if not v]
        raise InputError(f"Row{'s' if len(bad) > 1 else ''} {', '.join(bad)} of the codebook ha{'ve' if len(bad) > 1 else 's'} "
                         "a description, type or level but no variable name.")
    clean = [re.sub(r"\s+", "_", v) for v in names]
    renamed = [(a, b) for a, b in zip(names, clean) if a != b]
    if renamed:
        info("Variable names cannot contain spaces; renamed " + ", ".join(f"{a!r} to {b!r}" for a, b in renamed) + ".")
    names = clean
    dup = sorted({v for v in names if names.count(v) > 1}, key=names.index)
    if dup:
        raise InputError(f"The codebook lists {', '.join(repr(d) for d in dup)} more than once. Each variable needs its own name.")
    bad = [v for v in names if v in RESERVED_NAMES]
    if bad:
        raise InputError(f"{', '.join(repr(b) for b in bad)} cannot be used as variable name{'s' if len(bad) > 1 else ''}: "
                         f"amanuensis uses {'it' if len(bad) == 1 else 'them'} to record where each row comes from.",
                         f"Rename it, for example 'printed_{bad[0]}'.")
    type_raw = [type_all[i] for i in keep]
    catalan = idx["type"] is not None and norm_header(tab.columns[idx["type"]]) == "tipus"
    types = [_parse_type(t, sheet_row[k], catalan) for k, t in enumerate(type_raw)]
    values = [split_values(values_all[i]) for i in keep]
    # values given but no type: the user clearly means a category
    for k in range(len(types)):
        if types[k] == "text" and not type_raw[k] and values[k]:
            types[k] = "category"
    dates = [names[k] for k in range(len(types)) if types[k] == "date"]
    if dates:
        info(f"Dates are kept as text, exactly as written ({', '.join(repr(d) for d in dates)}). Convert them afterwards if you need to.")
        types = ["text" if t == "date" else t for t in types]
    need = [names[k] for k in range(len(types)) if types[k] == "category" and not values[k]]
    if need:
        raise InputError(f"Category variable{'s' if len(need) > 1 else ''} {', '.join(repr(x) for x in need)} need{'' if len(need) > 1 else 's'} a list of allowed values.",
                         "Write them in the 'values' column separated by ';', e.g. 'single; married; widowed; other'.")
    extra = [names[k] for k in range(len(types)) if types[k] != "category" and values[k]]
    if extra:
        info(f"Ignoring the values given for {', '.join(repr(x) for x in extra)}, which {'is' if len(extra) == 1 else 'are'} not of type category.")
        values = [v if types[k] == "category" else [] for k, v in enumerate(values)]
    # A category with one listed value is almost never meant: "H/M", "S/N" or
    # "soltero, casado, viudo" written in one piece (values are never split on
    # commas or slashes). Google enforces the list, so every row would come
    # back as that one value or "other": stop before anything is sent. A
    # catch-all listed after it does not make it a list: "H/M; otro" and
    # "soltero, casado, viudo; otro" stop too.
    def real(vals):
        return [v for v in vals if fold_accents(v).lower() not in OTHER_VALUES]

    single = [k for k in range(len(types)) if types[k] == "category" and
              (len(values[k]) == 1 or (len(real(values[k])) == 1 and re.search(r"[/,]", real(values[k])[0])))]
    if single:
        items = []
        for k in single:
            one = (real(values[k]) or values[k])[0]
            guess = re.sub(r"\s*[/,]\s*", "; ", one) if re.search(r"[/,]", one) else None
            items.append(("*", f"{names[k]!r} lists only {one!r}" + (f" (did you mean {guess!r}?)" if guess else "") + "."))
        head = ("A category variable has" if len(single) == 1 else "Category variables have") + \
            " only one allowed value, so the model could only ever answer that value or 'other':"
        raise InputError(head, *items,
                         ("i", "Separate the values with ';' or put each on its own line in the cell (Alt+Enter in Excel), "
                               "e.g. 'H; M'. Commas and slashes are kept as part of a value."),
                         ("i", "For a mark that is either there or not, use type 'yes/no' instead."))
    if other:
        add = [k for k in range(len(types)) if types[k] == "category" and
               not any(fold_accents(v).lower() in OTHER_VALUES for v in values[k])]
        for k in add:
            values[k] = values[k] + ["other"]
        if add:
            info(f"Added the value 'other' to {', '.join(repr(names[k]) for k in add)}, so the model is never forced to pick "
                 "a wrong category (other=False to turn this off).")
    level_raw = [level_all[i] for i in keep]
    levels = [_parse_level(l, sheet_row[k]) for k, l in enumerate(level_raw)]
    # the records of a page are the schema's field "records": a page variable
    # of that name would be replaced by them
    if "row" in levels and any(nm == "records" and lv == "page" for nm, lv in zip(names, levels)):
        raise InputError("'records' cannot be the name of a page variable: amanuensis uses it for the list of rows of each page.",
                         "Rename it, for example 'records_note'.")
    descs = [desc_all[i] for i in keep]
    vars_ = [Variable(names[k], types[k], levels[k], descs[k], values[k]) for k in range(len(names))]
    if as_written and any(v.type in ("number", "integer") for v in vars_):
        vars_ = _add_as_written(vars_)
    missing = sum(1 for d in descs if not d)
    if missing:
        say(("!", f"{missing} of {plural(len(descs), 'variable')} ha{'s' if missing == 1 else 've'} no description."),
            ("i", "Descriptions guide the model: say what to copy, give an example as printed, and what must never happen."))
    if free_text and "text" in [v.variable for v in vars_]:
        raise InputError("The codebook already has a variable called 'text'; with free_text=True that name is used for the page's text.")
    all_vars = vars_ + ([Variable("text", "text", "page", TEXT_DESCRIPTION_WITH_RECORDS, [])] if free_text else [])
    schema = schema_from_vars(vars_, free_text=bool(free_text), record=record)
    return Codebook(vars=all_vars, schema=schema, kind="table", free_text=bool(free_text), record=record,
                    source=str(source), sha12=sha12(to_json(schema)))


def _add_as_written(vars_: list) -> list:
    """For each number/integer variable, a text twin "<name>_as_written" just
    before it (an interpreted value sits next to the raw one, never replaces
    it). A variable of the codebook already called so is that twin: a second
    one would replace its description in the schema and list it twice."""
    have = {v.variable for v in vars_}
    out, kept = [], []
    for v in vars_:
        if v.type in ("number", "integer"):
            twin = f"{v.variable}_as_written"
            if twin in have:
                kept.append(twin)
            else:
                out.append(Variable(twin, "text", v.level,
                                    f"The value of {v.variable} exactly as written on the page, characters and spacing "
                                    "included (e.g. '1 069 544', '(66.0)'). Null if there is none.", []))
        out.append(v)
    if kept:
        info(f"{', '.join(repr(k) for k in kept)} {'is' if len(kept) == 1 else 'are'} already in the codebook, so as_written=True "
             "does not add another.")
    return out


def schema_property(v: Variable) -> dict:
    p = {"type": [JSON_TYPE[v.type], "null"]}
    desc = v.description
    suffix = TYPE_SUFFIX.get(v.type)
    if suffix:
        desc = f"{desc} {suffix}" if desc else suffix
    if desc:
        p["description"] = desc
    if v.type == "category":
        p["enum"] = [str(x) for x in v.values] + [None]
    return p


def schema_from_vars(vars_: list, free_text: bool = False, record: str | None = None) -> dict:
    page = [v for v in vars_ if v.level == "page"]
    rec = [v for v in vars_ if v.level == "row"]
    props = {}
    for v in page:
        props[v.variable] = schema_property(v)
    if rec:
        item_props = {v.variable: schema_property(v) for v in rec}
        what = record or "record (row of the table or entry of the list)"
        props["records"] = {
            "type": "array",
            "description": f"One object for each {what} on this image, in reading order. Return an empty list if there is none.",
            "items": {"type": "object", "properties": item_props, "required": list(item_props),
                      "additionalProperties": False},
        }
    if free_text:
        props["text"] = {"type": ["string", "null"],
                         "description": TEXT_DESCRIPTION_WITH_RECORDS if vars_ else TEXT_DESCRIPTION_ALONE}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def codebook_text_only() -> Codebook:
    schema = schema_from_vars([], free_text=True)
    return Codebook(vars=[Variable("text", "text", "page", TEXT_DESCRIPTION_ALONE, [])], schema=schema, kind="text",
                    free_text=True, record=None, source="free text", sha12=sha12(to_json(schema)))


# ------- JSON schemas ------------------------------------------------------------

def _types_of(p) -> list:
    t = p.get("type") if isinstance(p, dict) else None
    if t is None:
        return []
    t = t if isinstance(t, list) else [t]
    return [x for x in t if x != "null"]


def _is_object_node(p) -> bool:
    ty = _types_of(p)
    return ty == ["object"] or (not ty and isinstance(p.get("properties"), dict))


def _is_object_array(p) -> bool:
    items = p.get("items")
    return _types_of(p) == ["array"] and isinstance(items, dict) and (
        _types_of(items) == ["object"] or isinstance(items.get("properties"), dict))


def deref_schema_tree(node, root=None, seen=()):
    """A copy of the schema with local $ref replaced by what they point to (a
    reference met again inside itself is left as it is, so this ends). The
    schema sent to Google stays as written."""
    if root is None:
        root = node
    if isinstance(node, list):
        return [deref_schema_tree(v, root, seen) for v in node]
    if not isinstance(node, dict) or not node:
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        target = resolve_ref(ref, root)
        if isinstance(target, dict):
            rest = {k: v for k, v in node.items() if k != "$ref"}
            merged = dict(rest)
            for k, v in target.items():
                if k not in rest:
                    merged[k] = v
            return deref_schema_tree(merged, root, tuple(seen) + (ref,))
    return {k: deref_schema_tree(v, root, seen) for k, v in node.items()}


def main_array_path(schema):
    """The first array of objects (depth-first, schema order) as (path, items),
    e.g. (["tables", "rows"], {...}); None when there is none."""
    schema = deref_schema_tree(schema)

    def find(props, path):
        for nm, p in (props or {}).items():
            if not isinstance(p, dict):
                continue
            if _is_object_array(p):
                inner = find(p["items"].get("properties") or {}, path + [nm])
                return inner if inner is not None else (path + [nm], p["items"])
            if _types_of(p) == ["object"] and isinstance(p.get("properties"), dict):
                inner = find(p["properties"], path + [nm])
                if inner is not None:
                    return inner
        return None

    return find(schema.get("properties") or {}, [])


def object_arrays(schema) -> list:
    """Every array of objects in a schema, at any depth, as (path, items)."""
    schema = deref_schema_tree(schema)
    out = []

    def walk(props, path):
        for nm, p in (props or {}).items():
            if not isinstance(p, dict):
                continue
            if _is_object_array(p):
                out.append((path + [nm], p["items"]))
                walk(p["items"].get("properties") or {}, path + [nm])
            elif _is_object_node(p):
                walk(p.get("properties") or {}, path + [nm])

    walk(schema.get("properties") or {}, [])
    return out


def array_paths(schema) -> list:
    """Paths of the innermost lists of objects, nested ones included."""
    schema = deref_schema_tree(schema)
    out = []

    def walk(props, path):
        for nm, p in (props or {}).items():
            if not isinstance(p, dict):
                continue
            if _types_of(p) == ["object"] or (p.get("type") is None and isinstance(p.get("properties"), dict)):
                walk(p.get("properties") or {}, path + [nm])
                continue
            if not _is_object_array(p):
                continue
            before = len(out)
            walk(p["items"].get("properties") or {}, path + [nm])
            if len(out) == before:
                out.append(path + [nm])

    walk(schema.get("properties") or {}, [])
    return out


def vars_from_schema(schema) -> list:
    """The variables of a JSON schema as the flattened table shows them: root
    scalars are page level, the fields of the first array of objects are row
    level, nested objects become parent.child columns."""
    schema = deref_schema_tree(schema)

    def scalar_rows(props, prefix, level):
        out = []
        for nm, p in (props or {}).items():
            if not isinstance(p, dict):
                continue
            ty = _types_of(p)
            full = f"{prefix}.{nm}" if prefix else nm
            if ty == ["object"] or (not ty and isinstance(p.get("properties"), dict)):
                out += scalar_rows(p.get("properties") or {}, full, level)
            elif ty == ["array"]:
                continue
            else:
                if isinstance(p.get("enum"), list):
                    t = "category"
                    vals = [str(v) for v in p["enum"] if v is not None]
                else:
                    t = {"string": "text", "integer": "integer", "number": "number", "boolean": "boolean"}.get(ty[0] if ty else "string", "text")
                    vals = []
                out.append(Variable(full, t, level, str(p.get("description") or ""), vals))
        return out

    rows = scalar_rows(schema.get("properties") or {}, "", "page")
    main = main_array_path(schema)
    if main is not None:
        # a field of each entry with the name of a page field (a newspaper
        # block's kind and the kind of each of its parts) gets its list's path
        # in front: one column each, parts.kind beside kind
        page = {v.variable for v in rows}
        prefix = ".".join(main[0])
        rows += [replace(v, variable=f"{prefix}.{v.variable}") if v.variable in page else v
                 for v in scalar_rows(main[1].get("properties") or {}, "", "row")]
    return rows


def row_columns(cb) -> dict:
    """For a JSON schema: the fields of each entry of the main list whose
    column is not their own name ({"kind": "parts.kind"}); every reader of
    the rows (the table, scoring, benchmarks) renames them the same way."""
    if cb.kind != "json":
        return {}
    main = main_array_path(cb.schema)
    if main is None:
        return {}
    prefix = ".".join(main[0]) + "."
    page = {v.variable for v in cb.vars if v.level == "page"}
    return {v.variable[len(prefix):]: v.variable for v in cb.vars
            if v.level == "row" and v.variable.startswith(prefix) and v.variable[len(prefix):] in page}


def codebook_from_schema(schema, source, free_text=False) -> Codebook:
    if not isinstance(schema, dict):
        raise InputError("The JSON schema must be a JSON object (it starts with '{').")
    ty = schema.get("type")
    first = ty[0] if isinstance(ty, list) and ty else ty
    if first != "object" or "properties" not in schema:
        raise InputError("The top level of the JSON schema must be an object with properties.",
                         'For example {"type": "object", "properties": {...}}.')
    if free_text:
        info("`free_text` is ignored for a JSON schema: add a text field to the schema itself if you want one.")
    vars_ = vars_from_schema(schema)
    names = [v.variable for v in vars_]
    bad = [v for v in names if v in RESERVED_NAMES]
    if bad:
        raise InputError(f"The schema uses {', '.join(repr(b) for b in bad)} as field name{'s' if len(bad) > 1 else ''}; "
                         f"amanuensis needs {'it' if len(bad) == 1 else 'them'} for provenance columns. Rename {'it' if len(bad) == 1 else 'them'}.")
    # one column per name: a page field and a record field of the same name
    # would show the page value on every row and lose the record values
    dup = sorted({v for v in names if names.count(v) > 1}, key=names.index)
    if dup:
        raise InputError(f"The schema has more than one field that becomes the column {', '.join(repr(d) for d in dup)} of the results "
                         "table (for example a nested field and a field whose name has a dot); the table can keep only one.",
                         f"Rename one of them, for example to '{dup[0].replace('.', '_')}'.")
    # the other lists become sheets of their own, which start with file,
    # source, source_page and row: a field with one of those names would be
    # overwritten by the provenance
    main = main_array_path(schema)
    for path, items in object_arrays(schema):
        if main is not None and path == main[0]:
            continue
        props = items.get("properties") or {}
        plain = [nm for nm, p in props.items() if isinstance(p, dict) and not _is_object_node(p)]
        clash = [nm for nm in plain if nm in PROVENANCE_COLS]
        if clash:
            raise InputError(f"The list {'.'.join(path)!r} in the schema has {'a field' if len(clash) == 1 else 'fields'} called "
                             f"{', '.join(repr(c) for c in clash)}; amanuensis needs {'that name' if len(clash) == 1 else 'those names'} "
                             f"for provenance columns. Rename {'it' if len(clash) == 1 else 'them'}.",
                             f"For example '{path[-1]}_{clash[0]}'.")
    unchecked = unchecked_keywords(schema)
    if unchecked:
        info(f"These schema keywords are sent to Google but not checked locally: {', '.join(repr(u) for u in unchecked)}.")
    out = Codebook(vars=vars_, schema=schema, kind="json", free_text=False, record=None, source=str(source),
                   sha12=sha12(to_json(schema)))
    both = row_columns(out)
    if both:
        info("Fields both on the page and in each entry of the list: " +
             ", ".join(f"{k!r} (the entries' one is the column {c!r})" for k, c in both.items()) + ".")
    return out


def as_codebook(x) -> Codebook:
    if x is None:
        info("No codebook given: the model will transcribe each page as text (codebook(free_text=True)).")
        return codebook_text_only()
    if isinstance(x, Codebook):
        return x
    if isinstance(x, (pd.DataFrame, str, os.PathLike)):
        return codebook(x)
    raise InputError("`codebook` must come from codebook(), or be a codebook file or DataFrame.")


def subset(cb: Codebook, keep) -> Codebook:
    """The codebook with only the variables named in ``keep`` (for scoring)."""
    keep = set(keep)
    if all(v.variable in keep for v in cb.vars):
        return cb
    return replace(cb, vars=[v for v in cb.vars if v.variable in keep])


def records_path(cb: Codebook):
    """Where the records are in a reply: ["records"] for a codebook, the main
    array's path for a JSON schema, None when there are no row variables."""
    if cb.kind == "json":
        m = main_array_path(cb.schema)
        return None if m is None else m[0]
    if any(v.level == "row" for v in cb.vars):
        return ["records"]
    return None


def codebook_table(cb: Codebook) -> pd.DataFrame:
    return pd.DataFrame({
        "variable": [v.variable for v in cb.vars], "type": [v.type for v in cb.vars],
        "level": [v.level for v in cb.vars], "description": [v.description for v in cb.vars],
        "values": ["; ".join(map(str, v.values)) for v in cb.vars],
    }, columns=["variable", "type", "level", "description", "values"])


def codebook_template(file, overwrite: bool = False) -> Path:
    """Write a starter codebook (``.xlsx``, recommended, or ``.csv``) for a list
    of people, to change into your own; the Excel file has a second sheet that
    explains each column. Read it back with :func:`codebook`. Returns the path."""
    from ._text import write_csv
    from ._xlsx import SheetSpec, write_xlsx

    if not isinstance(file, (str, os.PathLike)):
        raise InputError("`file` must be a file name, e.g. 'codebook.xlsx'.")
    file = user_path(file)
    if os.path.exists(file) and not overwrite:
        raise InputError(f"{file} already exists.", "Choose another name, or use overwrite=True.")
    example = pd.DataFrame({
        "variable": ["printed_page", "surname", "first_name", "age", "sex", "occupation", "can_read"],
        "type": ["text", "text", "text", "integer", "category", "text", "yes/no"],
        "description": [
            "The page number printed on the page, exactly as printed (e.g. '12' or 'xii'). Read it, never compute it. Empty if there is none.",
            "Surname(s) exactly as written, with the original spelling and accents (e.g. 'Puig i Ferrer').",
            "First name(s) exactly as written (e.g. 'Josep Maria'). Do not expand abbreviations such as 'Ma.'.",
            "Age in years as written (e.g. 34). Empty if not given.",
            "Sex as recorded in the source.",
            "Occupation exactly as written (e.g. 'jornaler').",
            "Whether the column 'sabe leer' is marked for this person.",
        ],
        "values": ["", "", "", "", "male; female; other", "", ""],
        "level": ["page", "row", "row", "row", "row", "row", "row"],
    })
    how = pd.DataFrame({
        "column": ["variable", "type", "description", "values", "level"],
        "what_to_write": [
            "A short name for the variable, without spaces (use _ instead), e.g. surname.",
            "text (default), number, integer (whole number), yes/no, or category (one value from a fixed list). Keep dates and identifiers as text.",
            "What the model should write: what to copy, an example as printed and what to return, and what must never happen. This guides the model as much as the instructions do.",
            "Only for category: the allowed values, separated by ; (include a catch-all such as 'other').",
            "record (default): one value per row of the table or list on the page. page: one value for the whole page (a printed page number, a heading).",
        ],
    })
    e = ext(file)
    if e == "xlsx":
        write_xlsx({"codebook": SheetSpec(example), "how_to": SheetSpec(how)}, file)
    elif e == "csv":
        write_csv(example, file, bom=True)
    else:
        raise InputError("The template must be an .xlsx or .csv file.")
    done(f"Wrote {show_path(file)}.")
    info(f"Change the rows to your own variables, save, then use codebook({os.path.basename(str(file))!r}).")
    return Path(file)

"""From the model's JSON (one object per page) to one rectangular table: one
row per record, provenance first (file, source, source_page, row - stamped by
the package, never asked of the model), then the page-level variables
repeated on every row, then the record variables in codebook order. Pages
with no records do not appear here; the pages table lists every page.

For a JSON schema of the user's own the table is the "long table": one row
per element of the first array of objects, with the fields of any enclosing
arrays repeated beside it (prefixed with their path) and an ``<array>_n``
counter for each enclosing level. Nested objects become ``parent.child``
columns; lists of plain values are spread over numbered columns
(``cells_raw_1``, ``cells_raw_2``, ...). The JSON output keeps everything.

Column types: whole numbers are a nullable ``Int64`` column (pandas' integers
hold 2^63, so budget totals in pesetas never overflow); numbers ``float64``;
yes/no the nullable ``boolean``; text ``object`` with None for missing.
"""

from __future__ import annotations

import math
import re

import pandas as pd

from ._codebook import records_path, row_columns
from ._text import chr_value, is_missing, parse_yes_no, to_json

PROVENANCE = ("file", "source", "source_page", "row")


class _Spread(list):
    """A list of plain values, spread over numbered columns."""


def _is_object(x) -> bool:
    return isinstance(x, dict)


def object_scalars(obj: dict, prefix: str = "", skip=()) -> dict:
    """Scalars of an object, nested objects flattened to a.b names, arrays of
    objects skipped, arrays of plain values kept for spreading. ``skip`` is
    the path of the records array: only that array is left out, so the other
    fields of an object that encloses it stay page variables."""
    out = {}
    for nm, v in (obj or {}).items():
        inner_skip = ()
        if skip and nm == skip[0]:
            if len(skip) == 1 or not _is_object(v):
                continue
            inner_skip = tuple(skip[1:])
        full = f"{prefix}.{nm}" if prefix else nm
        if isinstance(v, dict) and v:
            out.update(object_scalars(v, full, inner_skip))
        elif isinstance(v, list):
            if v and all(isinstance(e, dict) for e in v):
                continue
            out[full] = _Spread(to_json(e) if isinstance(e, (dict, list)) else e for e in v)
        elif isinstance(v, dict):
            continue
        else:
            out[full] = v
    return out


def page_rows(data, path, rename=None) -> list[dict]:
    """The rows (dicts) of one page's records. An entry of an enclosing list
    whose own list is empty still gives one row, with its fields. ``rename``
    ({field: column}, from ``_codebook.row_columns``) names the entries'
    fields that share a name with a page field."""
    if not path:
        return []
    rename = rename or {}

    def walk(node, path, parents):
        arr = node.get(path[0]) if isinstance(node, dict) else None
        if _is_object(arr):
            return [] if len(path) == 1 else walk(arr, path[1:], parents)
        if not isinstance(arr, list) or not arr:
            return []
        rows = []
        for i, el in enumerate(arr, 1):
            if not isinstance(el, dict):
                continue
            if len(path) == 1:
                own = object_scalars(el)
                rows.append({**parents, **({rename.get(k, k): v for k, v in own.items()} if rename else own)})
            else:
                here = object_scalars(el, prefix=path[0])
                here[f"{path[0]}_n"] = i
                inner = walk(el, path[1:], {**parents, **here})
                rows += inner if inner else [{**parents, **here}]
        return rows

    return walk(data, list(path), {})


def counter_cols(path) -> list[str]:
    if not path or len(path) < 2:
        return []
    return [f"{p}_n" for p in path[:-1]]


def guess_type(vals) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "text"
    if all(isinstance(x, bool) for x in v):
        return "boolean"
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return "integer" if all(float(x) == int(x) for x in v if math.isfinite(float(x))) else "number"
    return "text"


def _num(v):
    if isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float)):
        return float(v)
    from ._text import as_number
    return as_number(v)


_INT_RE = re.compile(r"^\s*[-+]?[0-9]+\s*$")


def _whole(x) -> bool:
    if isinstance(x, int):
        return abs(x) < 2 ** 63
    return math.isfinite(x) and x == int(x) and abs(x) < 2 ** 63


def convert(values, type_: str) -> list:
    """JSON values as plain Python values of a codebook type (None = missing).

    Whole numbers stay Python integers, exact at any length (through float,
    16-digit identifiers came out equal). A yes/no value is read with the
    benchmark reader's spellings, and one that is none of them is missing,
    never False: on a page to check, "S" or "sí" counted as no."""
    out = []
    for v in values:
        if is_missing(v):
            out.append(None)
            continue
        if type_ in ("integer", "number"):
            if type_ == "integer" and isinstance(v, int) and not isinstance(v, bool):
                out.append(v)
            elif type_ == "integer" and isinstance(v, str) and _INT_RE.match(v):
                out.append(int(v))
            else:
                x = _num(v)
                out.append(None if math.isnan(x) else x)
        elif type_ == "boolean":
            out.append(v if isinstance(v, bool) else parse_yes_no(chr_value(v)))
        else:
            if isinstance(v, bool):
                out.append("true" if v else "false")
            elif isinstance(v, (int, float)):
                out.append(chr_value(v))
            elif isinstance(v, (dict, list)):
                out.append(to_json(v))
            else:
                out.append(str(v))
    if type_ == "integer":
        if all(x is None or _whole(x) for x in out):
            out = [None if x is None else int(x) for x in out]
    return out


def as_series(values, type_: str, name=None) -> pd.Series:
    vals = convert(values, type_)
    if type_ == "integer":
        if all(x is None or isinstance(x, int) for x in vals):
            return pd.Series(pd.array(vals, dtype="Int64"), name=name)
        return pd.Series([math.nan if x is None else x for x in vals], dtype="float64", name=name)
    if type_ == "number":
        return pd.Series([math.nan if x is None else x for x in vals], dtype="float64", name=name)
    if type_ == "boolean":
        return pd.Series(pd.array(vals, dtype="boolean"), name=name)
    return pd.Series(vals, dtype=object, name=name)


def flatten_records(recs, cb) -> pd.DataFrame:
    """The records table for a list of page records."""
    vars_ = cb.vars
    page_vars = [v.variable for v in vars_ if v.level == "page"]
    rec_vars = [v.variable for v in vars_ if v.level == "row"]
    types = {v.variable: v.type for v in vars_}
    path = records_path(cb)
    rename = row_columns(cb)
    rows = []
    for r in recs:
        if r.get("data") is None:
            continue
        prov = {"file": r["file"], "source": r.get("source") or r["file"], "source_page": r.get("source_page")}
        page_part = object_scalars(r["data"], skip=tuple(path or ()))
        if path is None:
            rows.append({**prov, **page_part})
        else:
            for k, row in enumerate(page_rows(r["data"], path, rename), 1):
                rows.append({**prov, "row": k, **page_part, **row})
    first = ["file", "source", "source_page"] if path is None else list(PROVENANCE)
    known = page_vars + rec_vars
    counters = counter_cols(path)
    if not rows:
        cols = {"file": pd.Series([], dtype=object), "source": pd.Series([], dtype=object),
                "source_page": pd.Series([], dtype="Int64")}
        if path is not None:
            cols["row"] = pd.Series([], dtype="Int64")
        for v in known:
            cols[v] = as_series([], types[v])
        return pd.DataFrame(cols)
    all_names = []
    for r in rows:
        for k in r:
            if k not in all_names:
                all_names.append(k)
    extra = [n for n in all_names if n not in first and n not in known]
    order = first + [v for v in page_vars if v in all_names] + extra + [v for v in rec_vars if v in all_names]
    order += [v for v in known if v not in order]
    out = {}
    for col in order:
        vals = [r.get(col) for r in rows]
        if any(isinstance(v, _Spread) for v in vals):
            width = max((len(v) for v in vals if isinstance(v, list)), default=0)
            for j in range(width):
                out[f"{col}_{j + 1}"] = as_series([v[j] if isinstance(v, list) and len(v) > j else None for v in vals], "text")
            continue
        if col in ("source_page", "row") or (col in counters and col not in known):
            t = "integer"
        elif col in ("file", "source"):
            t = "text"
        else:
            t = types.get(col) or guess_type(vals)
        out[col] = as_series(vals, t)
    return pd.DataFrame(out)


def page_unit(data, cb) -> dict:
    """One page for scoring: its page-level values (None when absent) and its
    rows (one list per row variable; a page with zero rows keeps its page values)."""
    page_vars = [v for v in cb.vars if v.level == "page"]
    row_vars = [v for v in cb.vars if v.level == "row"]
    path = records_path(cb)
    scal = {} if data is None else object_scalars(data, skip=tuple(path or ()))
    page = {v.variable: convert([scal.get(v.variable)], v.type)[0] for v in page_vars}
    raw = [] if data is None or path is None else page_rows(data, path, row_columns(cb))
    rows = {v.variable: convert([r.get(v.variable) for r in raw], v.type) for v in row_vars}
    return {"page": page, "rows": rows, "n": len(raw)}


def units_from_table(tab: pd.DataFrame, cb) -> dict:
    """The same units from a table in the results/benchmark layout (file, row,
    page variables on the first line of each page, row variables). A line
    with an empty ``row`` carries page values only."""
    page_vars = [v.variable for v in cb.vars if v.level == "page" and v.variable in tab.columns]
    row_vars = [v.variable for v in cb.vars if v.level == "row"]
    out = {}
    files = list(dict.fromkeys(tab["file"].tolist()))
    for f in files:
        sub = tab[tab["file"] == f]
        page = {}
        for v in page_vars:
            vals = [x for x in sub[v].tolist() if not is_missing(x) and str(x) != ""]
            page[v] = vals[0] if vals else None
        if "row" in sub.columns:
            has = [not is_missing(x) and str(x) != "" for x in sub["row"].tolist()]
        else:
            has = [True] * len(sub)
        rows_df = sub[has]
        if "row" in rows_df.columns:
            nums = [float(x) for x in rows_df["row"].tolist()]
            rows_df = rows_df.iloc[sorted(range(len(nums)), key=lambda i: nums[i])]
        rows = {v: [None if is_missing(x) else x for x in rows_df[v].tolist()] if v in rows_df.columns else [None] * len(rows_df)
                for v in row_vars}
        out[f] = {"page": page, "rows": rows, "n": len(rows_df)}
    return out

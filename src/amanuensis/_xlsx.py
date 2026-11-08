"""Excel files through openpyxl: every cell read as text, and sheets written
with a frozen header, column widths, Text-formatted columns and drop-down
lists.

Reading follows the R package's readxl calls with ``col_types = "text"``:
nothing is guessed, a number cell becomes its digits ("38", "2.5"), a
yes/no cell "TRUE"/"FALSE", and a date cell its Excel serial number
("46024"), with a second grid saying which cells Excel stored as dates (a
number typed 3/12 that Excel turned into a date must not pass for 46024), a
third with the number format of each number cell (01 shown, 1 stored), and
the rows hidden in Excel. Sheets are read in openpyxl's full mode, which
reads every cell whatever the file's <dimension> tag says.

Writing never lets a value become a formula: a text cell starting with "="
(a ditto mark "= id.") is stored as text, and characters Excel cannot hold
(control characters other than tab and line break) are left out and counted.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import re
import tempfile

from ._errors import InputError
from ._text import chr_value, is_missing, write_bytes_atomic

_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXCEL_MAX_CHARS = 32767
EXCEL_MAX_ROWS = 1048575
_EPOCH = _dt.datetime(1899, 12, 30)


def _open(path, read_only: bool = True):
    """The workbook, or a plain message: an .xls file and an encrypted or
    sensitivity-labelled one each get their own, instead of openpyxl's text
    followed by "close Excel"."""
    import openpyxl

    name = os.path.basename(str(path))
    if os.path.splitext(str(path))[1].lower() == ".xls":
        raise InputError(f"{name} is an old Excel (.xls) file, which cannot be read here.",
                         "Open it in Excel and save it as .xlsx (File > Save As > Excel Workbook).")
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as e:
        raise InputError(f"{name} could not be opened.", ("x", str(e.strerror or e)),
                         "If it is open in Excel, save it, close Excel and run the line again.") from None
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # an OLE2 container, not a zip: encrypted, or an .xls renamed
        raise InputError(f"{name} is protected by a password or a sensitivity label (or is an old .xls file renamed to "
                         ".xlsx), so it cannot be read here.",
                         "Save an unprotected copy as an Excel Workbook (.xlsx) and use that.")
    try:
        return openpyxl.load_workbook(path, read_only=read_only, data_only=True)
    except Exception as e:  # openpyxl raises many kinds for a damaged file
        raise InputError(f"{name} could not be opened as an Excel file.", ("x", str(e)),
                         "If it is open in Excel, save it, close Excel and run the line again.") from None


def sheet_names(path, visible: bool = False) -> list[str]:
    """The names of the sheets, in order; ``visible=True`` leaves out hidden ones."""
    wb = _open(path)
    try:
        return [ws.title for ws in wb.worksheets if not visible or getattr(ws, "sheet_state", "visible") == "visible"]
    finally:
        wb.close()


def _cell_text(v):
    if v is None:
        return None, False
    if isinstance(v, bool):
        return ("TRUE" if v else "FALSE"), False
    if isinstance(v, _dt.datetime):
        serial = (v - _EPOCH).total_seconds() / 86400
        return chr_value(serial), True
    if isinstance(v, _dt.date):
        return chr_value(float((_dt.datetime(v.year, v.month, v.day) - _EPOCH).days)), True
    if isinstance(v, _dt.time):
        return chr_value((v.hour * 3600 + v.minute * 60 + v.second) / 86400), True
    if isinstance(v, (int, float)):
        return chr_value(v), False
    return str(v), False


def read_cells(path, sheet: str) -> dict:
    """A sheet's cells: ``grid`` (text, None = empty), ``dates`` (which cells
    Excel stored as dates), ``formats`` (the number format of each number
    cell, None for other cells) and ``hidden`` (the sheet rows hidden in
    Excel, counted from 1).

    The workbook is read in openpyxl's full mode, not its read-only one: the
    read-only mode believes the sheet's <dimension> tag, which some writers
    other than Excel set too small, and then returns a truncated sheet
    without a word. Codebooks and benchmarks are small, so this costs
    nothing that matters."""
    wb = _open(path, read_only=False)
    try:
        ws = wb[sheet]
        hidden = sorted(int(k) for k, d in ws.row_dimensions.items() if getattr(d, "hidden", False))
        grid, dates, formats = [], [], []
        for row in ws.iter_rows():
            g, d, f = [], [], []
            for c in row:
                text, is_date = _cell_text(c.value)
                g.append(text)
                d.append(is_date)
                number = isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
                f.append(c.number_format if number else None)
            grid.append(g)
            dates.append(d)
            formats.append(f)
    finally:
        wb.close()
    width = max((len(r) for r in grid), default=0)
    # trailing empty columns and rows, as readxl leaves them out
    while width and all((len(r) < width or r[width - 1] is None) for r in grid):
        width -= 1
    grid = [(r + [None] * (width - len(r)))[:width] for r in grid]
    dates = [(d + [False] * (width - len(d)))[:width] for d in dates]
    formats = [(f + [None] * (width - len(f)))[:width] for f in formats]
    while grid and all(c is None for c in grid[-1]):
        grid.pop()
        dates.pop()
        formats.pop()
    return {"grid": grid, "dates": dates, "formats": formats, "hidden": [h for h in hidden if h <= len(grid)]}


def read_sheet(path, sheet: str):
    """A sheet as a grid of text (None = empty) and a grid of date flags."""
    c = read_cells(path, sheet)
    return c["grid"], c["dates"]


def read_table(path, sheet: str):
    """A sheet as a header plus rows of text (the first row is the header)."""
    grid, dates = read_sheet(path, sheet)
    if not grid:
        return [], [], []
    head = ["" if h is None else str(h) for h in grid[0]]
    return head, grid[1:], dates[1:]


# ------- writing ----------------------------------------------------------------

def _clean(value):
    """A value as openpyxl stores it, and how many characters had to go."""
    if is_missing(value):
        return None, 0
    try:
        import numpy as np
        if isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None, 0
    if isinstance(value, (bool, int, float)):
        return value, 0
    s = str(value)
    n = len(_ILLEGAL_RE.findall(s))
    if n:
        s = _ILLEGAL_RE.sub("", s)
    return s, n


class SheetSpec:
    """One sheet to write: a DataFrame plus its formatting."""

    def __init__(self, df, widths=None, freeze: str | None = "A2", text_columns: bool = False, validations=None):
        self.df = df
        self.widths = widths
        self.freeze = freeze
        self.text_columns = text_columns
        self.validations = validations or []


def col_letter(j: int) -> str:
    """1 -> A, 27 -> AA."""
    s = ""
    while j > 0:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


def auto_widths(df, cap: int = 60) -> list[float]:
    out = []
    for c in df.columns:
        col = df[c]
        vals = [chr_value(v) for v in col.tolist()[:500]]
        longest = max([len(str(c))] + [len(v.split("\n")[0]) for v in vals if v is not None])
        out.append(min(max(longest + 2, 8), cap))
    return out


def write_xlsx(sheets: dict, path, keep_other_sheets: bool = False, replaces=()) -> int:
    """Write the sheets (name -> DataFrame or SheetSpec) to ``path``, atomically.
    Returns the number of characters left out (control characters).

    ``keep_other_sheets``: when ``path`` exists, the sheets it has besides
    these (and besides those named in ``replaces``) stay, after these, with
    their contents and formatting (what openpyxl can keep: not charts)."""
    import openpyxl
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = None
    if keep_other_sheets and os.path.exists(path):
        try:
            wb = openpyxl.load_workbook(path)
        except Exception:
            wb = None
        if wb is not None:
            for name in list(sheets) + [r for r in replaces if r]:
                if name in wb.sheetnames:
                    wb.remove(wb[name])
            if not wb.sheetnames:
                wb = None
    if wb is None:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
    dropped = 0
    for pos, (name, spec) in enumerate(sheets.items()):
        if not isinstance(spec, SheetSpec):
            spec = SheetSpec(spec)
        df = spec.df
        ws = wb.create_sheet(title=name, index=pos)
        cols = [str(c) for c in df.columns]
        ws.append(cols)
        columns = [df[c].tolist() for c in df.columns]
        for i in range(len(df)):
            row = []
            for col in columns:
                v, n = _clean(col[i])
                dropped += n
                row.append(v)
            ws.append(row)
        # strings starting with "=" must stay text, never become formulas
        for row in ws.iter_rows(min_row=1):
            for cell in row:
                if cell.data_type == "f":
                    cell.data_type = "s"
        widths = spec.widths if spec.widths is not None else auto_widths(df)
        for j in range(1, len(cols) + 1):
            w = widths[min(j, len(widths)) - 1] if widths else 12
            ws.column_dimensions[col_letter(j)].width = w
        if spec.text_columns:
            # the column's own format covers the cells typed later; the cells
            # written now carry it too
            for j in range(1, len(cols) + 1):
                ws.column_dimensions[col_letter(j)].number_format = "@"
                for i in range(2, len(df) + 2):
                    ws.cell(row=i, column=j).number_format = "@"
        for v in spec.validations:
            dv = DataValidation(**v["args"])
            dv.add(v["range"])
            ws.add_data_validation(dv)
        if spec.freeze:
            ws.freeze_panes = spec.freeze
    wb.active = 0
    for ws in wb.worksheets:
        ws.sheet_view.tabSelected = ws is wb.worksheets[0]
    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        try:
            wb.save(tmp)
        except Exception as e:
            raise InputError(f"Could not write {path}: {e}",
                             "If the file is open in Excel, close it and run the line again.") from None
        with open(tmp, "rb") as f:
            write_bytes_atomic(f.read(), path)
    finally:
        os.unlink(tmp)
    return dropped

"""The accuracy report score() writes: one self-contained HTML file (inline CSS
and JS, nothing from elsewhere), so it can be e-mailed, archived beside the
data, or opened years later without a network.

Policies:

- Built by substituting @@MARKERS@@ into one TEMPLATE string; every marker
  must appear exactly once and none may survive, so a misspelt marker fails
  loudly. Data never contains "@" (escaped as &#64;), so it cannot fake one.
- Deterministic: no clock is read; the only dates shown are those of the
  extraction and of the benchmark's draw. The same score gives the same bytes.
- Every table is a plain <table> with a header row, no colspan or rowspan and
  no <th> in the body, so pandas.read_html() can read it back.
- Light or dark follows the reader's system (prefers-color-scheme).
"""

from __future__ import annotations

import math
import os
import re

from ._say import done, plural
from ._text import fmt_int, write_text
from ._version import __version__


def write_score_report(s, path: str) -> str:
    html = score_html(s)
    existed = os.path.exists(path)
    write_text(html, path)
    done(f"Wrote the report {os.path.realpath(path)}{' (replaced)' if existed else ''}.")
    return path


def html_escape(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = str(x)
    return (x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            .replace("'", "&#39;").replace("@", "&#64;"))


def fill_template(template: str, values: dict) -> str:
    pieces = template.split("@@")
    if len(pieces) % 2 != 1:
        raise AssertionError("a report marker is not closed")
    markers = pieces[1::2]
    if len(set(markers)) != len(markers) or set(markers) != set(values):
        raise AssertionError(f"report markers do not match: {sorted(set(markers) ^ set(values))}")
    pieces[1::2] = [values[m] for m in markers]
    out = "".join(pieces)
    if "@@" in out:
        raise AssertionError("a report marker survived")
    return out


def pct2(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "&ndash;"
    s = f"{100 * v:.2f}%"
    if v < 1 and s == "100.00%":
        return "&gt;99.99%"
    if v > 0 and s == "0.00%":
        return "&lt;0.01%"
    return s


def score_html(s) -> str:
    from ._score import score_subtitle

    a = s.about
    agree = a.get("reference") == "run"
    title = "Agreement with the first run" if agree else "Accuracy against your benchmark"
    cells = [dict(r) for r in s.cells.to_dict("records")] if len(s.cells) else []
    values = {
        "TITLE": html_escape(title), "HEADING": html_escape(title), "SUBTITLE": html_escape(score_subtitle(a)),
        "STAMP": report_stamp(a), "TILES": report_tiles(s, a, agree), "NOTES": report_notes(s, a, agree),
        "VARIABLES": report_variables(s, a), "COUNTS": report_counts(cells), "PAGES": report_pages(s, a, cells),
        "METHOD": report_method(a),
        "FOOTER": html_escape(f"Made with amanuensis {__version__} (Python). The same score always gives the same file."),
    }
    return fill_template(TEMPLATE, values)


def _folder_label(path) -> str:
    return f" (folder {os.path.basename(str(path))})" if path else ""


def report_stamp(a) -> str:
    bits = []
    if a.get("reference") == "run":
        bits.append(f"First run: {a.get('reference_model') or ''}{_folder_label(a.get('source'))}")
    else:
        src = os.path.basename(a["source"]) if a.get("source") else "a table given in Python"
        d = a.get("draw") or {}
        draw = ""
        if d.get("seed") is not None:
            seed = int(d["seed"]) if float(d["seed"]) == int(d["seed"]) else d["seed"]
            draw = f" (pages drawn with seed {seed}" + (f" on {str(d['date'])[:10]}" if d.get("date") else "") + ")"
        bits.append(f"Benchmark: {src}{draw}")
    dates = a.get("output_dates")
    when = ""
    if dates:
        when = f", extracted {dates[0] if dates[0] == dates[-1] else dates[0] + ' to ' + dates[-1]}"
    if a.get("model"):
        bits.append(f"{'Second run: ' if a.get('reference') == 'run' else 'Output: '}{a['model']}{when}{_folder_label(a.get('output_folder'))}")
    pairing = {"auto": "by position, or by content where the row counts differ", "position": "by position", "content": "by content"}[a["align"]]
    if a["align"] == "auto" and a.get("rows_limit"):
        pairing = (f"by content where the benchmark reaches its {a['rows_limit']} rows; elsewhere by position, or by content "
                   "where the row counts differ")
    if a["row_variables"] and a["mode"] != "text":
        bits.append(f"Rows paired {pairing}")
    if a.get("rows_limit"):
        bits.append(f"first {a['rows_limit']} rows of each page")
    return " &middot; ".join(html_escape(b) for b in bits)


def report_tiles(s, a, agree) -> str:
    sm = s.summary
    right = "the same" if agree else "right"

    def get(m):
        r = sm[sm["measure"] == m]
        return None if not len(r) else r.iloc[0].to_dict()

    def tile(label, r, detail, len_=None):
        v = pct2(r["value"]) if r is not None else "&ndash;"
        lt = f'<div class="d">lenient {pct2(len_["value"])}</div>' if len_ is not None else ""
        return (f'<div class="tile"><div class="k">{html_escape(label)}</div><div class="v">{v}</div>'
                f'<div class="d">{detail}</div>{lt}</div>')

    def lenient(m):
        return get(m + " (lenient)") if a["normalize"] else None

    def counts(r, what):
        if r is None or r["compared"] == 0:
            return f"no {what} to compare"
        return f"{fmt_int(r['matched'])} of {fmt_int(r['compared'])} {what}"

    def edits(r, what):
        if r is None or r["compared"] == 0:
            return f"no {what} to compare"
        return f"{fmt_int(r['compared'] - r['matched'])} edits over {fmt_int(r['compared'])} {what}"

    if a["mode"] == "text":
        return (tile(f"Characters {right}", get("characters"), edits(get("characters"), "characters"), lenient("characters"))
                + tile(f"Words {right}", get("words"), edits(get("words"), "words"), lenient("words"))
                + tile("Pages identical" if agree else "Pages entirely right", get("pages"), counts(get("pages"), "pages")))
    pv = get("page values")
    rows_detail = counts(get("rows"), "rows") if a["row_variables"] else "no row variables"
    if pv is not None and a["row_variables"]:
        rows_detail += f" &middot; page values: {fmt_int(pv['matched'])} of {fmt_int(pv['compared'])} pages"
    return (tile(f"Rows with every value {right}", get("rows"), rows_detail, lenient("rows"))
            + tile(f"Values {right}", get("values"), counts(get("values"), "values"), lenient("values"))
            + tile(f"Characters {right} (text)", get("characters"), edits(get("characters"), "characters"), lenient("characters")))


def report_notes(s, a, agree) -> str:
    from ._score import doubt_sentence

    notes = list(s.notes)
    dm = s.summary[s.summary["measure"] == "doubt marks"]
    if len(dm) and dm.iloc[0]["matched"] > 0:
        notes.append(doubt_sentence(dm.iloc[0].to_dict(), agree) + ".")
    if not notes:
        return ""
    return '<ul class="notes">' + "".join(f"<li>{html_escape(n)}</li>" for n in notes) + "</ul>"


def html_table(head, body, cls=None) -> str:
    opening = f'<table class="{cls}">' if cls else "<table>"
    return (opening + "<thead><tr>" + "".join(f"<th>{html_escape(h)}</th>" for h in head)
            + "</tr></thead><tbody>\n" + "\n".join(body) + "\n</tbody></table>")


def report_variables(s, a) -> str:
    bv = s.by_variable
    top = max([0] + [int(x) for x in bv["differences"]]) if len(bv) else 0
    len_ = a["normalize"]
    head = ["Variable", "Type", "Level", "Compared", "Wrong", "Mismatches", "Values right", "Characters right", "Edits"]
    if len_:
        head += ["Wrong (lenient)", "Values right (lenient)", "Characters right (lenient)"]
    body = []
    for _, r in bv.iterrows():
        w = f"{100 * r['differences'] / top:.1f}" if top > 0 else "0.0"
        ed = r["edit_distance"]
        cells = [f"<td>{html_escape(r['variable'])}</td>", f"<td>{html_escape(r['type'])}</td>", f"<td>{html_escape(r['level'])}</td>",
                 f'<td class="num">{fmt_int(r["compared"])}</td>', f'<td class="num">{fmt_int(r["differences"])}</td>',
                 f'<td><div class="bar"><span style="width:{w}%"></span></div></td>',
                 f'<td class="num">{pct2(r["variable_match"])}</td>', f'<td class="num">{pct2(r["character_match"])}</td>',
                 f'<td class="num">{"&ndash;" if ed is None or (isinstance(ed, float) and math.isnan(ed)) else fmt_int(ed)}</td>']
        if len_:
            cells += [f'<td class="num">{fmt_int(r["lenient_differences"])}</td>',
                      f'<td class="num">{pct2(r["lenient_variable_match"])}</td>',
                      f'<td class="num">{pct2(r["lenient_character_match"])}</td>']
        body.append("<tr>" + "".join(cells) + "</tr>")
    return html_table(head, body, "vars")


def report_counts(cells) -> str:
    wrong = [c for c in cells if not c["match"]]
    rows = len({(c["file"], c["pair"]) for c in wrong if c["level"] == "row"})
    return f"{plural(rows, 'row')} and {plural(len(wrong), 'value')} differ"


def report_method(a) -> str:
    if a["mode"] == "text":
        p = ["Characters right: one minus the number of single-character edits (insertions, deletions, substitutions) needed to turn "
             "the output into the benchmark, divided by the length of the longer of the two texts, summed over pages.",
             "Words right: the same count made on whole words, so a misread word costs one edit however long it is."]
    else:
        p = ["Rows with every value right: rows of the benchmark and of the output paired on each page; a row counts as right only if "
             "every variable matches. A row missing from the output, or added by it, counts as a wrong row.",
             "Values right: values that match, out of all values compared; page-level variables count once per page. An empty cell "
             "matches an empty cell; text is compared exactly as written, after a standard Unicode normalisation; numbers and yes/no by value.",
             "Characters right (text variables only): one minus the number of single-character edits needed to turn the output into the "
             "benchmark, divided by the number of characters (the longer of the two values each time). A small misreading costs little "
             "here and a whole value in the other two figures."]
    p.append("A page whose extraction failed counts as entirely wrong; a page not extracted yet is left out and listed.")
    if a["normalize"]:
        p.append("Lenient figures ignore capital letters, accents, punctuation (including doubt marks) and spacing; numbers are never changed.")
    return "\n".join(f'<p class="meta">{html_escape(x)}</p>' for x in p)


def mark_pair(a: str, b: str):
    """Mark what lies between the common beginning and the common end."""
    lim = min(len(a), len(b))
    p = 0
    while p < lim and a[p] == b[p]:
        p += 1
    s = 0
    while s < lim - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1

    def fmt(x, cls):
        pre, mid, suf = x[:p], x[p: len(x) - s], x[len(x) - s:] if s else ""
        return html_escape(pre) + (f'<mark class="{cls}">{html_escape(mid)}</mark>' if mid else "") + html_escape(suf)

    return fmt(a, "g"), fmt(b, "o")


def word_diff_html(a: str, b: str):
    """Whole texts: words paired by a longest common subsequence; unpaired words marked."""
    import numpy as np
    from ._score import dp_align

    ta = re.findall(r"\s+|\S+", a) if a else []
    tb = re.findall(r"\s+|\S+", b) if b else []
    wa = [i for i, t in enumerate(ta) if not t[0].isspace()]
    wb = [i for i, t in enumerate(tb) if not t[0].isspace()]
    if len(wa) * len(wb) > 4e6:
        return mark_pair(a, b)
    keep_a, keep_b = set(), set()
    if wa and wb:
        eq = np.array([[ta[i] == tb[j] for j in wb] for i in wa])
        for gi, oi in dp_align(eq.astype(float), eq):
            if gi is not None and oi is not None:
                keep_a.add(wa[gi - 1])
                keep_b.add(wb[oi - 1])

    def fmt(tok, ws, keep, cls):
        out = [html_escape(t) for t in tok]
        for i in ws:
            if i not in keep:
                out[i] = f'<mark class="{cls}">{out[i]}</mark>'
        return "".join(out)

    return fmt(ta, wa, keep_a, "g"), fmt(tb, wb, keep_b, "o")


def _show(v, miss):
    if miss:
        return '<span class="miss">missing</span>'
    if v is None:
        return '<span class="null">empty</span>'
    return html_escape(v)


def cell_html(c) -> str:
    g, o = _show(c["gold"], c["g_missing"]), _show(c["output"], c["o_missing"])
    if not c["match"] and not c["g_missing"] and not c["o_missing"] and c["gold"] is not None and c["output"] is not None:
        g, o = mark_pair(str(c["gold"]), str(c["output"]))
    if c["match"]:
        return f"<td>{g}</td>"
    return f'<td class="x"><div class="g">{g}</div>\n<div class="o">{o}</div></td>'


def _match_td(wrong, n):
    return '<td class="ok">&#10003;</td>' if wrong == 0 else f'<td class="bad">&#10007; {wrong} of {n}</td>'


def report_pages(s, a, cells) -> str:
    vars_ = a["variables"]
    by_file = {}
    for c in cells:
        by_file.setdefault(c["file"], []).append(c)
    out = []
    for _, p in s.by_page.iterrows():
        if p["status"] == "not run":
            out.append(f'<section class="page"><h3>{html_escape(p["file"])}</h3><p class="meta">not in the results yet: left out of every figure</p></section>')
            continue
        sub = by_file.get(p["file"], [])
        meta = [p["status"]]
        if p["alignment"] in ("position", "content"):
            meta.append(f"rows paired by {p['alignment']}")
        if p["gold_rows"] is not None and not _na(p["gold_rows"]):
            meta.append(f"{a['labels']['ref_col']} {plural(int(p['gold_rows']), 'row')}")
        if p["output_rows"] is not None and not _na(p["output_rows"]):
            meta.append(f"{a['labels']['out_col']} {plural(int(p['output_rows']), 'row')}")
        if not _na(p["missing_rows"]) and p["missing_rows"] > 0:
            meta.append(f"{int(p['missing_rows'])} missing")
        if not _na(p["extra_rows"]) and p["extra_rows"] > 0:
            meta.append(f"{int(p['extra_rows'])} extra")
        meta.append(f"{plural(int(p['differences'] or 0), 'value')} wrong")
        parts = [f'<h3>{html_escape(p["file"])}</h3><p class="meta">{" &middot; ".join(html_escape(m) for m in meta)}</p>']
        if a["mode"] == "text":
            parts.append(report_text_table(sub, a))
        else:
            pc = [c for c in sub if c["level"] == "page" and not c["separate"]]
            if pc:
                parts.append(report_page_values(pc, vars_, a))
            sc = [c for c in sub if c["separate"]]
            if sc:
                parts += ['<p class="meta">Other text on the page:</p>', report_text_table(sc, a)]
            rc = [c for c in sub if c["level"] == "row"]
            if rc:
                parts.append(report_rows(rc, vars_, a))
        cls = "page same" if int(p["differences"] or 0) == 0 else "page"
        out.append(f'<section class="{cls}">' + "\n".join(parts) + "</section>")
    return "\n".join(out)


def _na(x) -> bool:
    try:
        import pandas as pd
        return x is None or pd.isna(x)
    except (TypeError, ValueError):
        return False


def report_page_values(pc, vars_, a) -> str:
    pv = [v.variable for v in vars_ if v.level == "page" and v.variable not in a["separate"]]
    by_v = {c["variable"]: c for c in pc}
    tds = [cell_html(by_v[v]) for v in pv if v in by_v]
    wrong = sum(not c["match"] for c in pc)
    tr = f'<tr class="{"diff" if wrong else "same"}"><td>page values</td>{"".join(tds)}{_match_td(wrong, len(pc))}</tr>'
    return '<div class="scroll">' + html_table([""] + pv + ["Match"], [tr], "cmp") + "</div>"


def report_rows(rc, vars_, a) -> str:
    rv = [v.variable for v in vars_ if v.level == "row"]
    pairs = {}
    for c in rc:
        pairs.setdefault(c["pair"], {})[c["variable"]] = c
    trs = []
    for k in sorted(pairs):
        row = pairs[k]
        first = row[rv[0]] if rv and rv[0] in row else next(iter(row.values()))
        wrong = sum(not row[v]["match"] for v in rv if v in row)

        def num(x):
            return '<span class="miss">&mdash;</span>' if x is None or _na(x) else str(int(x))

        trs.append(f'<tr class="{"diff" if wrong else "same"}"><td class="num">{num(first["row_gold"])}</td>'
                   f'<td class="num">{num(first["row_output"])}</td>' + "".join(cell_html(row[v]) for v in rv if v in row)
                   + _match_td(wrong, len(rv)) + "</tr>")
    head = [f"Row ({a['labels']['ref_col']})", f"Row ({a['labels']['out_col']})"] + rv + ["Match"]
    return '<div class="scroll">' + html_table(head, trs, "cmp") + "</div>"


def report_text_table(tc, a) -> str:
    trs = []
    for c in tc:
        if c["match"]:
            txt = '<span class="null">empty</span>' if c["gold"] is None else html_escape(c["gold"])
            trs.append(f'<tr class="same"><td class="txt">{txt}</td><td class="txt">{txt}</td></tr>')
            continue
        if c["o_missing"]:
            m = (html_escape(c["gold"]), '<span class="miss">missing</span>')
        else:
            m = word_diff_html(c["gold"] or "", c["output"] or "")
        trs.append(f'<tr class="diff"><td class="txt">{m[0]}</td><td class="txt">{m[1]}</td></tr>')
    head = [a["labels"]["ref_col"][:1].upper() + a["labels"]["ref_col"][1:], a["labels"]["out_col"][:1].upper() + a["labels"]["out_col"][1:]]
    return '<div class="scroll">' + html_table(head, trs, "cmp text") + "</div>"


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
:root{color-scheme:light dark;--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--card:#f6f8fa;
--bad:#ffebe9;--badline:#ff8182;--good:#1a7f37;--bar:#cf222e;--mg:#cfe8ff;--mo:#ffcecb;--miss:#818b98}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;
--card:#151b23;--bad:#3c1618;--badline:#8e1519;--good:#3fb950;--bar:#f85149;--mg:#0c2d6b;--mo:#6e1a1f;--miss:#768390}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;padding:24px 16px 48px;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
main{max-width:1200px;margin:0 auto}
h1{font-size:24px;line-height:1.25;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 8px}
h3{font-size:16px;margin:0 0 4px;word-break:break-all}
.sub{color:var(--muted);margin:0 0 4px}
.stamp{color:var(--muted);font-size:13px;margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}
.tile .k{color:var(--muted);font-size:13px}
.tile .v{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .d{font-size:13px;color:var(--muted)}
ul.notes{padding-left:20px;margin:8px 0}
.scroll{overflow-x:auto;margin:8px 0 16px;max-width:100%}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
thead th{background:var(--card);white-space:nowrap}
td{white-space:pre-wrap}
td.num{text-align:right;white-space:nowrap}
.bar{width:120px;height:10px;background:var(--card);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--bar)}
td.x{background:var(--bad);border-color:var(--badline)}
td.x .o{border-top:1px dashed var(--badline);margin-top:2px;padding-top:2px}
mark{color:inherit;border-radius:2px}
mark.g{background:var(--mg)}
mark.o{background:var(--mo)}
.null,.miss{color:var(--miss);font-style:italic}
td.ok{color:var(--good);white-space:nowrap}
td.bad{color:var(--bar);white-space:nowrap}
td.txt{min-width:260px;max-width:560px}
.page{border-top:1px solid var(--line);padding-top:16px;margin-top:16px}
.meta{color:var(--muted);font-size:13px;margin:0 0 8px}
.toggle{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}
.toggle label{cursor:pointer;font-weight:600}
body.only tr.same,body.only section.same{display:none}
footer{color:var(--muted);font-size:12px;margin-top:40px}
</style>
</head>
<body>
<main>
<h1>@@HEADING@@</h1>
<p class="sub">@@SUBTITLE@@</p>
<p class="stamp">@@STAMP@@</p>
<div class="tiles">@@TILES@@</div>
@@NOTES@@
<h2>By variable</h2>
<div class="scroll">@@VARIABLES@@</div>
<h2>Page by page</h2>
<div class="toggle"><label><input type="checkbox" id="only"> Show only differences</label>
<span class="meta">&middot; @@COUNTS@@</span></div>
@@PAGES@@
<h2>How the figures are made</h2>
@@METHOD@@
<footer>@@FOOTER@@</footer>
</main>
<script>
(function(){var c=document.getElementById("only");if(!c)return;
function u(){document.body.classList.toggle("only",c.checked);}
c.addEventListener("change",u);u();})();
</script>
</body>
</html>
"""

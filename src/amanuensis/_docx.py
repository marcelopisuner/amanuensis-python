"""A minimal Word writer with no extra packages. A .docx file is a zip of a
few XML files; this writes the five a word processor needs (the content
types, the package relationships, the document, its relationships and a
style sheet). Word, LibreOffice, Pages and macOS textutil open the result.

- Only what the package needs: a title, headings, paragraphs and plain tables.
- Text is written exactly: every character escaped for XML, line breaks kept
  as Word line breaks and tabs as Word tabs, so instructions() reads back the
  same text it was given. Control characters XML forbids (they make Word
  refuse the whole file) are left out; ``control_chars()`` counts them.
- Headings use Word's built-in "heading 1" and "heading 2" styles; tables
  also carry their borders, bold header and font size directly, because
  LibreOffice ignores parts of table styles.
- Pages are A4; a document with a table of more than seven columns is set in
  landscape. The zip entries carry a fixed date, so the same content gives
  the same bytes.
"""

from __future__ import annotations

import io
import re
import zipfile

from ._text import chr_value, is_missing, write_bytes_atomic

_BAD_XML = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x00]")
_FIXED_DATE = (2026, 1, 1, 0, 0, 0)


def xml_escape(x) -> str:
    x = "" if is_missing(x) else str(x)
    x = _BAD_XML.sub("", x)
    return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def control_chars(values) -> int:
    """How many characters xml_escape() would drop from these values."""
    return sum(len(_BAD_XML.findall(str(v))) for v in values if not is_missing(v))


def cell_text(x) -> str:
    """A value in a document: empty for missing, yes/no for booleans, numbers without scientific notation."""
    if is_missing(x):
        return ""
    if isinstance(x, bool) or type(x).__name__ == "bool_":
        return "yes" if bool(x) else "no"
    s = chr_value(x)
    return "" if s is None else s


def _run(text, bold=False, size=None) -> str:
    text = "" if is_missing(text) else str(text)
    if not text:
        return ""
    rpr = ("<w:b/>" if bold else "") + (f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' if size else "")
    lines = re.sub(r"\r\n?", "\n", text).split("\n")
    inner = []
    for line in lines:
        pieces = line.split("\t")
        inner.append("<w:tab/>".join(f'<w:t xml:space="preserve">{xml_escape(p)}</w:t>' for p in pieces))
    return "<w:r>" + (f"<w:rPr>{rpr}</w:rPr>" if rpr else "") + "<w:br/>".join(inner) + "</w:r>"


def _paragraph(text, style=None, bold=False, size=None, spacing_after=None) -> str:
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + (f'<w:spacing w:after="{spacing_after}"/>' if spacing_after is not None else "")
    return "<w:p>" + (f"<w:pPr>{ppr}</w:pPr>" if ppr else "") + _run(text, bold, size) + "</w:p>"


def _table(df, text_width: int) -> str:
    cols = [str(c) for c in df.columns]
    if not cols:
        return _paragraph("")
    cells = [[cell_text(v) for v in df[c].tolist()] for c in df.columns]
    weight = []
    for j, h in enumerate(cols):
        vals = cells[j][:200]
        longest = max([len(h)] + [len(part) for v in vals for part in v.split("\n")] + [1])
        weight.append(min(max(longest, 3), 40))
    widths = [max(500, int(text_width * w / sum(weight))) for w in weight]

    def border(side):
        return f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'

    tblpr = ('<w:tblPr><w:tblStyle w:val="AmanuensisTable"/><w:tblW w:w="0" w:type="auto"/><w:tblBorders>'
             + "".join(border(s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
             + '</w:tblBorders><w:tblLayout w:type="autofit"/>'
             '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>')
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"

    def cell(text, w, header=False):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if header else ""
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shade}</w:tcPr>'
                + _paragraph(text, bold=header, size=18, spacing_after=0) + "</w:tc>")

    header = "<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(cell(h, w, True) for h, w in zip(cols, widths)) + "</w:tr>"
    rows = ["<w:tr>" + "".join(cell(cells[j][i], widths[j]) for j in range(len(cols))) + "</w:tr>" for i in range(len(df))]
    return "<w:tbl>" + tblpr + grid + header + "".join(rows) + "</w:tbl>"


def _block(b: dict, text_width: int) -> str:
    t = b.get("type", "para")
    if t == "table":
        return _table(b["table"], text_width)
    style = {"title": "Title", "heading": "Heading2" if b.get("level") == 2 else "Heading1", "para": None}[t]
    text = b.get("text") or ""
    if isinstance(text, (list, tuple)):
        text = "\n".join(map(str, text))
    return _paragraph(text, style=style)


def docx_bytes(parts: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, xml in parts.items():
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, xml.encode("utf-8"))
    return buf.getvalue()


def write_docx(blocks: list, path, landscape: bool | None = None) -> str:
    """Blocks: dicts with type "title" | "heading" | "para" | "table", and
    text, table (a DataFrame) or level (1 or 2 for headings)."""
    if landscape is None:
        landscape = any(b.get("type") == "table" and len(b["table"].columns) > 7 for b in blocks)
    w, h = (16838, 11906) if landscape else (11906, 16838)
    margin = 1134  # 2 cm
    body = "".join(_block(b, w - 2 * margin) for b in blocks)
    sect = (f'<w:sectPr><w:pgSz w:w="{w}" w:h="{h}"' + (' w:orient="landscape"' if landscape else "") + "/>"
            f'<w:pgMar w:top="{margin}" w:right="{margin}" w:bottom="{margin}" w:left="{margin}" w:header="709" '
            'w:footer="709" w:gutter="0"/></w:sectPr>')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f"<w:body>{body}{sect}</w:body></w:document>")
    parts = {"[Content_Types].xml": CONTENT_TYPES, "_rels/.rels": RELS, "word/document.xml": document,
             "word/styles.xml": STYLES, "word/_rels/document.xml.rels": DOCUMENT_RELS}
    return write_bytes_atomic(docx_bytes(parts), path)


def write_plain_docx(paragraphs: list, path) -> str:
    """One paragraph per line, tabs kept (the free-text benchmark file); read
    back by instructions' docx reader, which joins paragraphs with line breaks."""
    def esc(x):
        return x.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    paras = []
    for p in paragraphs:
        if not p:
            paras.append("<w:p></w:p>")
            continue
        t = esc(_BAD_XML.sub("", p)).replace("\t", '</w:t><w:tab/><w:t xml:space="preserve">')
        paras.append(f'<w:p><w:r><w:t xml:space="preserve">{t}</w:t></w:r></w:p>')
    parts = {
        "[Content_Types].xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                                '<Default Extension="xml" ContentType="application/xml"/>'
                                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                                "</Types>"),
        "_rels/.rels": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                        "</Relationships>"),
        "word/document.xml": ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                              '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                              + "".join(paras) + "<w:sectPr/></w:body></w:document>"),
    }
    return write_bytes_atomic(docx_bytes(parts), path)


CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    "</Types>")

RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>")

DOCUMENT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>")

STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:docDefaults><w:rPrDefault><w:rPr>"
    '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri" w:cs="Calibri"/>'
    '<w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>'
    '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
    "</w:docDefaults>"
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="240"/></w:pPr>'
    '<w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>'
    '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="360" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>'
    '<w:rPr><w:b/><w:color w:val="1F3864"/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>'
    '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr>'
    '<w:rPr><w:b/><w:color w:val="1F3864"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>'
    '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/>'
    '<w:uiPriority w:val="99"/><w:semiHidden/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/>'
    '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
    '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>'
    '<w:style w:type="table" w:styleId="AmanuensisTable"><w:name w:val="Amanuensis Table"/>'
    '<w:basedOn w:val="TableNormal"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
    '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:tblPr><w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    "</w:tblBorders></w:tblPr>"
    '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr>'
    '<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/></w:tcPr></w:tblStylePr>'
    "</w:style></w:styles>")

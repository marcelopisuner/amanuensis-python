"""The instructions (prompt) sent with every page.

The prompt has four parts - Role, Task, Rules, Details - and the Rules part
is the same for every source: transcribe exactly, "?" for doubt, null for
illegible, never invent, only this image. So ``instructions()`` appends that
standard block by default; a user writes only what is specific to the source.

The standard rules are part of the contract hashed into every result, so a
published text is never edited: a new wording gets a new id and the old ids
stay available (``instructions(rules="rules-1")``), so a folder extracted
under an older version can be continued unchanged. The texts are the R
package's, byte for byte, so the same instructions hash the same in both.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ._errors import InputError, NotFoundError
from ._say import done, info, warn
from ._text import ext, nfc, read_text, sha12, show_path, user_path, write_text

DEFAULT_INSTRUCTIONS = "\n".join([
    "# Role",
    "You are transcribing one image of a historical document for a research project.",
    "",
    "# Task",
    "Read the image and return what the response schema asks for, for this image only.",
])

RULES_TEXTS = {
    "rules-1": "\n".join([
        "# Rules",
        "- Transcribe exactly as written: keep the original spelling, abbreviations, capitalisation, accents and punctuation, including the document's own mistakes. Never correct, expand, modernise or translate.",
        "- Never compute anything: copy numbers, dates and page numbers as they are written, even when they look wrong.",
        "- If you are unsure of a reading in a text field, give your best reading followed by \"?\" (for example: Puig?). In number, yes/no and category fields give your best reading without a question mark.",
        "- If something cannot be read at all, or is not on the page, return null for that field; inside running text, write [...] for an illegible passage.",
        "- Never invent or guess anything that is not visible. Use only what is on THIS image; do not carry anything over from other pages.",
        "- Printed marks such as a dash, a ditto mark (\", „, id., idem) or a blank are content: copy dashes and ditto marks as written and never replace them with the value they stand for.",
        "- Ignore library stamps, rulers, colour charts and slivers of a neighbouring page.",
        "- If the text contains a backslash, write it as \\\\ so the JSON stays valid.",
    ]),
}
RULES_ID = "rules-1"

_TEXT_EXTS = ("txt", "md", "markdown", "tex", "text")
_FILE_EXTS = _TEXT_EXTS + ("docx", "doc", "rtf")


@dataclass(frozen=True)
class Instructions:
    """What is sent to the model with every page (print it to see it all).

    ``text`` is what you wrote, ``full`` what is sent (with the standard
    rules unless ``rules=False``), ``sha12`` the hash that identifies it in
    the results. It is read-only (its sha12 is taken when it is made): to
    change it, call instructions() again.
    """

    text: str
    rules: bool
    rules_id: str
    full: str
    source: str
    sha12: str

    def __repr__(self) -> str:
        src = {"text": "typed in Python", "default": "the package's default"}.get(self.source, os.path.basename(self.source))
        words = len(self.full.split())
        lines = self.full.split("\n")
        show = lines[:30] + ([f"... ({len(lines) - 30} more lines)"] if len(lines) > 30 else [])
        rule = "-" * 60
        out = [f"Instructions ({src}, {words} words{', standard rules added' if self.rules else ''})", rule, *show, rule]
        if not self.rules:
            out.append("The standard transcription rules are NOT added (rules=False).")
        return "\n".join(out)

    __str__ = __repr__


def instructions(file=None, text: str | None = None, rules=True) -> Instructions:
    """The instructions for the model, from a file or typed as text.

    Write what the source is, what one image shows, what to return, and
    anything unusual about how the source is written (reading order,
    abbreviations, what to ignore); about one page. What each variable means
    belongs in the codebook. The standard transcription rules are
    added at the end (``rules=True``): transcribe exactly as written, mark
    doubtful readings with ``?``, leave illegible things empty, never invent,
    use only what is on this image.

    Parameters
    ----------
    file : path, optional
        A ``.docx``, ``.txt``, ``.md`` or ``.tex`` file.
    text : str, optional
        The instructions as text, instead of a file.
    rules : bool or str
        True (default) adds the standard rules, False sends exactly your
        text, or the id of an older version of the rules (``"rules-1"``).
    """
    file = user_path(file)
    if file is not None and text is not None:
        raise InputError("Give either `file` or `text`, not both.")
    # instructions("Transcribe every row ...") : text given where a file was
    # expected. Only a string that cannot be a path is taken for text: one
    # with a / or \ or an extension is a path with a typo, and sending the
    # path as the prompt cost a paid redo once fixed
    if isinstance(file, str) and not os.path.exists(file) and re.search(r"\s", file) and not looks_like_path(file):
        info("Using this text as the instructions (no file with that name exists).")
        text, file = file, None
    if rules is True:
        rules_id = RULES_ID
    elif rules is False:
        rules_id = None
    elif isinstance(rules, str) and rules in RULES_TEXTS:
        rules_id = rules
    else:
        raise InputError(f"`rules` must be True, False or one of {', '.join(repr(r) for r in RULES_TEXTS)}.")
    if file is not None:
        if not isinstance(file, (str, os.PathLike)):
            raise InputError("`file` must be a path to a .docx, .txt, .md or .tex file.")
        body = read_instructions_file(file)
        source = str(file)
    elif text is not None:
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(t) for t in text)
        if not isinstance(text, str):
            raise InputError("`text` must be text.")
        body = re.sub(r"\r\n?", "\n", text)
        source = "text"
    else:
        body = DEFAULT_INSTRUCTIONS
        source = "default"
    body = nfc(re.sub(r"\s+$", "", body))
    if not body:
        raise InputError("The instructions are empty.")
    full = body + "\n\n" + RULES_TEXTS[rules_id] if rules_id else body
    out = Instructions(text=body, rules=rules_id is not None, rules_id=rules_id or "none", full=full,
                       source=source, sha12=sha12(full))
    _check_instructions(out)
    return out


def looks_like_path(s: str) -> bool:
    """A folder separator, or an extension made of letters (".md", ".docx")."""
    return "/" in s or "\\" in s or re.search(r"\.[A-Za-z][A-Za-z0-9]{0,4}$", s.strip()) is not None


def _path_hints(p: str) -> list[str]:
    """What is probably wrong with a path that does not exist."""
    out = []
    if p != p.strip() and os.path.exists(p.strip()):
        out.append(f"The name starts or ends with a space; without it the file exists: {p.strip()!r}.")
    if "\\ " in p and os.path.exists(p.replace("\\ ", " ")):
        out.append("Leave out the backslashes the terminal puts before spaces: "
                   f"{p.replace(chr(92) + ' ', ' ')!r} exists.")
    if not ext(p):
        found = [f"{p}.{e}" for e in _FILE_EXTS if os.path.exists(f"{p}.{e}")]
        if found:
            out.append(f"Did you mean {found[0]!r}? The extension is part of the name.")
    return out


def as_instructions(x) -> Instructions:
    """What transcribe() and benchmark() accept: an Instructions, a file, or
    None (the default). A string is always a file: text goes through
    instructions(text=...)."""
    if x is None:
        info("No instructions given: using the package's general transcription instructions (instructions()).")
        return instructions()
    if isinstance(x, Instructions):
        return x
    x = user_path(x)
    if isinstance(x, (str, os.PathLike)):
        if not os.path.exists(x):
            raise NotFoundError(f"Cannot find the instructions file {show_path(x)}.", *_path_hints(str(x)),
                                f"Relative paths start from your working folder, {os.getcwd()}.",
                                "To give the instructions as text, use instructions(text='...').")
        return instructions(x)
    raise InputError("`instructions` must come from instructions(), or be a file path.")


def read_instructions_file(file) -> str:
    if not os.path.exists(file):
        raise NotFoundError(f"Cannot find the instructions file {show_path(file)}.", *_path_hints(str(file)),
                            f"Check the spelling. Relative paths start from your working folder, {os.getcwd()}.",
                            "To give the instructions as text, use instructions(text='...').")
    e = ext(file)
    if e in _TEXT_EXTS:
        return read_text(file)
    if e == "docx":
        return read_docx_text(file)
    if e == "doc":
        raise InputError("Old Word .doc files cannot be read. Open it in Word and save it as .docx.")
    if e == "rtf":
        raise InputError("RTF files cannot be read. Save the file as .docx or plain text (.txt).")
    raise InputError("Instructions must be a .docx, .txt, .md or .tex file.",
                     ("x", f"{os.path.basename(str(file))} has the extension {e!r}."))


def xml_unescape(x: str) -> str:
    x = x.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
    x = re.sub(r"&#x([0-9A-Fa-f]+);", lambda m: chr(int(m.group(1), 16)), x)
    x = re.sub(r"&#([0-9]+);", lambda m: chr(int(m.group(1))), x)
    return x.replace("&amp;", "&")


_VANISH_RE = re.compile(r'<w:vanish(?:\s+w:val="(?:true|1|on)")?\s*/>')
_MOVE_FROM_RE = re.compile(r"(?s)<w:moveFrom\b[^>]*(?<!/)>.*?</w:moveFrom>")
_RUN_RE = re.compile(r"(?s)<w:r\b[^>]*(?<!/)>.*?</w:r>")
_INNER_TABLE_RE = re.compile(r"(?s)<w:tbl\b(?:(?!<w:tbl\b).)*?</w:tbl>")


def _paragraph_texts(xml: str) -> list[str]:
    """The text of each paragraph (each "</w:p>"): its w:t runs, with tabs and
    line breaks; deleted text (w:delText) and field codes (w:instrText) are
    not w:t, so they are left out."""
    out = []
    for p in xml.split("</w:p>")[:-1]:
        p = p.replace("<w:tab/>", "<w:t>\t</w:t>")
        p = re.sub(r"<w:br/>|<w:br [^>]*/>|<w:cr/>", "<w:t>\n</w:t>", p)
        runs = re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p)
        out.append(xml_unescape("".join(runs)))
    return out


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table_as_rows(m) -> str:
    """One innermost table as plain paragraphs, one per row: the text of each
    cell (its paragraphs joined by a space), cells separated by " | ", the way
    the package asks the model to write a table."""
    rows = []
    for tr in re.findall(r"(?s)<w:tr\b.*?</w:tr>", m.group(0)):
        cells = [" ".join(t for t in _paragraph_texts(tc) if t.strip()) for tc in re.findall(r"(?s)<w:tc\b.*?</w:tc>", tr)]
        line = " | ".join(cells) if any(c.strip() for c in cells) else ""
        rows.append(f"<w:p><w:r><w:t>{_xml_escape(line)}</w:t></w:r></w:p>")
    return "".join(rows)


def read_docx_text(path) -> str:
    """The text of a .docx as Word shows it: the w:t runs of each paragraph,
    with tabs and line breaks. Left out, as Word does not show them: deleted
    text (w:delText), field codes (w:instrText), the old place of a tracked
    move (w:moveFrom) and hidden text (w:vanish). A table becomes one line
    per row, its cells separated by " | ". Footnotes and endnotes are not
    read, and a warning says so. Documents with none of these read exactly as
    before, so their contract ids are unchanged."""
    name = os.path.basename(str(path))
    try:
        with zipfile.ZipFile(path) as z:
            x = z.read("word/document.xml").decode("utf-8")
    except (zipfile.BadZipFile, KeyError, OSError, UnicodeDecodeError):
        raise InputError(f"{name} could not be opened as a Word document.") from None
    if "<w:numPr>" in x:
        warn(f"{name} uses automatic numbering or bullets; the numbers and bullets themselves are not read.",
             "If they matter, type them as plain text.")
    if "w:txbxContent" in x:
        warn(f"{name} has text boxes; check that their text appears in the printed instructions.")
    if "<w:footnoteReference" in x or "<w:endnoteReference" in x:
        warn(f"{name} has footnotes or endnotes, which are not read.",
             "If the model should see them, put their text in the body of the document.")
    body = re.sub(r"(?s)^.*?<w:body>", "", x, count=1)
    if "<w:moveFrom" in body:
        body = _MOVE_FROM_RE.sub("", body)
    if "<w:vanish" in body:
        body = _RUN_RE.sub(lambda m: "" if _VANISH_RE.search(m.group(0).split("</w:rPr>")[0]) and "<w:rPr" in m.group(0) else m.group(0), body)
    while "<w:tbl" in body:  # innermost tables first: a table in a cell becomes part of that cell's text
        new = _INNER_TABLE_RE.sub(_table_as_rows, body)
        if new == body:
            break
        body = new
    return "\n".join(_paragraph_texts(body))


# An unfilled placeholder: text in angle brackets with a letter in it, up to
# 200 characters. The bound must clear every slot instructions_template()
# writes (the longest is 88); the brackets must hug the text, so "ages < 20
# and > 60" is not taken for one, and short HTML-like tags (<b>) are left alone.
PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>\n]{0,198}[^<>\s]>")


def _check_instructions(x: Instructions) -> None:
    slots = [s for s in PLACEHOLDER_RE.findall(x.text)
             if re.search(r"[^\W\d_]", s) and not re.fullmatch(r"</?[a-z]+>", s)]
    if slots:
        warn(f"The instructions still contain {len(slots)} placeholder{'s' if len(slots) != 1 else ''} to fill in, such as {slots[0]!r}.",
             "Replace the text in angle brackets with what is true of your source.")
    words = len(x.text.split())
    if words > 1200:
        warn(f"The instructions are long ({words:,} words).",
             "About one page works best. Put what each variable means in the codebook descriptions instead.")


TEMPLATE_LINES = [
    "# Role",
    "You are transcribing <what the source is: title, place, date, archive and shelfmark>.",
    "Each image shows <one page / a two-page spread / one card> of this source.",
    "",
    "# Task",
    "For each image, return <what you want: for example, one record for every person listed on the page>, following the codebook.",
    "",
    "# Details",
    "- Reading order: <for example: two columns; read the left column from top to bottom, then the right column>.",
    "- Ignore: <for example: the printed page header, library stamps, the ruler at the bottom>.",
    "- Notation: <for example: \"id.\" means the same as the line above; transcribe \"id.\" as printed>.",
    "- <anything else the model should know about how this source is written>",
    "",
    "(The standard transcription rules - transcribe exactly, mark doubt with ?, leave illegible things empty,",
    "never invent - are added automatically by instructions(). Delete this note.)",
]


def instructions_template(file, overwrite: bool = False) -> Path:
    """Write a starter file for the instructions (``.docx``, ``.md`` or ``.txt``).

    The parts to fill in are in angle brackets, e.g. ``<title and date of
    the source>``. Replace them, save, and read the file with
    :func:`instructions`. Returns the path.
    """
    from ._docx import write_docx

    if not isinstance(file, (str, os.PathLike)):
        raise InputError("`file` must be a file name, e.g. 'instructions.docx'.")
    file = user_path(file)
    if os.path.exists(file) and not overwrite:
        raise InputError(f"{file} already exists.", "Choose another name, or use overwrite=True.")
    e = ext(file)
    if e == "docx":
        blocks = [{"type": "heading" if l.startswith("# ") else "para", "text": l[2:] if l.startswith("# ") else l}
                  for l in TEMPLATE_LINES]
        write_docx(blocks, file)
    elif e in ("md", "txt", "markdown"):
        write_text(TEMPLATE_LINES + [""], file)
    else:
        raise InputError("The template must be a .docx, .md or .txt file.")
    done(f"Wrote {show_path(file)}.")
    info(f"Fill in the parts in <angle brackets>, save, then use instructions({os.path.basename(str(file))!r}).")
    return Path(file)

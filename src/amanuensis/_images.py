"""images(): every source goes through here, even a folder of JPEGs, so that
each page the model will see is located, checked and listed before any money
is spent. The rule for rendering: rasterise only - no deskew, no contrast, no
cropping. The only changes made to an image are those needed to send it:
decode, apply the camera's rotation tag, convert to sRGB, encode as JPEG.

PDFs are rendered with pypdfium2 (PDFium) one page at a time; multi-page TIFFs
are read one frame at a time, never whole (a 30-frame archive TIFF is 560 MB
once decoded). A page that renders completely uniform is flagged: real scans
never are, and a broken renderer once made every page white.

The folder of page images gets an index, images.csv (file, source,
source_page, width, height, dpi, source_bytes), so a later session that passes
only the folder still knows which original file and page each image came
from. Page numbers are never parsed out of file names. Two sources with the
same name (IMG_0001.JPG from two camera cards, padro.pdf in a folder per year)
never share a page image: see ``_plan_outputs()``.

Unlike the R package, the default folder is created without asking: by
default everything the package makes is saved.
"""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tempfile

import pandas as pd

from ._errors import InputError, NotFoundError
from ._say import Progress, done, info, plural, say, warn
from ._text import (as_number, ext, natural_order, page_key, read_delim_text, safe_stem, show_path, stem, user_path,
                    write_bytes_atomic, write_csv)

IMAGE_EXTS = ("pdf", "tif", "tiff", "heic", "heif", "png", "webp", "jp2", "j2k", "jpx", "jpg", "jpeg", "bmp", "gif")
SEND_EXTS = ("jpg", "jpeg", "png", "webp", "heic", "heif")
# Google accepts at most 20 MB per request, and base64 inflates by 4/3.
MAX_JPEG_BYTES = 14 * 1024 ** 2
# Above this a page is rendered at a lower dpi: an A0 sheet at 600 dpi is
# 2 GB of pixels, and Google downscales it anyway.
MAX_RENDER_PIXELS = 60e6
STATUS_ORDER = ("rendered", "converted", "copied", "in place", "already there")
# Pillow refuses images above about 179 megapixels as a possible
# "decompression bomb" (a small file that unpacks to gigabytes). The files
# images() reads are the user's own scans (an A0 map at 400 dpi, an archive's
# TIFF master), so while it reads them the limit is LARGEST_PIXELS, and an
# image above Pillow's own limit is reduced to TARGET_PIXELS before encoding.
LARGEST_PIXELS = 1_000_000_000
TARGET_PIXELS = 50e6
_REDUCE_ABOVE = None  # Pillow's own limit, while images() runs


class Images(pd.DataFrame):
    """The page images: one row per page (``image``, ``file``, ``source``,
    ``source_page``, ``width``, ``height``, ``status``, ``note``), with the
    folder they are in as ``.folder``. Pass it to :func:`transcribe`."""

    _metadata = ["folder", "dpi"]

    @property
    def _constructor(self):
        return Images

    def __repr__(self) -> str:
        if "status" not in self.columns:
            return pd.DataFrame.__repr__(self)
        ok = self["status"] != "failed"
        head = f"Page images: {int(ok.sum())} ready"
        if (~ok).any():
            head += f", {int((~ok).sum())} failed"
        head += f" | folder: {getattr(self, 'folder', None) or '-'}"
        cols = [c for c in ("file", "source", "source_page", "width", "height", "status", "note") if c in self.columns]
        show = pd.DataFrame(self[cols])
        if "note" in show:
            show["note"] = [n[:37] + "..." if isinstance(n, str) and len(n) > 40 else n for n in show["note"]]
        body = show.head(10).to_string(index=False)
        more = f"\n... and {len(show) - 10} more pages" if len(show) > 10 else ""
        return head + "\n" + body + more


def images(source, dpi: int = 200, save_to=None, pages=None, quality: int = 85, overwrite: bool = False) -> Images:
    """Prepare page images: one checked JPEG per page.

    PDFs are rendered page by page, multi-page TIFFs split into pages, and
    HEIC (iPhone), PNG, WEBP, JPEG 2000, BMP and GIF images converted. JPEGs
    that are already fine are used as they are. Every file is checked, so a
    damaged or misnamed file is reported now rather than halfway through a
    paid run. Images are not altered beyond what is needed to send them;
    photos are turned upright by the camera's rotation tag. Running the same
    line again only does what is missing.

    ``dpi`` only matters for PDFs: 200 suits most manuscripts and ordinary
    print, 250-300 small or worn type (the function says when the scans
    inside a PDF are coarser than the dpi asked for).

    Parameters
    ----------
    source : path or list of paths
        A PDF or image file, several files, or a folder (the files directly
        inside it, in natural order: p2 before p10).
    dpi : int
        Resolution for rendering PDF pages. Default 200.
    save_to : path, optional
        Folder for the page images. Default: a new folder next to the source
        (``<folder>/pages`` for a folder, ``<name>_pages`` for one file). A
        folder of JPEGs that need no conversion is used where it is.
    pages : iterable of int, optional
        For PDFs and multi-page TIFFs, the page numbers to prepare, counted
        from 1 as in a PDF viewer (e.g. ``range(1, 6)``). Default: all.
    quality : int
        JPEG quality, 1-100. Default 85.
    overwrite : bool
        Redo page images that already exist (after changing ``dpi``).
    """
    if source is None or (isinstance(source, (list, tuple)) and not source):
        raise InputError("Tell images() where your scans are.",
                         "For example images('scans/book.pdf'), or images('scans') for a folder.")
    sources = [str(user_path(s)) for s in source] if isinstance(source, (list, tuple)) else [str(user_path(source))]
    save_to = user_path(save_to)
    _check_num(dpi, "dpi", 30)
    _check_num(quality, "quality", 1)
    if pages is not None:
        try:
            pages = [int(p) for p in pages]
        except (TypeError, ValueError):
            raise InputError("`pages` must be page numbers, e.g. range(1, 6).") from None
    if dpi > 600:
        warn(f"dpi={dpi} is very high: pages will be large and slow, and Google reduces them anyway. 200-300 is usually enough.")
    inputs = collect_inputs(sources)
    in_place = (save_to is None and all(k == "jpeg" for _, k in inputs)
                and len({os.path.dirname(show_path(p)) for p, _ in inputs}) == 1
                and all(jpeg_ready(p) for p, _ in inputs))
    if in_place:
        save_to = os.path.dirname(show_path(inputs[0][0]))
    elif save_to is None:
        first = inputs[0][0]
        if len(sources) == 1 and os.path.isdir(sources[0]):
            save_to = os.path.join(sources[0], "pages")
        elif len(inputs) == 1:
            save_to = os.path.join(os.path.dirname(first), f"{stem(first)}_pages")
        else:
            save_to = os.path.join(os.path.dirname(first), "pages")
    if not isinstance(save_to, (str, os.PathLike)):
        raise InputError("`save_to` must be a folder path.")
    refuse_library_path(save_to)
    os.makedirs(save_to, exist_ok=True)
    folder = show_path(save_to)
    if in_place:
        info(f"The JPEGs in {folder} are ready to use as they are; nothing is copied.")
    else:
        info(f"Page images go in {folder}.")
    plan = _plan_outputs(inputs, folder)
    rows = []
    bar = Progress("Preparing pages", len(inputs))
    large = _large_images()
    large.__enter__()
    try:
        for i, (path, kind) in enumerate(inputs):
            try:
                if kind == "pdf":
                    r = _render_pdf(path, folder, dpi, pages, quality, overwrite, plan[i]["stem"])
                elif kind == "tiff":
                    r = _convert_tiff(path, folder, plan[i]["name"], pages, quality, overwrite, plan[i]["stem"])
                elif kind == "jpeg":
                    r = [_take_jpeg(path, folder, plan[i]["name"], quality, overwrite)]
                else:
                    r = [_convert_image(path, folder, plan[i]["name"], quality, overwrite)]
            except KeyboardInterrupt:
                raise
            except Exception as e:  # a damaged file is listed, never dropped
                r = [_page_row(None, path, None, status="failed", note=f"could not be read: {_too_large(e) or _why(e)}")]
            size = os.path.getsize(path)
            for row in r:
                row["source"] = plan[i]["source"]
                if plan[i]["note"]:
                    row["note"] = f"{row['note']}; {plan[i]['note']}" if row["note"] else plan[i]["note"]
                row["source_bytes"] = size
            rows.extend(r)
            bar.update()
    finally:
        large.__exit__(None, None, None)
        bar.close()
    if not in_place:
        _write_image_index(rows, folder, dpi)
    out = Images(rows, columns=["image", "file", "source", "source_page", "width", "height", "status", "note"])
    for c in ("source_page", "width", "height"):
        out[c] = out[c].astype("Int64")
    out.folder = folder
    # the dpi the PDF pages really have
    used = {r["dpi"] for r in rows if r.get("dpi") is not None}
    out.dpi = used.pop() if len(used) == 1 else (None if used else dpi)
    _report_images(out)
    return out


def _why(e: Exception) -> str:
    s = str(e) or type(e).__name__
    return s.split("\n")[0]


class _large_images:
    """While images() reads the user's scans: Pillow's size limit raised to
    LARGEST_PIXELS (its warning between the two silenced), and Pillow's own
    limit kept in _REDUCE_ABOVE for _write_jpeg."""

    def __enter__(self):
        global _REDUCE_ABOVE
        import warnings

        from PIL import Image

        self.old = Image.MAX_IMAGE_PIXELS
        _REDUCE_ABOVE = self.old
        if self.old is not None:
            Image.MAX_IMAGE_PIXELS = max(LARGEST_PIXELS, self.old)
        self.w = warnings.catch_warnings()
        self.w.__enter__()
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return self

    def __exit__(self, *exc):
        global _REDUCE_ABOVE
        from PIL import Image

        self.w.__exit__(*exc)
        Image.MAX_IMAGE_PIXELS = self.old
        _REDUCE_ABOVE = None
        return False


def _too_large(e: Exception) -> str | None:
    """Pillow's decompression-bomb refusal, in plain words."""
    try:
        from PIL import Image
        if isinstance(e, Image.DecompressionBombError):
            return (f"the image is larger than {LARGEST_PIXELS / 1e6:,.0f} megapixels, too large to convert here: reduce it to "
                    f"at most {TARGET_PIXELS / 1e6:.0f} megapixels in an image program first")
    except ImportError:  # pragma: no cover
        pass
    return None


def _check_num(x, arg, minimum):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x < minimum:
        raise InputError(f"`{arg}` must be a single number of at least {minimum}.")


def refuse_library_path(path, arg: str = "save_to") -> None:
    """Never write inside Python's package folders (they are replaced on updates)."""
    p = show_path(path)
    roots = set()
    for key in ("purelib", "platlib", "stdlib", "platstdlib"):
        try:
            roots.add(show_path(sysconfig.get_paths()[key]))
        except KeyError:  # pragma: no cover
            pass
    roots.add(show_path(os.path.dirname(os.path.abspath(__file__))))
    for r in roots:
        if p == r or p.startswith(r + os.sep):
            raise InputError(f"{p} is inside Python's package folders.", f"Choose a folder of your own with `{arg}`.")


def collect_inputs(sources: list[str]) -> list[tuple[str, str]]:
    """Folders expanded, existence and formats checked: [(path, kind)]."""
    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise NotFoundError(f"Cannot find {', '.join(show_path(m) for m in missing)}.",
                            f"Check the spelling. Relative paths start from your working folder, {os.getcwd()}.",
                            "Use forward slashes / in paths, also on Windows.")
    paths, skipped = [], []
    for s in sources:
        if os.path.isdir(s):
            names = [n for n in os.listdir(s) if not os.path.isdir(os.path.join(s, n))
                     and not n.startswith(".") and not n.startswith("~$")]
            ok = [n for n in names if ext(n) in IMAGE_EXTS]
            skipped += [n for n in names if ext(n) not in IMAGE_EXTS and n != "images.csv"]
            paths += [os.path.join(s, ok[i]) for i in natural_order(ok)]
        else:
            if ext(s) not in IMAGE_EXTS:
                raise InputError(f"{os.path.basename(s)} is not a PDF or an image file this package reads.",
                                 "Accepted: PDF, TIFF, HEIC, PNG, WEBP, JPEG 2000, JPEG, BMP, GIF.")
            paths.append(s)
    if skipped:
        shown = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        info(f"Skipped {plural(len(skipped), 'file')} that {'is' if len(skipped) == 1 else 'are'} not a PDF or an image: {shown}.")
    if not paths:
        raise InputError(f"No PDF or image files found in {', '.join(sources)}.",
                         "Accepted: PDF, TIFF, HEIC, PNG, WEBP, JPEG 2000, JPEG, BMP, GIF.")
    seen, unique = set(), []
    for p in paths:
        k = show_path(p)
        if k not in seen:
            seen.add(k)
            unique.append(p)

    def kind(p):
        e = ext(p)
        return "pdf" if e == "pdf" else "tiff" if e in ("tif", "tiff") else "jpeg" if e in ("jpg", "jpeg") else "other"

    return [(p, kind(p)) for p in unique]


# ------- cheap checks -----------------------------------------------------------

def jpeg_header(path) -> dict:
    """Is it a JPEG, its size, and its EXIF orientation (None when there is none)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(min(size, 65536))
    is_jpeg = len(head) > 3 and head[:3] == b"\xff\xd8\xff"
    return {"jpeg": is_jpeg, "bytes": size, "orientation": exif_orientation(head) if is_jpeg else None}


def exif_orientation(b: bytes):
    pos = b.find(b"Exif\x00\x00")
    if pos < 0:
        return None
    tiff = pos + 6
    if tiff + 8 > len(b):
        return None
    little = b[tiff:tiff + 2] == b"II"

    def u16(at):
        v = b[at:at + 2]
        return v[0] + 256 * v[1] if little else 256 * v[0] + v[1]

    def u32(at):
        v = b[at:at + 4]
        return int.from_bytes(v, "little" if little else "big")

    ifd = tiff + u32(tiff + 4)
    if ifd + 2 > len(b):
        return None
    n = u16(ifd)
    for k in range(min(n, 200)):
        entry = ifd + 2 + k * 12
        if entry + 12 > len(b):
            break
        if u16(entry) == 0x0112:
            return u16(entry + 8)
    return None


def jpeg_ready(path) -> bool:
    h = jpeg_header(path)
    return h["jpeg"] and h["bytes"] <= MAX_JPEG_BYTES and h["orientation"] in (None, 1)


def jpeg_complete(path) -> bool:
    """A JPEG written by an interrupted run may be truncated: done only if it
    starts with FFD8 and ends with FFD9."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size < 4:
        return False
    with open(path, "rb") as f:
        start = f.read(2)
        f.seek(size - 2)
        end = f.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def same_bytes(a, b) -> bool:
    """Same size and the same first and last 64 kB."""
    try:
        size = os.path.getsize(a)
        if size != os.path.getsize(b):
            return False
    except OSError:
        return False
    k = min(size, 65536)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        if fa.read(k) != fb.read(k):
            return False
        fa.seek(size - k)
        fb.seek(size - k)
        return fa.read(k) == fb.read(k)


def _is_page_of(name_key: str, stem_key: str) -> bool:
    pre = stem_key + "_p"
    return name_key.startswith(pre) and name_key[len(pre):].isdigit()


def _page_name(i: int, n: int, page_stem: str) -> str:
    width = max(3, len(str(n)))
    return f"{page_stem}_p{i:0{width}d}.jpg"


# ------- naming ------------------------------------------------------------------

def _plan_outputs(inputs, folder) -> list[dict]:
    """Output names for every input, planned before anything is written.

    Page images are found, and their results files named, by file name
    ignoring capitals and the extension, so two sources must never share one.
    JPEGs keep their name; other images become <stem>.jpg, or <stem>_<ext>.jpg
    beside a file of the same stem; PDF and TIFF pages become
    <stem>_p001.jpg. Sources with the same name in different folders are
    normal (camera numbering restarts on every card): the first keeps its
    plain name, so pages and results made earlier stay valid; a later one
    gets its folder's name in front (1931_padro_p001.jpg), then a counter.
    The same inputs always get the same names, so a rerun resumes. A name
    already in the folder is taken when images.csv says it was made from
    another file, or when it is another source's original: such a page is
    never reported "already there" nor overwritten.
    """
    paths = [p for p, _ in inputs]
    kinds = [k for _, k in inputs]
    n = len(paths)
    exts = [ext(p) for p in paths]
    stems = [safe_stem(stem(p)) for p in paths]
    norm = [show_path(p) for p in paths]
    in_folder = [os.path.dirname(x) == show_path(folder) for x in norm]
    ready = [kinds[i] == "jpeg" and jpeg_ready(paths[i]) for i in range(n)]
    in_place = [kinds[i] == "jpeg" and in_folder[i] and ready[i] for i in range(n)]
    single = [os.path.basename(paths[i]) if kinds[i] == "jpeg" else f"{stems[i]}.jpg" for i in range(n)]
    for i in range(n):
        if kinds[i] == "jpeg" and in_folder[i] and not ready[i]:
            single[i] = f"{stems[i]}_upright.jpg"
    skey = [page_key(stem(s)) for s in single]
    for i in range(n):
        if kinds[i] in ("other", "tiff"):
            if any(skey[j] == skey[i] and exts[j] != exts[i] and kinds[j] != "pdf" for j in range(n)):
                single[i] = f"{stems[i]}_{exts[i]}.jpg"
    multi = [k in ("pdf", "tiff") for k in kinds]
    has_single = [k != "pdf" for k in kinds]

    existing = [e for e in os.listdir(folder) if ext(e) in IMAGE_EXTS and not e.startswith(".")]
    ekey = [page_key(stem(e)) for e in existing]
    enorm = [show_path(os.path.join(folder, e)) for e in existing]
    try:
        idx = read_image_index(folder)
    except Exception:
        idx = None
    ikey = [] if idx is None else [page_key(stem(f)) for f in idx["file"]]

    def same_source(rows, i):
        out = []
        for r in rows:
            b = as_number(idx["source_bytes"].iloc[r]) if "source_bytes" in idx.columns else math.nan
            same_name = page_key(os.path.basename(str(idx["source"].iloc[r]))) == page_key(os.path.basename(paths[i]))
            out.append(same_name and (math.isnan(b) or b == os.path.getsize(paths[i])))
        return out

    others = set(norm)

    def taken_by_folder(i, name, page_stem):
        hit = [j for j in range(len(existing)) if name is not None and ekey[j] == page_key(stem(name))]
        if page_stem is not None:
            hit += [j for j in range(len(existing)) if _is_page_of(ekey[j], page_key(page_stem))]
        if not hit:
            return False
        if any(enorm[j] in (others - {norm[i]}) for j in hit):
            return True
        hit = [j for j in hit if enorm[j] != norm[i]]
        if not hit:
            return False
        if name is not None and any(ekey[j] == page_key(stem(name)) and existing[j] != name for j in hit):
            return True
        rows = []
        if name is not None:
            rows += [r for r in range(len(ikey)) if ikey[r] == page_key(stem(name))]
        if page_stem is not None:
            rows += [r for r in range(len(ikey)) if _is_page_of(ikey[r], page_key(page_stem))]
        if rows:
            return not all(same_source(rows, i))
        if kinds[i] == "jpeg" and ready[i] and name is not None:
            f = os.path.join(folder, name)
            return os.path.exists(f) and jpeg_complete(f) and not same_bytes(paths[i], f)
        return False

    taken_single, taken_multi = [], []

    def clashes(s, m):
        if s is not None and (s in taken_single or any(_is_page_of(s, t) for t in taken_multi)):
            return True
        if m is not None and (m in taken_multi or any(_is_page_of(t, m) for t in taken_single)):
            return True
        return False

    prefix = [safe_stem(os.path.basename(os.path.dirname(x))) for x in norm]
    names = list(single)
    page_stems = [stems[i] if multi[i] else None for i in range(n)]
    renamed = [False] * n
    order = [i for i in range(n) if in_place[i]] + [i for i in range(n) if not in_place[i]]
    for i in order:
        s_stem, s_ext = stem(single[i]), os.path.splitext(single[i])[1][1:]
        for k in range(1000):
            def base(x):
                return x if k == 0 else f"{prefix[i]}_{x}" if k == 1 else f"{prefix[i]}_{x}_{k}"
            nm = f"{base(s_stem)}.{s_ext}" if has_single[i] else None
            ps = base(stems[i]) if multi[i] else None
            s_key = page_key(stem(nm)) if nm is not None else None
            m_key = page_key(ps) if ps is not None else None
            if in_place[i] or (not clashes(s_key, m_key) and not taken_by_folder(i, nm, ps)):
                break
        names[i] = nm
        page_stems[i] = ps
        renamed[i] = k > 0
        if s_key is not None:
            taken_single.append(s_key)
        if m_key is not None:
            taken_multi.append(m_key)
    bkey = [page_key(os.path.basename(p)) for p in paths]
    out = []
    for i in range(n):
        shared = bkey.count(bkey[i]) > 1 or renamed[i]
        src = f"{os.path.basename(os.path.dirname(norm[i]))}/{os.path.basename(paths[i])}" if shared else os.path.basename(paths[i])
        out.append({"name": names[i], "stem": page_stems[i], "source": src,
                    "note": "named after its folder: another source has the same file name" if renamed[i] else ""})
    return out


# ------- reading and writing pixels --------------------------------------------------

def _register_heif() -> bool:
    try:
        import pillow_heif  # noqa: F401
        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


def _open_image(path):
    """A Pillow image, HEIC included (pillow-heif, else macOS's sips)."""
    from PIL import Image

    if ext(path) in ("heic", "heif") and not _register_heif():
        if sys.platform == "darwin" and shutil.which("sips"):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            try:
                subprocess.run(["sips", "-s", "format", "png", str(path), "--out", tmp.name],
                               check=True, capture_output=True, timeout=300)
                with Image.open(tmp.name) as im:
                    im.load()
                    return im.copy()
            except (subprocess.SubprocessError, OSError):
                raise RuntimeError("macOS could not convert this HEIC photo") from None
            finally:
                os.unlink(tmp.name)
        raise RuntimeError("HEIC photos need the optional package pillow-heif (pip install pillow-heif)")
    with open(path, "rb") as f:
        data = f.read()
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def _to_srgb(img):
    """Upright, 8-bit, RGB or grey, in sRGB: what a JPEG for Google needs."""
    from PIL import Image, ImageCms, ImageOps

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        img = img.point(lambda v: v * (1 / 256)).convert("L")
    elif img.mode == "F":
        img = img.convert("L")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    icc = img.info.get("icc_profile")
    if icc and img.mode in ("RGB", "CMYK"):
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(img, src, dst, outputMode="RGB")
        except Exception:
            img = img.convert("RGB")
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _save_jpeg(img, out_path, quality, dpi=None):
    buf = io.BytesIO()
    kw = {"quality": int(quality)}
    if dpi:
        kw["dpi"] = (int(dpi), int(dpi))
    img.save(buf, "JPEG", **kw)
    write_bytes_atomic(buf.getvalue(), out_path)


def _reduce_huge(img):
    """An image above Pillow's own size limit, reduced to TARGET_PIXELS before
    anything else is done with it; (img, note)."""
    w, h = img.size
    if _REDUCE_ABOVE is None or w * h <= _REDUCE_ABOVE:
        return img, ""
    from PIL import Image

    f = math.sqrt(TARGET_PIXELS / (w * h))
    size = (max(1, int(w * f)), max(1, int(h * f)))
    try:
        small = img.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    except (ValueError, OSError):  # a mode resize cannot take: convert first
        small = _to_srgb(img).resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    return small, (f"a very large scan ({w * h / 1e6:,.0f} megapixels): reduced to {size[0]} x {size[1]} "
                   f"({size[0] * size[1] / 1e6:.0f} megapixels)")


def _write_jpeg(img, out_path, quality, dpi=None):
    """Write a page JPEG, staying under Google's request limit; returns (img, note)."""
    img, huge = _reduce_huge(img)
    img = _to_srgb(img)
    _save_jpeg(img, out_path, quality, dpi)
    note = huge
    if os.path.getsize(out_path) > MAX_JPEG_BYTES:
        _save_jpeg(img, out_path, 70, dpi)
        note = "; ".join(x for x in (huge, "saved at JPEG quality 70 to stay under Google's 20 MB request limit") if x)
        if os.path.getsize(out_path) > MAX_JPEG_BYTES:
            w, h = img.size
            f = math.sqrt(50e6 / (w * h))
            if f < 1:
                img = img.resize((max(1, int(w * f)), max(1, int(h * f))))
            _save_jpeg(img, out_path, 70, dpi)
            note = "; ".join(x for x in (huge, "reduced to 50 megapixels to stay under Google's 20 MB request limit") if x)
    return img, note


def _image_size(path):
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None, None


def _page_row(out_path, path, page, img=None, status="", note="", dpi=None) -> dict:
    w = h = None
    if img is not None:
        w, h = img.size
    elif out_path is not None and os.path.exists(out_path):
        w, h = _image_size(out_path)
    return {"image": show_path(out_path) if out_path else None,
            "file": os.path.basename(out_path) if out_path else None,
            "source": os.path.basename(path), "source_page": page, "width": w, "height": h,
            "status": status, "note": note, "dpi": dpi}


def _is_uniform(img) -> bool:
    ex = img.getextrema()
    if isinstance(ex[0], tuple):
        return all(lo == hi for lo, hi in ex)
    return ex[0] == ex[1]


def pdf_native_dpi(raw, page_sizes=None):
    """Best-effort resolution of the scans inside a PDF: image dictionaries
    are plain text in the file, so find "/Subtype /Image" and read /Width and
    /Height nearby, then divide by the page size in inches. None when unsure.
    ``raw`` is the PDF's bytes or its path; a path is searched through a
    memory map, never read into memory whole."""
    if isinstance(raw, (str, os.PathLike)):
        import mmap

        try:
            with open(raw, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return pdf_native_dpi(mm, page_sizes)
        except (OSError, ValueError):
            return None
    try:
        import itertools

        hits = [m.start() for m in itertools.islice(re.finditer(rb"/Subtype */Image", raw), 40)]
        if not hits:
            return None
        px = []
        for pos in hits:
            chunk = raw[max(0, pos - 400): pos + 400].replace(b"\x00", b" ")
            w = re.search(rb"/Width +([0-9]+)", chunk)
            h = re.search(rb"/Height +([0-9]+)", chunk)
            if w and h:
                px.append(max(int(w.group(1)), int(h.group(1))))
        if not px or not page_sizes:
            return None
        page_in = statistics.median(max(w, h) for w, h in page_sizes) / 72
        m = statistics.median(px)
        if page_in <= 0:
            return None
        dpi = m / page_in
        if dpi < 50 or dpi > 1200:
            return None
        return int(round(dpi / 10) * 10)
    except Exception:
        return None


def _render_pdf(path, folder, dpi, pages, quality, overwrite, page_stem):
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(path)  # read from disk as it is needed, never whole
    except Exception as e:
        raise RuntimeError(f"it is damaged, password-protected, or not really a PDF ({_why(e)})") from None
    try:
        n = len(pdf)
        todo = list(range(1, n + 1)) if pages is None else list(pages)
        out_of_range = [p for p in todo if p < 1 or p > n]
        if out_of_range:
            warn(f"{os.path.basename(path)} has {plural(n, 'page')}; ignoring page number{'s' if len(out_of_range) > 1 else ''} "
                 f"{', '.join(map(str, out_of_range))}.")
            todo = [p for p in todo if 1 <= p <= n]
        sizes = []
        for i in range(n):
            try:
                sizes.append(tuple(pdf[i].get_size()))
            except Exception:
                sizes.append((0, 0))

        def there(i):
            p = os.path.join(folder, _page_name(i, n, page_stem))
            return os.path.exists(p) and not overwrite and jpeg_complete(p)

        if any(not there(i) for i in todo):
            native = pdf_native_dpi(path, sizes)
            if native is not None and dpi > 1.1 * native:
                info(f"The scans inside {os.path.basename(path)} are about {native} dpi; rendering at {dpi} dpi makes "
                     f"bigger files, not sharper pages (dpi={native} would do).")
        # the dpi the pages already there were rendered at, from images.csv
        try:
            idx = read_image_index(folder)
        except Exception:
            idx = None
        old_dpi = {}
        if idx is not None and "dpi" in idx.columns:
            for f, d in zip(idx["file"], idx["dpi"]):
                x = as_number(d)
                if not math.isnan(x):
                    old_dpi[page_key(f)] = int(x)
        rows, blank, kept_other = [], [], []
        for i in todo:
            out_path = os.path.join(folder, _page_name(i, n, page_stem))
            if there(i):
                was = old_dpi.get(page_key(os.path.basename(out_path)))
                if was is not None and was != dpi:
                    kept_other.append(was)
                rows.append(_page_row(out_path, path, i, status="already there", dpi=was))
                continue
            page_dpi, note = dpi, ""
            w_pt, h_pt = sizes[i - 1]
            px = (w_pt / 72 * dpi) * (h_pt / 72 * dpi)
            if math.isfinite(px) and px > MAX_RENDER_PIXELS:
                page_dpi = int(math.floor(dpi * math.sqrt(MAX_RENDER_PIXELS / px)))
                note = f"very large page: rendered at {page_dpi} dpi instead of {dpi}"
            page = pdf[i - 1]
            bitmap = page.render(scale=page_dpi / 72)
            img = bitmap.to_pil()
            if _is_uniform(img):
                blank.append(i)
            img, wnote = _write_jpeg(img, out_path, quality, page_dpi)
            notes = [x for x in (note, wnote, "completely blank" if i in blank else "") if x]
            rows.append(_page_row(out_path, path, i, img=img, status="rendered", note="; ".join(notes), dpi=page_dpi))
            del bitmap, img
    finally:
        pdf.close()
    if kept_other:
        # a new dpi without overwrite=True changes nothing: say so
        was = ", ".join(str(d) for d in sorted(set(kept_other)))
        say(("!", f"{plural(len(kept_other), 'page')} of {os.path.basename(path)} {'was' if len(kept_other) == 1 else 'were'} "
                  f"rendered earlier at {was} dpi and {'is' if len(kept_other) == 1 else 'are'} kept as {'it is' if len(kept_other) == 1 else 'they are'}."),
            ("i", f"To render {'it' if len(kept_other) == 1 else 'them'} at {dpi} dpi, add overwrite=True."))
    if blank:
        warn(f"{plural(len(blank), 'page')} of {os.path.basename(path)} came out completely blank "
             f"(page{'s' if len(blank) > 1 else ''} {', '.join(map(str, blank))}).",
             "If they are not blank in the PDF, check the file in a PDF viewer; run again with overwrite=True after fixing it.")
    return rows


def _convert_tiff(path, folder, single_name, pages, quality, overwrite, page_stem):
    from PIL import Image

    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
    if n <= 1:
        return [_convert_image(path, folder, single_name, quality, overwrite)]
    todo = list(range(1, n + 1)) if pages is None else [p for p in pages if 1 <= p <= n]
    rows = []
    for i in todo:
        out_path = os.path.join(folder, _page_name(i, n, page_stem))
        if os.path.exists(out_path) and not overwrite and jpeg_complete(out_path):
            rows.append(_page_row(out_path, path, i, status="already there"))
            continue
        with Image.open(path) as im:
            im.seek(i - 1)
            frame = im.copy()
        frame, note = _write_jpeg(frame, out_path, quality)
        rows.append(_page_row(out_path, path, i, img=frame, status="converted", note=note))
    return rows


def _convert_image(path, folder, name, quality, overwrite):
    out_path = os.path.join(folder, name)
    if os.path.exists(out_path) and not overwrite and jpeg_complete(out_path):
        return _page_row(out_path, path, 1, status="already there")
    img = _open_image(path)
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img, note = _write_jpeg(img, out_path, quality)
    return _page_row(out_path, path, 1, img=img, status="converted", note=note)


def _take_jpeg(path, folder, name, quality, overwrite):
    """A JPEG that is really a JPEG, upright and small enough is sent as it is
    (copied byte for byte, or used in place); anything else is re-encoded."""
    in_place = show_path(os.path.dirname(show_path(path))) == show_path(folder)
    out_path = os.path.join(folder, name)
    hdr = jpeg_header(path)
    if hdr["jpeg"] and hdr["bytes"] <= MAX_JPEG_BYTES and hdr["orientation"] in (None, 1):
        if in_place:
            return _page_row(path, path, 1, status="in place")
        if os.path.exists(out_path) and not overwrite and same_bytes(path, out_path):
            return _page_row(out_path, path, 1, status="already there")
        with open(path, "rb") as f:
            write_bytes_atomic(f.read(), out_path)
        return _page_row(out_path, path, 1, status="copied")
    if os.path.exists(out_path) and not overwrite and jpeg_complete(out_path):
        return _page_row(out_path, path, 1, status="already there")
    img = _open_image(path)
    why = []
    if not hdr["jpeg"]:
        why.append(f"is really a {img.format or 'different'} file")
    if hdr["orientation"] not in (None, 1):
        why.append("was stored sideways (camera rotation tag)")
    if hdr["bytes"] > MAX_JPEG_BYTES:
        why.append("was over 14 MB")
    img, note = _write_jpeg(img, out_path, quality)
    notes = ["re-encoded: " + ", ".join(why)] + ([note] if note else [])
    return _page_row(out_path, path, 1, img=img, status="converted", note="; ".join(notes))


# ------- the index -------------------------------------------------------------------

def _write_image_index(rows, folder, dpi) -> None:
    path = os.path.join(folder, "images.csv")
    new = [r for r in rows if r["image"] and os.path.dirname(r["image"]) == show_path(folder)]
    if not new:
        return
    cols = ["file", "source", "source_page", "width", "height", "dpi", "source_bytes"]
    # each page's own dpi: a page rendered lower (a very large page) or kept
    # from an earlier call is not indexed at the dpi asked for now
    table = [{"file": r["file"], "source": r["source"], "source_page": r["source_page"], "width": r["width"],
              "height": r["height"], "dpi": r.get("dpi") if str(r["source"]).lower().endswith(".pdf") else None,
              "source_bytes": r.get("source_bytes")} for r in new]
    try:
        old = read_image_index(folder)
    except Exception:
        old = None
    if old is not None:
        keys = {page_key(r["file"]) for r in table}
        for _, o in old.iterrows():
            if page_key(o["file"]) in keys:
                continue
            table.append({c: (o[c] if c in old.columns else None) for c in cols})
    order = natural_order([r["file"] for r in table])
    df = pd.DataFrame([table[i] for i in order], columns=cols)
    df = df.astype(object).where(pd.notna(df), None)
    write_csv(df, path, bom=False)


def read_image_index(folder):
    path = os.path.join(folder, "images.csv")
    if not os.path.exists(path):
        return None
    idx = read_delim_text(path)
    if "file" not in idx.columns:
        return None
    idx = idx.replace({"": None})
    idx["source_page"] = [None if v is None else int(float(v)) for v in idx["source_page"]] if "source_page" in idx else None
    return idx


def _report_images(x: Images) -> None:
    ok = x["status"] != "failed"
    counts = [(s, int((x["status"] == s).sum())) for s in STATUS_ORDER]
    counts = ", ".join(f"{s}: {c}" for s, c in counts if c)
    done(f"{plural(int(ok.sum()), 'page image')} ready ({counts}).")
    if ok.any():
        mp = (x["width"].astype("float") * x["height"].astype("float")) / 1e6
        if mp.notna().any():
            big = mp.idxmax()
            info(f"Largest page: {x['width'][big]} x {x['height'][big]} pixels ({mp[big]:.1f} megapixels).")
    if (~ok).any():
        bad = x[~ok]
        warn(f"{plural(len(bad), 'file')} could not be prepared and will be left out:",
             *[("x", f"{s}: {n}") for s, n in zip(bad["source"], bad["note"])])
    flagged = x[ok & (x["note"].fillna("") != "")]
    if len(flagged):
        say(("!", f"{plural(len(flagged), 'page')} with a note (see the note column), e.g. "
                  f"{flagged['file'].iloc[0]}: {flagged['note'].iloc[0]}"))


# ------- what transcribe() and benchmark() accept ---------------------------------------

def as_images(x) -> Images:
    """images() output, a DataFrame with an ``image`` column, a folder of page
    images, or image paths, as a clean table of pages to send. Raw paths get
    the same cheap checks images() makes (format, size, rotation)."""
    folder = None
    if isinstance(x, pd.DataFrame):
        if "image" not in x.columns:
            raise InputError("`images` must come from images(), or be a folder or image files.")
        folder = getattr(x, "folder", None)
        df = pd.DataFrame(x).copy()
        keep = df["image"].notna()
        if "status" in df.columns:
            keep &= df["status"] != "failed"
        df = df[keep].reset_index(drop=True)
        if "file" not in df.columns:
            df["file"] = [os.path.basename(p) for p in df["image"]]
        if "source" not in df.columns:
            df["source"] = df["file"]
        if "source_page" not in df.columns:
            df["source_page"] = 1
        df = df[["image", "file", "source", "source_page"]]
    elif isinstance(x, (str, os.PathLike, list, tuple)):
        items = [str(user_path(s)) for s in x] if isinstance(x, (list, tuple)) else [str(user_path(x))]
        paths = []
        for s in items:
            if os.path.isdir(s):
                if folder is None:
                    folder = show_path(s)
                names = [n for n in os.listdir(s) if ext(n) in IMAGE_EXTS and not n.startswith(".")
                         and not os.path.isdir(os.path.join(s, n))]
                paths += [os.path.join(s, names[i]) for i in natural_order(names)]
            elif os.path.exists(s):
                paths.append(s)
            else:
                raise NotFoundError(f"Cannot find {show_path(s)}.",
                                    f"Relative paths start from your working folder, {os.getcwd()}.")
        if not paths:
            raise InputError(f"No page images found in {', '.join(items)}.", "Run images() on your scans first.")
        bad = [p for p in paths if ext(p) not in SEND_EXTS]
        if bad:
            raise InputError(f"{', '.join(os.path.basename(b) for b in bad[:5])} must be prepared first "
                             "(Gemini reads JPEG, PNG, WEBP and HEIC).",
                             f"Run pages = images({items[0]!r}) and pass pages.")
        not_ready = [p for p in paths if ext(p) in ("jpg", "jpeg") and not jpeg_ready(p)]
        if not_ready:
            raise InputError(f"{plural(len(not_ready), 'image')} need{'s' if len(not_ready) == 1 else ''} preparing first "
                             f"(sideways, too large, or not really JPEG): {', '.join(os.path.basename(p) for p in not_ready[:3])}.",
                             f"Run pages = images({items[0]!r}) and pass pages.")
        df = pd.DataFrame({"image": [show_path(p) for p in paths], "file": [os.path.basename(p) for p in paths],
                           "source": [os.path.basename(p) for p in paths], "source_page": [1] * len(paths)})
        # recover the original file and page from images.csv when there is one
        for d in sorted({os.path.dirname(p) for p in df["image"]}):
            try:
                idx = read_image_index(d)
            except Exception:
                idx = None
            if idx is None:
                continue
            ikeys = {page_key(f): k for k, f in enumerate(idx["file"])}
            for r in range(len(df)):
                if os.path.dirname(df.at[r, "image"]) != d:
                    continue
                k = ikeys.get(page_key(df.at[r, "file"]))
                if k is not None:
                    df.at[r, "source"] = idx["source"].iloc[k]
                    if idx["source_page"].iloc[k] is not None:
                        df.at[r, "source_page"] = idx["source_page"].iloc[k]
    else:
        raise InputError("`images` must come from images(), or be a folder or image files.")
    if not len(df):
        raise InputError("There are no page images to process.")
    keys = [page_key(safe_stem(stem(f))) for f in df["file"]]
    seen = {}
    for k, img in zip(keys, df["image"]):
        if k in seen:
            raise InputError(f"Two images would share one results file: {seen[k]} and {img}.",
                             "Results are saved by file name, ignoring capitals and the extension, so every page image needs its own name.",
                             "Rename one of them, or prepare the pages again with images() (it names a page after its "
                             "folder when another source has the same file name).")
        seen[k] = img
    missing = [p for p in df["image"] if not os.path.exists(p)]
    if missing:
        raise NotFoundError(f"Page image{'s' if len(missing) > 1 else ''} not found: {', '.join(missing[:5])}.")
    df["source_page"] = [int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 1 for v in df["source_page"]]
    out = Images(df)
    out.folder = folder or os.path.dirname(df["image"].iloc[0])
    return out

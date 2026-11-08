"""example_project(): the bundled example copied into a folder of yours. The
files are copied rather than used where they are installed because every
later step writes next to its input (page images, results, a gold file), and
the package's own folder must never be written to.

The default folder is ``amanuensis-example`` in the working folder, so what
the example makes stays on the computer to look at (the R package uses R's
temporary folder instead). Files already there are kept unless
``overwrite=True``, so a codebook or gold file you edited survives running the
line again. replies.json, which the "example" model reads from the installed
package, and the README listing the planted errors are not copied: the
exercise is to find the errors with score().
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ._errors import InputError
from ._images import refuse_library_path
from ._say import done, info, say
from ._text import show_path, unwritable, user_path

FILES = {"scans": "padro1930.pdf", "instructions": "instructions.md", "codebook": "codebook.xlsx",
         "codebook_csv": "codebook.csv", "gold": "gold.xlsx"}


@dataclass
class ExampleProject:
    """The example's files: folder, scans (the PDF), instructions, codebook,
    codebook_csv and gold (the benchmark)."""

    folder: Path
    scans: Path
    instructions: Path
    codebook: Path
    codebook_csv: Path
    gold: Path


def example_project(save_to=None, overwrite: bool = False) -> ExampleProject:
    """Copy the example project to practise with.

    A made-up 1930 municipal register of three scanned pages, with
    instructions, a codebook and a benchmark (gold standard) transcribed by
    hand. With ``model="example"``, :func:`transcribe` replays prepared
    replies for these pages, so nothing needs a key or costs money. The
    replies are not real Gemini output: they are the benchmark with a few
    typical mistakes planted on purpose (a misread age, a dropped accent, a
    skipped row, a "corrected" spelling and others), for :func:`score` to find.

    Parameters
    ----------
    save_to : path, optional
        Folder to copy the files into (created if needed). Default:
        ``amanuensis-example`` in the working folder.
    overwrite : bool
        Replace example files already in the folder (by default they are kept).
    """
    if not isinstance(overwrite, bool):
        raise InputError("`overwrite` must be True or False.")
    save_to = os.path.join(os.getcwd(), "amanuensis-example") if save_to is None else user_path(save_to)
    if not isinstance(save_to, (str, os.PathLike)):
        raise InputError("`save_to` must be a folder path.")
    src = resources.files("amanuensis").joinpath("data/example")
    if not all(src.joinpath(f).is_file() for f in FILES.values()):
        raise InputError("The example files are missing from the installed package.", "Reinstall amanuensis and try again.")
    if os.path.exists(save_to) and not os.path.isdir(save_to):
        raise InputError(f"{show_path(save_to)} is a file, not a folder.", "Choose a folder with `save_to`.")
    refuse_library_path(save_to)
    try:
        os.makedirs(save_to, exist_ok=True)
    except OSError:
        raise InputError(f"Could not create the folder {show_path(save_to)}.", "Check the path, or choose another folder with `save_to`.") from None
    folder = show_path(save_to)
    kept = []
    for key, name in FILES.items():
        to = os.path.join(folder, name)
        if os.path.exists(to) and not overwrite:
            kept.append(name)
            continue
        try:
            with resources.as_file(src.joinpath(name)) as p:
                shutil.copyfile(p, to)
        except OSError:
            raise unwritable(to) from None
    out = ExampleProject(folder=Path(folder), **{k: Path(folder) / v for k, v in FILES.items()})
    if len(kept) == len(FILES):
        say(("v", f"The example project is already in {folder}; its files were kept."), ("i", "Use overwrite=True for fresh copies."))
    elif kept:
        say(("v", f"Copied the example project to {folder}."),
            ("i", f"Kept {len(kept)} file{'s' if len(kept) > 1 else ''} already there ({', '.join(kept)}); overwrite=True replaces "
                  f"{'it' if len(kept) == 1 else 'them'}."))
    else:
        done(f"Copied the example project to {folder}.")
    say(("i", "A made-up 1930 municipal register of 3 scanned pages, with instructions, a codebook and a benchmark (gold standard) made by hand."),
        ("i", "Try each step, with ex = example_project():"),
        (" ", "pages = images(ex.scans)"),
        (" ", "res = transcribe(pages, ex.instructions, ex.codebook, model=\"example\")"),
        (" ", "output(res, ex.folder / \"padro1930.xlsx\")"),
        (" ", "gold = benchmark(ex.gold)"),
        (" ", "score(gold, res)"),
        ("i", "model=\"example\" needs no key and costs nothing: it replays prepared replies with a few planted mistakes for score() to find."))
    return out

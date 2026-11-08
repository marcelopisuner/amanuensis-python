"""amanuensis: turn photographs and scans of historical documents into data with Gemini.

A vision language model (Google Gemini, through google-genai) reads each page
image and returns records that follow a codebook you write in a
spreadsheet. The workflow:

1. ``export GEMINI_API_KEY="..."`` in the terminal, then :func:`check_setup`.
2. :func:`images` turns PDFs, TIFFs, photos and scans into page images.
3. :func:`instructions` says what the source is; :func:`codebook` lists the
   variables you want.
4. :func:`transcribe` reads every page; each page is saved as it arrives in a
   results folder next to the images, so a stopped run resumes and nothing
   is paid twice (:func:`gemini_models` lists models and prices).
5. :func:`output` saves the results as Excel, CSV, Stata, Word, text, JSON
   or a pickle.
6. :func:`benchmark` and :func:`score` measure accuracy against your own
   transcription of a few pages.

:func:`example_project` copies a small example to practise with, and
``model="example"`` or ``model="mock"`` run everything without a key or cost.
"""

from ._batch import cancel_batch
from ._codebook import Codebook, Variable, codebook, codebook_template
from ._errors import (AmanuensisError, AmanuensisWarning, ApiError, ContractError, GoldEmptyError, InputError,
                      Interrupted, MissingKeyError, NetworkError, NotFoundError)
from ._example import ExampleProject, example_project
from ._gold import Gold, benchmark
from ._images import Images, images
from ._instructions import Instructions, instructions, instructions_template
from ._models import gemini_models
from ._options import options
from ._output import output
from ._result import Result
from ._score import Score, score
from ._setup import check_setup
from ._store import session_reset
from ._transcribe import transcribe
from ._version import __version__

__all__ = [
    "images", "instructions", "codebook", "transcribe", "output", "benchmark", "score", "gemini_models",
    "check_setup", "example_project", "cancel_batch", "instructions_template", "codebook_template",
    "session_reset", "options",
    "Images", "Instructions", "Codebook", "Variable", "Result", "Gold", "Score", "ExampleProject",
    "AmanuensisError", "InputError", "NotFoundError", "MissingKeyError", "ContractError", "GoldEmptyError",
    "ApiError", "NetworkError", "Interrupted", "AmanuensisWarning",
    "__version__",
]

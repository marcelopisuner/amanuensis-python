# The example project

A small, complete project to practise the whole workflow without a key and
without spending anything: `example_project()` copies these files into a
folder of yours, and `transcribe(..., model = "example")` replays prepared
replies for its three pages.

**Everything here is synthetic.** Vilanova de la Riba is a fictional town; the
people, the clerk and his signature are invented. The register was drawn by
`data-raw/make-example.R` and made to look scanned. The replies in
`replies.json` are **hand-made, not Gemini output**: they are the gold standard
with a few realistic errors planted on purpose (listed below), so that
`score()` has something to find.

## Files

| File | What it is |
|---|---|
| `padro1930.pdf` | The source: three scanned-looking sheets ("Full 1" to "Full 3") of a *Padró municipal d'habitants* (municipal register of inhabitants), 1930. One row per person: Núm., Cognoms, Nom, Edat, Sexe (H/D), Estat civil (S/C/V), Professió, Sap llegir (S/N). Grayscale JPEG pages at 200 dpi. `images()` turns it into `padro1930_p001.jpg` … `padro1930_p003.jpg`. |
| `instructions.md` | The instructions for the model (Role, Task, Details). `instructions()` adds the standard transcription rules. |
| `codebook.xlsx` | The codebook: sheet `codebook` (one row per variable: `full`, `num`, `cognoms`, `nom`, `edat`, `sexe`, `estat_civil`, `professio`, `sap_llegir`) and sheet `how_to`. |
| `codebook.csv` | The same codebook as CSV (UTF-8 with BOM); `codebook()` builds the identical schema from either file. |
| `gold.xlsx` | The benchmark (gold standard) for all three pages, in the layout `benchmark()` reads: sheets `benchmark`, `pages` (all done), `about` (seed 1930, method, draw order), `codebook` and `how_to`. |
| `replies.json` | The replies `model = "example"` returns, one per page image: `data` (the JSON a model returns under the codebook's schema) and `tokens` (plausible prompt and output counts for one ultra_high image). Not copied by `example_project()`. |

## What the source tests

Accents (Fàbregas, Antònia) and the middle dot of l·l (Ul·ldemolins,
Marcel·lí); ditto marks `id.` for surnames and occupations, to be copied as
written; an old spelling (Joseph); blank occupations for small children; a
household that runs on from sheet 1 to sheet 2; a double line between two
households on sheet 2; unused rows struck through on sheet 3; a handwritten
certificate at the foot of sheet 3 that is not a record; and one smudged first
name (sheet 2, row 19), which the gold records as a doubtful reading,
`Teresa?`.

## Planted errors

The replies differ from the gold in seven places:

| Page | Row | Núm. | Variable | Gold | Reply | Kind of error |
|---|---|---|---|---|---|---|
| p001 | 6 | 6 | `edat` | 38 | 36 | misread digit |
| p001 | 14 | 14 | `nom` | Antònia | Antonia | dropped accent |
| p002 | 5 | 19 | `nom` | Teresa? | Teresa | doubt mark dropped on the smudged cell |
| p002 | 6 | 20 | whole row | Agustí … | (missing) | row skipped: the reply has 13 rows, so rows 21–28 move up one |
| p003 | 1 | 29 | `nom` | Joseph | Josep | spelling "corrected" (modernised) |
| p003 | 3 | 31 | `estat_civil` | C | V | wrong category |
| p003 | 5 | 33 | `cognoms` | id. | Mas Riera | ditto mark replaced by the surname it stands for |

The replies have 14, 13 and 6 rows where the gold has 14, 14 and 6; the sheet
number (`full`) is right on every page. Paired by position, the skipped row
makes every later row of page 2 look wrong; paired by content, it is one
missing row. A lenient comparison, which ignores a trailing `?`, does not
count the `Teresa?`/`Teresa` difference.

## Rebuilding

From the package folder: `Rscript data-raw/make-example.R`. The script is
deterministic (`set.seed(1930)`, frozen dates): on the same machine a rerun
rebuilds every file byte for byte (the drawing depends on the fonts installed)
and prints each file's md5 with "unchanged" or "CHANGED".

# amanuensis

> [!WARNING]
> **This is a beta version.** amanuensis is still in development and testing.
> It has been tested and is believed to be safe to use, but it may still have
> bugs, and details may change before the first stable release. Start with a
> few pages, check the results, and please
> [report anything that looks wrong](https://github.com/marcelopisuner/amanuensis-python/issues).

**Turn photographs and scans of historical documents into data, from Python.**

An *amanuensis* is a scribe who copies documents for someone else. This
package gives you one: Google's Gemini model reads each page image and fills
in the variables you ask for, in a table you can open in Excel, Stata, R or
Python.

It is not OCR. OCR gives you text, and you then need a program to turn that
text into a table, which fails on the irregular layouts of historical sources.
Here the model reads the page *and* fills in the table at once, following a
codebook you write in a spreadsheet. The package does everything around that:
it prepares the images, sends one page at a time, checks every reply against
your codebook, keeps track of the cost, saves every page as it arrives (so a
run that stops can resume without paying twice), writes the results, and
measures accuracy against your own transcription of a few pages.

There is also an R version, with the same workflow:
[amanuensis for R](https://github.com/marcelopisuner/amanuensis-r).

## Install

You need Python 3.10 or newer (from [python.org](https://www.python.org/downloads/)
or [Anaconda](https://www.anaconda.com/download)) and a terminal: the
**Terminal** app on a Mac, **PowerShell** or the **Anaconda Prompt** on
Windows. Type this in the terminal and press Enter:

```bash
pip install amanuensis
```

If it does not work:

- If the terminal says that `pip` is not found or not recognized, write
  `python3 -m pip` (Mac, Linux) or `py -m pip` (Windows) instead of `pip`,
  followed by the rest of the line.

In a Jupyter notebook you can also install it from a cell:
`%pip install amanuensis`.

Optional: `pip install pillow-heif` reads iPhone photos (HEIC) on Windows and
Linux (a Mac reads them without it), and `pip install rapidfuzz` makes scoring
long free-text pages faster.

To update to the newest version later:

```bash
pip install --upgrade amanuensis
```

## Try it now: the worked example (no key, no cost)

The package comes with a small example to practise with: a made-up 1930
municipal register of three pages, with its instructions, its codebook, a
transcription made by hand, and prepared model replies with a few realistic
mistakes planted in them. `model="example"` replays those replies, so every
step runs without a key and costs nothing.

Start Python by typing `python` in the terminal (on a Mac it may be
`python3`), and paste the blocks below one at a time. In Jupyter, put each
block in a cell of its own. A line with only a name on it, like `pages`, shows
what that name holds.

**1. Copy the example and make its page images.** This creates a folder
called `amanuensis-example` in the folder where you started Python.

```python
import amanuensis as am

ex = am.example_project()      # copies the example's files into amanuensis-example
pages = am.images(ex.scans)    # turns its 3-page PDF into one JPEG image per page
pages
```

**2. Read the instructions and the codebook.** The instructions say what the
source is and how to read it; the codebook lists the variables to fill in for
every person on the register.

```python
prompt = am.instructions(ex.instructions)
cb = am.codebook(ex.codebook)
cb
```

**3. Transcribe the pages.** You get one row per person (33 rows), with the
file and page each row comes from. The package also shows what the pages
would have cost with the default model; nothing is charged.

```python
res = am.transcribe(pages, prompt, cb, model="example")
res
```

**4. Save the table as an Excel file.** Besides the table, `padro1930.xlsx`
has a sheet with every page (its status, tokens and cost), the codebook, and a
sheet about the run (model, settings, instructions, date). `overwrite=True`
lets you run the example again: without it, `output()` never replaces a file
that already exists.

```python
am.output(res, ex.folder / "padro1930.xlsx", overwrite=True)
```

**5. Measure the accuracy** against the transcription made by hand.

```python
gold = am.benchmark(ex.gold)
s = am.score(gold, res)
s
```

You should see:

```text
Accuracy against your benchmark (3 pages, 34 rows, 9 variables; example)
  Rows with every value right          27 of 34   79.4%
  Values right                       261 of 275   94.9%
  Characters right (text)                         96.8%
  Pages with every page value right      3 of 3  100.0%
```

**6. See every mistake, cell by cell.**

```python
s.differences
```

All the planted mistakes are there: a misread age, a dropped accent, a doubt
mark left out, a skipped row, a modernised spelling, a wrong category, and a
ditto mark replaced by the surname it stands for. The full report,
`accuracy.html`, is in `amanuensis-example/padro1930_pages/extraction/`: open
it in your browser.

Everything the example made stays on your computer:

```text
amanuensis-example/
├── padro1930.pdf, instructions.md, codebook.xlsx, codebook.csv, gold.xlsx
│                             the example's files
├── padro1930.xlsx            the table you saved
└── padro1930_pages/          one image per page
    └── extraction/           one result per page, the logs of the run, accuracy.html
```

## Your Gemini key

To read your own pages you need a key for Google's Gemini API. The worked
example does not need one.

**Get a key** at <https://aistudio.google.com/apikey>, with a Google account.
It is a long code that starts with `AIza` (newer keys start with `AQ.`). Keep
it secret, like a password: whoever has it can use it at your expense. Never
write it in a script, a notebook or a shared document; the package never asks
for it and never stores it. (With a university Google account, AI Studio may
be switched off by the administrator: try a personal account.)

**Give it to the terminal** you will start Python from, with your key in place
of `your-key`. On a Mac or Linux:

```bash
export GEMINI_API_KEY="your-key"
```

On Windows, in PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-key"
```

Then start Python (or Jupyter, with `jupyter lab`) **from that same
terminal**. The key lasts until you close the terminal, so set it again each
time you open a new one.

On Windows you can keep the key for every new terminal, and for Jupyter
started from Anaconda Navigator or the Start menu: run
`setx GEMINI_API_KEY "your-key"` once, then open a new terminal. In the
Anaconda Prompt or the Command Prompt (cmd), the command is
`set GEMINI_API_KEY=your-key`, without quotes.

## Your own documents: the workflow

### 1. Make a project folder

Put your scans in a folder of their own, inside a folder for the project:

```text
my-project/
└── scans/
    └── register.pdf
```

PDFs, TIFFs, JPEGs, PNGs, WEBP, JPEG 2000 and iPhone photos (HEIC) all work,
and so does a folder full of photos (`am.images("scans")`). Your scans are
never changed: the package only reads them.

Open a terminal in `my-project`. On a Mac, type `cd ` (with a space), drag the
folder from the Finder into the Terminal window, and press Enter. On Windows,
open the folder in File Explorer, type `powershell` in the address bar and
press Enter. Set your key there and start Python. Python looks for files
starting from that folder, so `"scans/register.pdf"` is found.

### 2. Write the instructions and the codebook

These two files tell the model what to do. Make a starter file of each, then
open them in Word and Excel and fill them in:

```python
import amanuensis as am

am.instructions_template("instructions.docx")
am.codebook_template("codebook.xlsx")
```

- **instructions.docx** (about one page): what the source is, what one page
  shows, what to return, and anything unusual about how it is written
  (abbreviations, ditto marks, what to ignore). Replace the parts in
  `<angle brackets>`. The standard transcription rules (copy exactly as
  written, mark doubtful readings with `?`, never invent) are added for you.
- **codebook.xlsx**: one row per variable you want. `variable` is a short
  name; `type` is text, number, integer, yes/no or category; `description`
  says what to copy, with an example as it is written; `values` lists the
  allowed values of a category, separated by `;`; `level` is `row` (one value
  for each row of the page's table, such as each person) or `page` (one value
  for the whole page, such as a printed page number). The second sheet
  explains each column.

The example's `instructions.md` and `codebook.xlsx`, in `amanuensis-example`,
show what good ones look like.

### 3. Check that everything is ready

```python
am.check_setup()
```

It checks Python, the packages, your key (Google says whether it accepts it,
for free) and your folder. It changes nothing and costs nothing, and every
line marked PROBLEM says how to fix it.

### 4. Read a few pages, then all of them

Start with three pages:

```python
pages = am.images("scans/register.pdf")
prompt = am.instructions("instructions.docx")
cb = am.codebook("codebook.xlsx")

res = am.transcribe(pages, prompt, cb, limit=3)
res
```

Look at the rows, and at `res.pages` (one line per page, with its status and
cost). If something is wrong, improve the instructions or the codebook and run
these lines again: the package notices the change and asks what to do with
the pages already read.

When the first pages look right, read the rest and save the table. The same
line without `limit` sends only the pages not read yet:

```python
res = am.transcribe(pages, prompt, cb)
am.output(res, "register.xlsx")
```

For another format, change the extension: `.csv`, `.dta` (Stata), `.docx`,
`.txt`, `.md`, `.json` or `.pkl`.

If a run stops (you press Ctrl-C, the budget runs out, the connection drops,
the laptop closes), run the same line again: only the pages still missing are
sent.

### 5. Measure the accuracy

The model makes mistakes, so measure how many on your own source: transcribe
a few pages yourself and compare.

```python
gold = am.benchmark("gold.xlsx", pages, cb, n=3, seed=1930)
```

This draws 3 pages at random and opens an editor in your browser: the page
image beside a form made from your codebook. Type what you see on the page,
not what the model wrote. The editor saves `gold.xlsx` after every page, and
only the pages marked finished are scored. To go back to the editor later,
run `am.benchmark("gold.xlsx", pages, cb, app=True)`. If you prefer Excel,
add `app=False` to the first line: you get `gold.xlsx` to fill in, and you
write yes in the `done` column of its `pages` sheet for each finished page.

The `seed` fixes the draw: take it from the source (its year, say) and report
it, so that anyone can check which pages were drawn. For a figure you publish,
transcribe 10 pages or more (`n=10`).

When you have finished, compare:

```python
gold = am.benchmark("gold.xlsx")
s = am.score(gold, res)
s
```

`s.differences` lists every disagreement, cell by cell, and the full report,
`accuracy.html`, is saved in the results folder,
`scans/register_pages/extraction/`.

## What each function does

| Function | What it does |
|---|---|
| `check_setup()` | Checks Python, the packages, the key, the connection to Google and your folder. Changes nothing. |
| `images()` | Turns PDFs, TIFFs, photos and scans into one checked JPEG per page, in a folder next to them. |
| `instructions()` | Reads the instructions for the model from a Word or text file, and adds the standard transcription rules. |
| `codebook()` | Reads the variables you want from a spreadsheet. `codebook(free_text=True)` asks for a plain transcription of each page instead. |
| `transcribe()` | Reads every page, checks every reply, tracks the cost, and saves each page as it arrives. |
| `output()` | Saves the results as Excel, CSV, Stata, Word, text, Markdown, JSON or a Python pickle. |
| `benchmark()` | Draws pages for you to transcribe by hand, and reads your transcription back. |
| `score()` | Compares the model's results with your transcription and writes an HTML report. |
| `gemini_models()` | The Gemini models the package knows, with their prices. |
| `instructions_template()`, `codebook_template()` | Starter files to fill in. |
| `example_project()` | Copies the worked example into a folder of yours. |
| `cancel_batch()` | Cancels a batch job still running at Google. |

Every function has its own help, with all its options: for example
`help(am.transcribe)`.

## Good to know

- **Faithful copies.** The model is told to copy exactly what is written
  (spelling, abbreviations, accents, the document's own mistakes), to mark
  doubtful readings with `?`, to leave out what cannot be read, and never to
  invent. The file, source and page of every row are recorded by the package,
  never by the model.
- **Cost.** Every page's tokens and estimated cost are recorded, and the cost
  so far is shown as the pages arrive. With 25 pages or more, the package
  shows the projected cost after the first two pages and asks before going on.
  `budget=5` stops a run before it is expected to spend more than $5.
  `batch=True` (Google's Batch API: results in minutes or hours) and
  `flex=True` (slower replies) cost half. `am.gemini_models()` lists the
  models and their prices; the default is `gemini-3.8-flash`, and for hard
  handwriting you can try `model="gemini-4-argon"`, which costs more.
- **Changing your mind.** Pages read with other instructions, another
  codebook or another model never mix in one table. When you change one of
  them, the package asks whether to keep the earlier results apart or to read
  the pages again; earlier results are never deleted.
- **Practising on your own scans.** `model="mock"` invents replies in the
  shape of your codebook for any pages (they are not real readings), so you
  can try every step on your own scans without a key and without cost.
- **Small or faded print.** `am.images("scans/register.pdf", dpi=300)` makes
  the page images of a PDF at a higher resolution (the default, 200, suits
  most manuscripts and print).
- **Your pages go to Google.** `transcribe()` sends each page image, with
  your instructions and codebook, to Google's Gemini API (`model="example"`
  and `model="mock"` send nothing). Check that the archive's conditions and
  any data-protection rules allow it.
- **Paths on Windows.** In Python, write paths with forward slashes, on
  Windows too: `am.images("C:/Users/ana/my-project/scans/register.pdf")`.

## For Python users

- The results are pandas DataFrames. `res` has one row per record, starting
  with `file`, `source`, `source_page` and `row`; `res.pages` is the pages
  table and `res.info` describes the run (called, `res.info()` is pandas'
  `DataFrame.info()`). Selections such as `res[res.edat > 30]` keep their
  link to the run, so `output()` still writes their provenance.
- `score()` compares whole pages, so it needs the whole result. To leave pages
  out, select them in the benchmark: `gold[gold.file != "p002.jpg"]`.
- Ctrl-C stops a run (and a script) with `amanuensis.Interrupted`, a
  `KeyboardInterrupt` whose `.result` holds the pages read; `limit=0` returns
  the result so far without sending anything.
- `transcribe(save_to="results")` uses another results folder, and
  `save_to=False` keeps everything in memory. While a run works in a folder it
  holds `transcribe.lock` there, so a second run on the same folder stops
  instead of paying for the same pages.
- The key is read only from `GEMINI_API_KEY`; a `GOOGLE_API_KEY` on the
  computer is never used.

## Problems and questions

Please open an issue at
<https://github.com/marcelopisuner/amanuensis-python/issues>: say what you
ran, what you expected, and the message you got. Never paste your key.

## License

MIT, © 2026 Marcelo Pi-Suñer. See [LICENSE](https://github.com/marcelopisuner/amanuensis-python/blob/main/LICENSE).

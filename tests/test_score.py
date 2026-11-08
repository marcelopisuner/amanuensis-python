import json
import math
import os
import re

import pandas as pd

import amanuensis as am
from amanuensis import _score as SC
from amanuensis._codebook import Codebook, Variable, schema_from_vars
from amanuensis._errors import InputError

from .helpers import FIXTURES, Case, make_jpegs, quiet



def cb_of(*vars_, free_text=False):
    vs = [Variable(*v) if isinstance(v, tuple) else v for v in vars_]
    return Codebook(vars=vs, schema=schema_from_vars(vs), kind="table", free_text=free_text, source="test", sha12="x")


def unit(rows=None, page=None, status=None):
    rows = rows or {}
    n = len(next(iter(rows.values()))) if rows else 0
    u = {"page": page or {}, "rows": rows, "n": n}
    if status:
        u["status"] = status
    return u


def summary(s):
    return {r["measure"]: (r["matched"], r["compared"]) for r in s.summary.to_dict("records")}


NAMES = cb_of(("name", "text", "row"), ("age", "integer", "row"))


class Example(Case):
    def setUp(self):
        super().setUp()
        ex = quiet(am.example_project)
        self.ex = ex
        self.res = quiet(am.transcribe, quiet(am.images, ex.scans), ex.instructions, ex.codebook, model="example")

    def test_every_planted_error_is_caught_in_the_right_variable(self):
        s = quiet(am.score, self.ex.gold, self.res, report=False)
        d = s.differences
        planted = [(f, r, v) for f, r, v in zip(d["file"], d["row_gold"], d["variable"]) if not str(d.loc[d["file"] == f].iloc[0]["note"]).startswith("row")]
        got = {(f, int(r), v) for f, r, v, n in zip(d["file"], d["row_gold"], d["variable"], d["note"]) if n is None or not str(n).startswith("row")}
        self.assertEqual(got, {("padro1930_p001.jpg", 6, "edat"), ("padro1930_p001.jpg", 14, "nom"), ("padro1930_p002.jpg", 5, "nom"),
                               ("padro1930_p003.jpg", 1, "nom"), ("padro1930_p003.jpg", 3, "estat_civil"), ("padro1930_p003.jpg", 5, "cognoms")})
        missing = d[d["note"] == "row missing from the output"]
        self.assertEqual(set(missing["row_gold"]), {6})
        self.assertEqual(s.by_page["alignment"].tolist(), ["position", "content", "position"])
        self.assertEqual(d.loc[(d["row_gold"] == 5) & (d["variable"] == "nom"), "note"].iloc[0], "only a final ?")

    def test_the_figures_are_the_r_packages(self):
        # the R package's own score() on the same example (tests/fixtures/r_score_example.json)
        ref = json.load(open(os.path.join(FIXTURES, "r_score_example.json")))

        def table(df, cols):
            return [[(round(v, 12) if isinstance(v, float) else v) for v in r] for r in df[cols].itertuples(index=False)]

        cols = ["measure", "value", "matched", "compared"]
        for key, kw in (("strict", {}), ("lenient", {"normalize": True}), ("position", {"align": "position"})):
            s = quiet(am.score, self.ex.gold, self.res, report=False, **kw)
            want = [[(round(r[c], 12) if isinstance(r[c], float) else r[c]) for c in cols] for r in ref[key]]
            got = [[a if not isinstance(a, float) else round(a, 12) for a in r] for r in table(s.summary, cols)]
            self.assertEqual([[float(x) if isinstance(x, (int, float)) else x for x in r] for r in got],
                             [[float(x) if isinstance(x, (int, float)) else x for x in r] for r in want], key)
        s = quiet(am.score, self.ex.gold, self.res, report=False)
        bv = s.by_variable
        for r in ref["by_var"]:
            row = bv[bv["variable"] == r["variable"]].iloc[0]
            self.assertEqual((row["compared"], row["differences"]), (r["compared"], r["differences"]), r["variable"])
        unescape = lambda x: None if x is None else re.sub(r"<U\+([0-9A-F]{4})>", lambda m: chr(int(m.group(1), 16)), x)
        want = [(r["file"], r["row_gold"], r["variable"], unescape(r["gold"]), r["output"], r["note"]) for r in ref["diffs"]]
        na = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else x
        got = [(f, None if pd.isna(rg) else int(rg), v, na(g), na(o), na(n)) for f, rg, v, g, o, n in
               zip(s.differences["file"], s.differences["row_gold"], s.differences["variable"], s.differences["gold"],
                   s.differences["output"], s.differences["note"])]
        self.assertEqual(got, want)

    def test_score_prints_in_plain_words_and_writes_its_report_by_default(self):
        s = am.score(self.ex.gold, self.res)
        text = repr(s)
        self.assertIn("Rows with every value right          27 of 34   79.4%", text)
        self.assertIn("1 of the 14 wrong values differ only by a final \"?\"", text)
        self.assertIn("Weakest: nom 88.2% (4 wrong)", text)
        self.assertTrue(os.path.exists(os.path.join(self.res.info.folder, "accuracy.html")))

    def test_the_report_is_one_self_contained_deterministic_html_file(self):
        quiet(am.score, self.ex.gold, self.res, report="a.html", normalize=True)
        quiet(am.score, self.ex.gold, self.res, report="b.html", normalize=True)
        a, b = open("a.html", "rb").read(), open("b.html", "rb").read()
        self.assertEqual(a, b)
        html = a.decode()
        self.assertNotIn("@@", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        self.assertNotIn("<script src", html)
        tables = pd.read_html(__import__("io").StringIO(html)) if False else None
        self.assertIn("Accuracy against your benchmark", html)
        self.assertIn("padro1930_p002.jpg", html)

    def test_score_compares_two_runs_the_first_as_reference(self):
        mock = quiet(am.transcribe, quiet(am.images, self.ex.scans), self.ex.instructions, self.ex.codebook, model="mock", save_to="mockrun")
        s = quiet(am.score, self.res, mock, report=False)
        self.assertIn("Agreement with the first run", repr(s))
        self.assertEqual(s.about["reference"], "run")

    def test_score_reads_a_benchmark_file_through_benchmark(self):
        s = quiet(am.score, str(self.ex.gold), "padro1930_pages/extraction" if False else self.res.info.folder, report=False)
        self.assertEqual(summary(s)["rows"], (27.0, 34.0))

    def test_arguments_are_checked(self):
        gold = quiet(am.benchmark, self.ex.gold)
        with self.assertRaises(InputError):
            am.score(self.ex.gold, gold)
        with self.assertRaises(InputError):
            am.score(self.ex.gold, self.res, align="random")
        with self.assertRaises(InputError):
            am.score(self.ex.gold, self.res, report="r.pdf")
        with self.assertRaises(InputError):
            am.score("nope.xlsx", self.res)


class Pairing(Case):
    def test_position_pairing_cascades_after_a_deleted_row_and_content_pairing_does_not(self):
        g = {"p.jpg": unit({"name": ["Anna", "Bet", "Carme", "Dolors"], "age": [1, 2, 3, 4]})}
        o = {"p.jpg": unit({"name": ["Anna", "Carme", "Dolors"], "age": [1, 3, 4]}, status="ok")}
        pos = SC.score_units(g, o, NAMES, align="position")
        con = SC.score_units(g, o, NAMES, align="content")
        self.assertEqual(summary(pos)["rows"], (1, 4))
        self.assertEqual(summary(con)["rows"], (3, 4))
        self.assertEqual(con.by_page["missing_rows"].tolist(), [1])
        auto = SC.score_units(g, o, NAMES)
        self.assertEqual(auto.by_page["alignment"].tolist(), ["content"])

    def test_content_pairing_leaves_unrelated_rows_unpaired_and_keeps_the_order(self):
        g = {"p.jpg": unit({"name": ["Anna", "Bet"], "age": [1, 2]})}
        o = {"p.jpg": unit({"name": ["Zzzzz", "Anna", "Bet"], "age": [9, 1, 2]}, status="ok")}
        s = SC.score_units(g, o, NAMES)
        self.assertEqual((s.by_page["missing_rows"].iloc[0], s.by_page["extra_rows"].iloc[0]), (0, 1))
        self.assertEqual(summary(s)["rows"], (2, 3))

    def test_rows_n_a_skipped_row_is_one_missing_row_not_a_cascade(self):
        g = {"p.jpg": unit({"name": ["A", "B", "C"], "age": [1, 2, 3]})}
        o = {"p.jpg": unit({"name": ["A", "C", "D", "E", "F"], "age": [1, 3, 4, 5, 6]}, status="ok")}
        s = SC.score_units(g, o, NAMES, rows=3)
        self.assertEqual(s.by_page["missing_rows"].tolist(), [1])
        self.assertEqual(s.by_page["extra_rows"].tolist(), [0])
        self.assertEqual(s.by_page["output_rows"].tolist(), [2])

    def test_rows_n_the_last_row_does_not_pair_with_a_look_alike_far_down(self):
        # the R package's own cases: 2 gold rows, 6 output rows; the second
        # gold row looks like the sixth output row only
        import numpy as np
        sim = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype=float)
        p = SC.dp_align(sim, sim >= 0.5, gap=0.5, open_end=True)
        self.assertEqual(p, [(1, 1), (2, None)])
        sim2 = sim[:, [0, 1, 2, 5, 3, 4]]
        p2 = SC.dp_align(sim2, sim2 >= 0.5, gap=0.5, open_end=True)
        self.assertEqual(p2, [(1, 1), (None, 2), (None, 3), (2, 4)])
        p0 = SC.dp_align(sim, sim >= 0.5)
        self.assertEqual(p0, [(1, 1), (None, 2), (None, 3), (None, 4), (None, 5), (2, 6)])
        # on a page: the model skipped row 10 and row 25 looks like it
        names = [f"Person {i}" for i in range(1, 31)]
        ages = list(range(30, 60))
        g = {"p.jpg": unit({"name": names[:10], "age": ages[:10]})}
        out_names, out_ages = names[:9] + names[10:], ages[:9] + ages[10:]
        out_names[23], out_ages[23] = names[9], ages[9]
        o = {"p.jpg": unit({"name": out_names, "age": out_ages}, status="ok")}
        s = SC.score_units(g, o, NAMES, rows=10)
        self.assertEqual((s.by_page["missing_rows"].iloc[0], s.by_page["extra_rows"].iloc[0]), (1, 0))
        self.assertEqual(summary(s)["rows"][1], 10)


class Values(Case):
    def test_normalize_adds_lenient_figures_beside_the_strict_ones(self):
        g = {"p.jpg": unit({"name": ["Antònia", "Puig?"], "age": [1, 2]})}
        o = {"p.jpg": unit({"name": ["ANTONIA", "Puig"], "age": [1, 2]}, status="ok")}
        s = SC.score_units(g, o, NAMES, normalize=True)
        sm = summary(s)
        self.assertEqual(sm["values"], (2, 4))
        self.assertEqual(sm["values (lenient)"], (4, 4))

    def test_the_lenient_form_folds_case_accents_and_punctuation_but_never_digits(self):
        self.assertEqual(SC.lenient("  Fàbregas,  Pere?"), "fabregas pere")
        self.assertEqual(SC.lenient("1.5"), "1.5")
        self.assertNotEqual(SC.lenient_key(SC.lenient("1.5")), SC.lenient_key(SC.lenient("15")))
        self.assertEqual(SC.lenient_key(SC.lenient("Ul·ldemolins")), SC.lenient_key(SC.lenient("Ulldemolins")))
        self.assertEqual(SC.lenient_key(SC.lenient("Puig-Ferrer")), SC.lenient_key(SC.lenient("PuigFerrer")))
        self.assertNotEqual(SC.lenient_key("1 5"), SC.lenient_key("15"))
        self.assertIsNone(SC.lenient("?"))

    def test_values_are_compared_after_converting_both_sides_to_the_codebook_type(self):
        cb = cb_of(("n", "integer", "row"), ("ok", "boolean", "row"), ("t", "text", "row"))
        g = {"p.jpg": unit({"n": ["034", "x"], "ok": ["S", "no"], "t": [12, True]})}
        o = {"p.jpg": unit({"n": [34, None], "ok": [True, False], "t": ["12", "true"]}, status="ok")}
        s = SC.score_units(g, o, cb)
        self.assertEqual(summary(s)["values"], (5, 6))

    def test_an_empty_cell_matches_an_empty_cell_and_a_missing_row_matches_nothing(self):
        g = {"p.jpg": unit({"name": [None, ""], "age": [None, None]})}
        o = {"p.jpg": unit({"name": [None], "age": [None]}, status="ok")}
        s = SC.score_units(g, o, NAMES, align="position")
        self.assertEqual(summary(s)["values"], (2, 4))

    def test_a_failed_page_counts_as_entirely_wrong_and_a_page_not_run_is_left_out(self):
        g = {"a.jpg": unit({"name": ["A"], "age": [1]}), "b.jpg": unit({"name": ["B"], "age": [2]}), "c.jpg": unit({"name": ["C"], "age": [3]})}
        o = {"a.jpg": unit({"name": ["A"], "age": [1]}, status="ok"), "b.jpg": {"page": {}, "rows": {}, "n": 0, "status": "failed"}}
        s = SC.score_units(g, o, NAMES)
        self.assertEqual(summary(s)["rows"], (1, 2))
        self.assertEqual(s.not_run, ["c.jpg"])
        self.assertIn("not in the results yet", s.notes[0])
        self.assertIn("failed in the output", s.notes[1])
        self.assertEqual(s.differences["note"].tolist(), ["page failed", "page failed"])

    def test_no_benchmark_page_in_the_output_is_a_plain_error(self):
        with self.assertRaises(InputError) as e:
            SC.score_units({"a.jpg": unit({"name": ["A"], "age": [1]})}, {"z.jpg": unit(status="ok")}, NAMES)
        self.assertIn("only=gold", str(e.exception))

    def test_empty_denominators_give_none_never_100_percent(self):
        cb = cb_of(("n", "integer", "row"))
        s = SC.score_units({"a.jpg": unit({"n": [1]})}, {"a.jpg": unit({"n": [1]}, status="ok")}, cb)
        self.assertTrue(pd.isna(s.summary.set_index("measure").loc["characters", "value"]))
        self.assertEqual(SC.pct1(None), "-")

    def test_free_text_is_scored_by_characters_and_words(self):
        cb = am.codebook(free_text=True)
        g = {"a.jpg": unit(page={"text": "one two three four"})}
        o = {"a.jpg": unit(page={"text": "one too three"}, status="ok")}
        s = SC.score_units(g, o, cb)
        sm = summary(s)
        self.assertEqual(sm["words"], (2, 4))
        self.assertEqual(sm["pages"], (0, 1))
        self.assertIn("characters", sm)

    def test_words_are_counted_whole_whatever_their_length(self):
        self.assertEqual(SC.word_totals(["Constantinopolitana one"], ["X one"], [False]), (1, 2))

    def test_a_tables_other_text_is_scored_on_its_own(self):
        cb = cb_of(("name", "text", "row"), ("text", "text", "page"), free_text=True)
        g = {"a.jpg": unit({"name": ["A"]}, page={"text": "Long other text here"})}
        o = {"a.jpg": unit({"name": ["A"]}, page={"text": "Long other text"}, status="ok")}
        s = SC.score_units(g, o, cb)
        sm = summary(s)
        self.assertEqual(sm["values"], (1, 1))
        self.assertIn("other text: words", sm)
        self.assertIn("scored on its own", s.notes[-1])


class Benchmarks(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 2)
        self.cb = quiet(am.codebook, pd.DataFrame({"variable": ["name", "age"], "type": ["text", "integer"]}))
        self.res = quiet(am.transcribe, "pages", codebook=self.cb, model="mock")

    def test_a_frame_benchmark_is_read_like_a_file_and_missing_columns_are_not_compared(self):
        gold = pd.DataFrame({"file": ["p001.jpg", "p001.jpg", "p001.jpg"], "row": [None, None, None], "name": ["name 1", "name 2", "name 3"]})
        s = quiet(am.score, gold, self.res, report=False)
        self.assertEqual(summary(s)["values"], (3, 3))
        self.assertIn("Not compared (only in the output): age.", s.notes)

    def test_a_benchmark_cut_down_after_reading_scores_only_what_is_left(self):
        open("g.csv", "w").write("file,row,name,age\np001.jpg,1,name 1,1\np002.jpg,1,name 1,1\n")
        gold = quiet(am.benchmark, "g.csv", codebook=self.cb)
        s = quiet(am.score, gold[gold["file"] == "p001.jpg"], self.res, report=False)
        self.assertEqual(s.by_page["file"].tolist(), ["p001.jpg"])
        s = quiet(am.score, gold[["file", "row", "name"]], self.res, report=False)
        self.assertEqual(s.by_variable["variable"].tolist(), ["name"])

    def test_a_prefilled_benchmark_is_said_to_be_optimistic(self):
        quiet(am.benchmark, "pre.xlsx", "pages", self.cb, n=2, seed=1, prefill=self.res)
        import openpyxl
        wb = openpyxl.load_workbook("pre.xlsx")
        for r in wb["pages"].iter_rows(min_row=2):
            r[1].value = "yes"
        wb.save("pre.xlsx")
        s = quiet(am.score, "pre.xlsx", self.res, report=False)
        self.assertIn("optimistic", s.notes[0])
        self.assertEqual(summary(s)["rows"][0], summary(s)["rows"][1])

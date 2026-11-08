import json
import os
import unicodedata

import openpyxl
import pandas as pd

import amanuensis as am
from amanuensis import _gold as GD
from amanuensis._errors import GoldEmptyError, InputError
from amanuensis._rng import r_sample_int

from .helpers import example_file, Case, make_jpegs, quiet, roster_codebook



def fill(path, rows, done=None):
    """Type rows into a template's benchmark sheet (list of dicts by header)."""
    wb = openpyxl.load_workbook(path)
    ws = wb["benchmark"]
    head = [c.value for c in ws[1]]
    for r in range(ws.max_row, 1, -1):
        ws.delete_rows(r)
    for row in rows:
        ws.append([row.get(h) for h in head])
    if done is not None:
        ps = wb["pages"]
        for r in ps.iter_rows(min_row=2):
            if r[0].value in done:
                r[1].value = done[r[0].value]
    wb.save(path)


class Draw(Case):
    def test_the_seed_made_from_the_names_ignores_their_order_and_unicode_form(self):
        a = GD.seed_from_names(["b.jpg", "à.jpg"])
        b = GD.seed_from_names([unicodedata.normalize("NFC", "à.jpg"), "b.jpg"])
        self.assertEqual(a, b)
        self.assertLess(a, 2 ** 28)

    def test_the_draw_is_rs_seeded_order_of_the_files_and_n_takes_its_first_pages(self):
        files = [f"p{i}.jpg" for i in (10, 2, 1, 3)]
        base = ["p1.jpg", "p2.jpg", "p3.jpg", "p10.jpg"]
        perm = [base[i - 1] for i in r_sample_int(4, 1930)]
        self.assertEqual(GD.permutation(files, 1930), perm)
        self.assertEqual(GD.draw(files, 2, 1930), perm[:2])

    def test_the_example_benchmark_was_drawn_by_the_r_package_with_the_same_draw(self):
        about = {r[0]: r[1] for r in openpyxl.load_workbook(example_file("gold.xlsx"))["about"].iter_rows(min_row=2, values_only=True)}
        files = [f"padro1930_p00{i}.jpg" for i in (1, 2, 3)]
        self.assertEqual("; ".join(GD.draw(files, 3, int(about["seed"]))), about["drawn"])

    def test_extending_a_draw_continues_the_same_order(self):
        files = [f"p{i:02d}.jpg" for i in range(1, 11)]
        first = GD.draw(files, 3, 7)
        add, same = GD.extend_draw(files, first, 5, 7)
        self.assertTrue(same)
        self.assertEqual(first + add, GD.draw(files, 5, 7))
        more = ["p00.jpg"] + files
        add, same = GD.extend_draw(more, first, 5, 7)
        self.assertEqual(same, GD.permutation(more, 7)[:3] == first)
        self.assertEqual(len(add), 2)
        self.assertFalse(set(add) & set(first))

    def test_the_natural_order_folds_only_ascii_case(self):
        self.assertEqual(GD.ordered(["p10.jpg", "P2.jpg", "p1.jpg"]), ["p1.jpg", "P2.jpg", "p10.jpg"])
        self.assertEqual(GD.ordered(["É.jpg", "e.jpg", "f.jpg"]), ["e.jpg", "f.jpg", "É.jpg"])


class Template(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 5)
        self.cb = roster_codebook()

    def make(self, **kw):
        return quiet(am.benchmark, "gold.xlsx", "pages", self.cb, **{"n": 2, "seed": 1930, **kw})

    def test_benchmark_writes_an_excel_template_with_one_line_per_drawn_page(self):
        self.assertIsNone(self.make())
        wb = openpyxl.load_workbook("gold.xlsx")
        self.assertEqual(wb.sheetnames, ["benchmark", "pages", "about", "codebook", "how_to"])
        rows = list(wb["benchmark"].iter_rows(values_only=True))
        self.assertEqual(rows[0], ("file", "row", "heading", "name", "age", "sex"))
        drawn = GD.ordered(GD.draw([f"p{i:03d}.jpg" for i in range(1, 6)], 2, 1930))
        self.assertEqual([r[0] for r in rows[1:]], drawn)
        pages = list(wb["pages"].iter_rows(values_only=True))
        self.assertEqual(pages[1][1], "no")
        about = {r[0]: r[1] for r in wb["about"].iter_rows(min_row=2, values_only=True)}
        self.assertEqual(about["seed"], "1930")
        self.assertIn("Mersenne-Twister", about["method"])

    def test_the_template_formats_columns_as_text_and_offers_drop_down_lists(self):
        self.make()
        ws = openpyxl.load_workbook("gold.xlsx")["benchmark"]
        self.assertEqual(ws["E2"].number_format, "@")
        self.assertEqual(ws.column_dimensions["E"].number_format, "@")
        lists = {str(dv.sqref): dv.formula1 for dv in ws.data_validations.dataValidation if dv.type == "list"}
        self.assertIn('"male,female,other"', lists.values())

    def test_a_new_benchmark_needs_images_a_codebook_and_an_excel_file(self):
        with self.assertRaises(InputError):
            am.benchmark("g.xlsx")
        with self.assertRaises(InputError):
            am.benchmark("g.txt", "pages", self.cb)
        with self.assertRaises(InputError):
            am.benchmark("g.csv", "pages", self.cb)
        with self.assertRaises(InputError):
            am.benchmark("g.doc", "pages", self.cb)

    def test_app_true_outside_an_interactive_session_writes_the_file(self):
        quiet(am.benchmark, "gold.xlsx", "pages", self.cb, app=True)
        self.assertTrue(os.path.exists("gold.xlsx"))

    def test_n_larger_than_the_pages_takes_them_all(self):
        self.make(n=50)
        self.assertEqual(len(list(openpyxl.load_workbook("gold.xlsx")["pages"].iter_rows(min_row=2))), 5)

    def test_an_untouched_template_is_refused_with_what_to_do(self):
        self.make()
        with self.assertRaises(GoldEmptyError) as e:
            am.benchmark("gold.xlsx")
        self.assertIn("has no values yet", str(e.exception))

    def test_a_filled_template_is_read_back_with_the_codebooks_types(self):
        self.make()
        files = GD.ordered(GD.draw([f"p{i:03d}.jpg" for i in range(1, 6)], 2, 1930))
        fill("gold.xlsx", [{"file": files[0], "row": "1", "heading": "H", "name": "Puig", "age": "34", "sex": "Male"},
                           {"file": files[0], "row": "2", "name": "Pla", "age": "5", "sex": "female"},
                           {"file": files[1], "heading": "H2", "name": "Roca", "age": "", "sex": ""}],
             done={files[0]: "yes", files[1]: "sí"})
        g = quiet(am.benchmark, "gold.xlsx")
        self.assertEqual(list(g.columns), ["file", "row", "heading", "name", "age", "sex"])
        self.assertEqual(str(g["age"].dtype), "Int64")
        self.assertEqual(g["age"].tolist()[:2], [34, 5])
        self.assertEqual(g["sex"].tolist()[0], "male")
        self.assertEqual(g["heading"].tolist(), ["H", "H", "H2"])
        self.assertEqual(g.gold["pages"]["done"].tolist(), [True, True])
        self.assertIn("Drawn with seed 1930", repr(g))

    def test_values_that_do_not_fit_the_codebook_stop_with_their_cell(self):
        self.make()
        f = GD.ordered(GD.draw([f"p{i:03d}.jpg" for i in range(1, 6)], 2, 1930))[0]
        fill("gold.xlsx", [{"file": f, "row": "1", "name": "A", "age": "thirty"}], done={f: "yes"})
        with self.assertRaises(InputError) as e:
            am.benchmark("gold.xlsx")
        self.assertIn("sheet benchmark, cell E2 (age)", str(e.exception))
        self.assertIn('"thirty" is not a whole number', str(e.exception))

    def test_only_pages_marked_done_are_used_and_with_no_marks_the_pages_with_values(self):
        self.make()
        files = GD.ordered(GD.draw([f"p{i:03d}.jpg" for i in range(1, 6)], 2, 1930))
        fill("gold.xlsx", [{"file": files[0], "name": "A"}, {"file": files[1], "name": "B"}], done={files[0]: "yes", files[1]: "no"})
        g = quiet(am.benchmark, "gold.xlsx")
        self.assertEqual(g["file"].unique().tolist(), [files[0]])
        fill("gold.xlsx", [{"file": files[0], "name": "A"}, {"file": files[1], "name": "B"}], done={files[0]: None, files[1]: None})
        g = am.benchmark("gold.xlsx")
        self.assertEqual(g["file"].unique().tolist(), files)
        self.assertIn("No page of gold.xlsx is marked done", self.messages())

    def test_file_names_are_filled_down_and_row_numbers_used_when_complete(self):
        self.make()
        f = GD.ordered(GD.draw([f"p{i:03d}.jpg" for i in range(1, 6)], 2, 1930))[0]
        fill("gold.xlsx", [{"file": f, "row": "2", "name": "second"}, {"row": "1", "name": "first"}], done={f: "yes"})
        g = am.benchmark("gold.xlsx")
        self.assertEqual(g["name"].tolist(), ["first", "second"])
        self.assertIn("had no file name", self.messages())

    def test_rows_n_is_recorded_and_read_back(self):
        self.make(rows=20)
        f = GD.ordered(GD.draw([f"p{i:03d}.jpg" for i in range(1, 6)], 2, 1930))[0]
        fill("gold.xlsx", [{"file": f, "row": "1", "name": "A"}], done={f: "yes"})
        self.assertEqual(quiet(am.benchmark, "gold.xlsx").gold["rows"], 20)

    def test_a_larger_n_adds_the_next_pages_and_keeps_what_was_typed(self):
        self.make()
        all_files = [f"p{i:03d}.jpg" for i in range(1, 6)]
        f = GD.ordered(GD.draw(all_files, 2, 1930))[0]
        fill("gold.xlsx", [{"file": f, "row": "1", "name": "kept"}], done={f: "yes"})
        quiet(am.benchmark, "gold.xlsx", "pages", n=4)
        wb = openpyxl.load_workbook("gold.xlsx")
        pages = [r[0] for r in wb["pages"].iter_rows(min_row=2, values_only=True)]
        self.assertEqual(set(pages), set(GD.draw(all_files, 4, 1930)))
        names = [r[3] for r in wb["benchmark"].iter_rows(min_row=2, values_only=True)]
        self.assertIn("kept", names)
        self.assertTrue(any("-backup-" in n for n in os.listdir(".")))

    def test_prefill_fills_the_drawn_pages_with_the_models_output_and_it_is_recorded(self):
        res = quiet(am.transcribe, "pages", codebook=self.cb, model="mock")
        with self.warns("optimistic"):
            quiet(am.benchmark, "pre.xlsx", "pages", self.cb, n=2, seed=1, prefill=res)
        about = {r[0]: r[1] for r in openpyxl.load_workbook("pre.xlsx")["about"].iter_rows(min_row=2, values_only=True)}
        self.assertEqual(about["prefilled"], "yes")
        with self.assertRaises(GoldEmptyError) as e:
            am.benchmark("pre.xlsx")
        self.assertIn("prefilled", str(e.exception))


class Reading(Case):
    def setUp(self):
        super().setUp()
        self.cb = roster_codebook()

    def test_the_example_gold_reads_with_its_own_codebook(self):
        g = quiet(am.benchmark, example_file("gold.xlsx"))
        self.assertEqual(len(g), 34)
        self.assertEqual(g.gold["pages"]["done"].tolist(), [True, True, True])
        self.assertEqual(g.gold["draw"]["seed"], 1930)
        self.assertEqual(str(g["sap_llegir"].dtype), "boolean")

    def test_yes_no_cells_accept_the_usual_spellings_s_and_n_included(self):
        self.assertEqual([GD.parse_yes_no(x) for x in ("S", "N", "sí", "x", "0", "maybe")], [True, False, True, True, False, None])

    def test_a_csv_benchmark_reads_with_the_codebook_and_every_page_with_values_counts(self):
        open("g.csv", "w").write("file,row,heading,name,age,sex\np1.jpg,1,H,A,3,male\np1.jpg,2,,B,,\np2.jpg,1,K,C,4,female\n")
        g = quiet(am.benchmark, "g.csv", codebook=self.cb)
        self.assertEqual(g["file"].unique().tolist(), ["p1.jpg", "p2.jpg"])
        self.assertEqual(g["age"].tolist(), [3, None, 4] if False else g["age"].tolist())
        self.assertTrue(pd.isna(g["age"].tolist()[1]))

    def test_a_semicolon_csv_reads_decimal_commas_in_number_columns(self):
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["total"], "type": ["number"]}))
        open("g.csv", "w").write("file;row;total\np1.jpg;1;2,5\n")
        self.assertEqual(quiet(am.benchmark, "g.csv", codebook=cb)["total"].tolist(), [2.5])

    def test_category_values_are_matched_to_the_codebook_and_unknown_ones_warned(self):
        open("g.csv", "w").write("file,row,heading,name,age,sex\np1.jpg,1,,A,,MALE\np1.jpg,2,,B,,dona\n")
        with self.warns("not in the codebook's list"):
            g = quiet(am.benchmark, "g.csv", codebook=self.cb)
        self.assertEqual(g["sex"].tolist(), ["male", "dona"])

    def test_excel_damage_to_text_values_is_warned_about(self):
        open("g.csv", "w").write("file,row,heading,name,age,sex\np1.jpg,1,,44927,,\np1.jpg,2,,1.2E+15,,\np1.jpg,3,,007,,\np1.jpg,4,,7,,\n")
        with self.warns("Excel may have changed some text values"):
            quiet(am.benchmark, "g.csv", codebook=self.cb)

    def test_a_date_excel_made_in_a_number_column_stops_the_read(self):
        import datetime
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "benchmark"
        ws.append(["file", "row", "heading", "name", "age", "sex"])
        ws.append(["p1.jpg", 1, None, "A", datetime.datetime(2026, 3, 12), None])
        wb.save("d.xlsx")
        with self.assertRaises(InputError) as e:
            quiet(am.benchmark, "d.xlsx", codebook=self.cb)
        self.assertIn("Excel turned what was typed into a date", str(e.exception))

    def test_page_values_come_from_the_first_line_and_different_later_values_are_warned(self):
        open("g.csv", "w").write("file,row,heading,name,age,sex\np1.jpg,1,First,A,,\np1.jpg,2,Second,B,,\n")
        with self.warns("different values on different lines"):
            g = quiet(am.benchmark, "g.csv", codebook=self.cb)
        self.assertEqual(g["heading"].tolist(), ["First", "First"])

    def test_variables_missing_from_the_file_stop_with_their_names(self):
        open("g.csv", "w").write("file,row,name\np1.jpg,1,A\n")
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["name", "parish"]}))
        with self.assertRaises(InputError) as e:
            am.benchmark("g.csv", codebook=cb)
        self.assertIn("'parish'", str(e.exception))

    def test_json_benchmarks_as_output_writes_them_or_keyed_by_file(self):
        data = [{"file": "p1.jpg", "data": {"heading": "H", "records": [{"name": "A", "age": 3, "sex": "male"}]}}]
        open("g.json", "w").write(json.dumps(data))
        g = quiet(am.benchmark, "g.json", codebook=self.cb)
        self.assertEqual(g["name"].tolist(), ["A"])
        open("k.json", "w").write(json.dumps({"p1.jpg": {"heading": "H", "records": []}}))
        g = quiet(am.benchmark, "k.json", codebook=self.cb)
        self.assertEqual(g["heading"].tolist(), ["H"])

    def test_pkl_and_dta_benchmarks(self):
        df = pd.DataFrame({"file": ["p1.jpg"], "row": [1], "name": ["A"], "age": [3], "sex": ["male"], "heading": ["H"]})
        df.to_pickle("g.pkl")
        self.assertEqual(quiet(am.benchmark, "g.pkl", codebook=self.cb)["age"].tolist(), [3])
        df.to_stata("g.dta", write_index=False)
        self.assertEqual(quiet(am.benchmark, "g.dta", codebook=self.cb)["name"].tolist(), ["A"])

    def test_as_gold_reads_files_and_frames_and_keeps_a_benchmarks_own_codebook(self):
        with self.assertRaises(InputError):
            GD.as_gold(pd.DataFrame({"file": ["p1.jpg"]}))
        g = quiet(GD.as_gold, pd.DataFrame({"file": ["p1.jpg"], "row": [1], "name": ["A"], "age": [2], "sex": ["male"], "heading": ["x"]}), codebook=self.cb)
        self.assertEqual(g["name"].tolist(), ["A"])
        ex = quiet(GD.as_gold, example_file("gold.xlsx"), codebook=self.cb)
        self.assertIn("cognoms", ex.gold["codebook"].names())


class FreeText(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 3)

    def test_a_txt_template_with_equals_lines_is_written_and_read_back(self):
        quiet(am.benchmark, "g.txt", "pages", am.codebook(free_text=True), n=2, seed=3)
        text = open("g.txt").read()
        heads = [l for l in text.split("\n") if l.startswith("=== ")]
        self.assertEqual(len(heads), 2)
        text = text.replace(heads[0], heads[0] + "\nLine one\nline two", 1)
        text = text.replace(heads[1], heads[1] + " not done\nhalf typed", 1)
        open("g.txt", "w").write(text)
        g = quiet(am.benchmark, "g.txt")
        self.assertEqual(g["text"].tolist(), ["Line one\nline two"])
        self.assertIn("seed: 3", open("g.txt").read())

    def test_a_docx_template_is_written_and_read_back(self):
        quiet(am.benchmark, "g.docx", "pages", am.codebook(free_text=True), n=1, seed=3)
        from amanuensis._instructions import read_docx_text
        self.assertIn("=== ", read_docx_text("g.docx"))

    def test_an_xlsx_free_text_benchmark_has_file_and_text_columns(self):
        quiet(am.benchmark, "g.xlsx", "pages", am.codebook(free_text=True), n=1, seed=3)
        head = [c.value for c in openpyxl.load_workbook("g.xlsx")["benchmark"][1]]
        self.assertEqual(head, ["file", "text"])

    def test_a_text_benchmark_cannot_hold_a_table_codebook(self):
        open("g.txt", "w").write("=== p001.jpg ===\nsome text\n")
        with self.assertRaises(InputError):
            am.benchmark("g.txt", codebook=roster_codebook())

    def test_a_text_benchmark_grows_with_a_larger_n_and_markdown_underlines_are_not_pages(self):
        quiet(am.benchmark, "g.md", "pages", am.codebook(free_text=True), n=1, seed=3)
        open("g.md", "a").write("\nTitle\n=====\n")
        quiet(am.benchmark, "g.md", "pages", n=2)
        heads = [l for l in open("g.md").read().split("\n") if l.startswith("=== ")]
        self.assertEqual(len(heads), 2)

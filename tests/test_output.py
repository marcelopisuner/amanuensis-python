import json
import os
import zipfile

import openpyxl
import pandas as pd

import amanuensis as am
from amanuensis import _gemini as G
from amanuensis._errors import InputError
from amanuensis._instructions import read_docx_text
from amanuensis._output import stata_names

from .helpers import Case, FakeGemini, gemini_reply, make_jpegs, quiet, roster_codebook, roster_data, test_instructions


class Base(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 3)
        self.cb = roster_codebook()

    def run_with(self, handler, codebook=None, **kw):
        fake = FakeGemini("pages", handler)
        with G.mocked(fake):
            return quiet(am.transcribe, "pages", test_instructions(), codebook or self.cb, **kw)

    def roster(self, **kw):
        def h(f, b, n, r):
            d = roster_data(f, 2)
            d["records"][0]["name"] = "= id."
            if f == "p003.jpg":
                d["records"][0]["age"] = "old"
            return gemini_reply(data=d)
        return self.run_with(h, **kw)


class Tables(Base):
    def test_without_a_file_output_returns_the_plain_table(self):
        res = self.roster()
        tab = am.output(res)
        self.assertIs(type(tab), pd.DataFrame)
        self.assertEqual(len(tab), 4)
        self.assertEqual(list(am.output(res, table="pages")["file"]), ["p001.jpg", "p002.jpg", "p003.jpg"])
        self.assertEqual(list(am.output(res, table="codebook")["variable"]), ["heading", "name", "age", "sex"])

    def test_include_all_adds_the_pages_to_check_and_refuses_a_selection(self):
        res = self.roster()
        full = am.output(res, include="all")
        self.assertEqual(list(full.columns[:4]), ["file", "source", "source_page", "page_status"])
        self.assertEqual(full["page_status"].tolist(), ["ok", "ok", "ok", "ok", "check", "check"])
        with self.assertRaises(InputError):
            am.output(res[res["row"] == 1], include="all")

    def test_documents_of_a_selection_without_the_file_column_are_refused_before_anything_is_written(self):
        res = self.roster()
        sel = res[["name", "age"]]
        with self.assertRaises(InputError):
            am.output(sel, ["a.csv", "a.txt"])
        self.assertFalse(os.path.exists("a.csv"))
        quiet(am.output, sel, "b.csv")
        self.assertTrue(os.path.exists("b.csv"))

    def test_a_results_folder_gives_the_same_table(self):
        res = self.roster()
        again = quiet(am.output, "pages/extraction")
        pd.testing.assert_frame_equal(pd.DataFrame(am.output(res)), again)

    def test_bad_input_gets_plain_errors(self):
        res = self.roster()
        for f in ("x.xls", "x.rds", "noext", "x.pdf"):
            with self.assertRaises(InputError, msg=f):
                am.output(res, f)
        os.makedirs("adir.csv")
        with self.assertRaises(InputError):
            am.output(res, "adir.csv")
        with self.assertRaises(InputError):
            am.output(res, "pages/extraction/x.json")
        with self.assertRaises(InputError):
            am.output("x.xlsx", res)
        with self.assertRaises(InputError):
            am.output(pd.DataFrame(res), "x.csv")
        with self.assertRaises(InputError):
            am.output(res, "x.csv", table="nope")

    def test_existing_files_are_never_replaced_without_overwrite(self):
        res = self.roster()
        quiet(am.output, res, "a.csv")
        with self.assertRaises(InputError):
            am.output(res, "a.csv")
        quiet(am.output, res, "a.csv", overwrite=True)

    def test_a_missing_folder_is_created(self):
        res = self.roster()
        am.output(res, "new/dir/a.csv")
        self.assertTrue(os.path.exists("new/dir/a.csv"))
        self.assertIn("Created the folder", self.messages())


class Csv(Base):
    def test_csv_is_utf8_with_a_bom_and_sep_and_bom_are_honoured(self):
        res = self.roster()
        quiet(am.output, res, "a.csv")
        data = open("a.csv", "rb").read()
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(data[3:].decode().startswith("file,source,source_page,row,heading,name,age,sex\n"))
        quiet(am.output, res, "b.csv", sep=";", bom=False)
        self.assertTrue(open("b.csv", "rb").read().startswith(b"file;source"))

    def test_sep_semicolon_writes_decimal_commas_only_for_numbers_with_decimals(self):
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["total", "n", "note"], "type": ["number", "integer", "text"]}))
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"records": [{"total": 2.5, "n": 3, "note": "1,5"}]}), codebook=cb)
        am.output(res, "d.csv", sep=";")
        body = open("d.csv", encoding="utf-8-sig").read().splitlines()[1]
        self.assertTrue(body.endswith(";2,5;3;1,5"))
        self.assertIn("decimal comma", self.messages())

    def test_values_excel_would_run_as_formulas_are_kept_and_the_columns_named(self):
        res = self.roster()
        am.output(res, "f.csv")
        self.assertIn('"= id."' if False else "= id.", open("f.csv", encoding="utf-8-sig").read())
        self.assertIn("The column name has values starting with =", self.messages())


class Excel(Base):
    def test_xlsx_has_records_pages_codebook_and_about_sheets_that_read_back(self):
        res = self.roster()
        quiet(am.output, res, "r.xlsx")
        wb = openpyxl.load_workbook("r.xlsx")
        self.assertEqual(wb.sheetnames, ["records", "pages", "codebook", "about"])
        ws = wb["records"]
        self.assertEqual([c.value for c in ws[1]], ["file", "source", "source_page", "row", "heading", "name", "age", "sex"])
        self.assertEqual(ws["F2"].value, "= id.")
        self.assertEqual(ws["F2"].data_type, "s")
        self.assertEqual(ws.freeze_panes, "A2")
        about = {r[0].value: r[1].value for r in wb["about"].iter_rows(min_row=2)}
        self.assertEqual(about["model"], "gemini-3.8-flash")
        self.assertTrue(about["package"].startswith("amanuensis "))
        self.assertIn("1 to check", about["pages"])

    def test_free_text_goes_to_a_text_sheet_and_over_long_cells_are_cut_with_a_warning(self):
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"text": "x" * 40000}), codebook=am.codebook(free_text=True))
        with self.warns("32,767 characters"):
            quiet(am.output, res, "t.xlsx")
        wb = openpyxl.load_workbook("t.xlsx")
        self.assertEqual(wb.sheetnames[0], "text")
        self.assertEqual(len(wb["text"]["D2"].value), 32767)

    def test_a_json_schemas_further_lists_get_sheets_of_their_own(self):
        s = {"type": "object", "properties": {"people": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}},
                                              "notes": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}}}}}
        cb = quiet(am.codebook, json=json.dumps(s))
        data = {"people": [{"name": "A"}, {"name": "B"}], "notes": [{"text": "n1", "tags": ["x", "y"]}]}
        res = self.run_with(lambda f, b, n, r: gemini_reply(data=data), codebook=cb)
        quiet(am.output, res, "j.xlsx")
        wb = openpyxl.load_workbook("j.xlsx")
        self.assertEqual(wb.sheetnames, ["records", "notes", "pages", "codebook", "about"])
        notes = list(wb["notes"].iter_rows(values_only=True))
        self.assertEqual(notes[0], ("file", "source", "source_page", "row", "text", "tags"))
        self.assertEqual(notes[1][4:], ("n1", "x; y"))


class Stata(Base):
    def test_dta_reads_back_with_labels_yes_no_as_0_1_and_accents(self):
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["nom", "sap_llegir", "edat"], "type": ["text", "yes/no", "integer"],
                                              "description": ["Nom.", "Sap llegir.", "Edat."]}))
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"records": [{"nom": "Antònia", "sap_llegir": True, "edat": None},
                                                                             {"nom": "Pere", "sap_llegir": False, "edat": 3}]}), codebook=cb)
        quiet(am.output, res, "s.dta")
        back = pd.read_stata("s.dta", convert_categoricals=False)
        self.assertEqual(back["nom"].tolist()[:2], ["Antònia", "Pere"])
        self.assertEqual(back["sap_llegir"].tolist()[:2], [1, 0])
        with pd.io.stata.StataReader("s.dta") as r:
            labels = r.variable_labels()
            vl = r.value_labels()
        self.assertEqual(labels["sap_llegir"], "Sap llegir.")
        self.assertEqual(labels["file"], "Page image file")
        self.assertEqual(list(vl.values())[0], {0: "no", 1: "yes"})

    def test_dta_keeps_very_long_text(self):
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"text": "ab" * 3000}), codebook=am.codebook(free_text=True))
        quiet(am.output, res, "long.dta")
        back = pd.read_stata("long.dta")
        self.assertEqual(len(back["text"].iloc[0]), 6000)

    def test_stata_names_are_valid_short_unreserved_and_unique(self):
        self.assertEqual(stata_names(["edat_ànima", "a.b", "1st", "_n", "x" * 40, "if"]),
                         ["edat_anima", "a_b", "v1st", "v_n", "x" * 32, "if_"])
        self.assertEqual(stata_names(["edat_anima", "edat_ànima"]), ["edat_anima", "edat_anima_2"])
        self.assertEqual(stata_names(["a_b", "a.b"]), ["a_b", "a_b_2"])


class Documents(Base):
    def test_pkl_gives_the_whole_result_back_and_output_reads_it(self):
        res = self.roster()
        quiet(am.output, res, "r.pkl")
        back = quiet(am.output, "r.pkl")
        self.assertEqual(len(back), len(res))
        again = am._result.as_result("r.pkl")
        self.assertEqual(again.info.contract, res.info.contract)

    def test_json_holds_one_object_per_page_with_the_models_data(self):
        res = self.roster()
        quiet(am.output, res, "r.json", include="all")
        x = json.load(open("r.json"))
        self.assertEqual([p["file"] for p in x], ["p001.jpg", "p002.jpg", "p003.jpg"])
        self.assertEqual(x[2]["status"], "check")
        self.assertIn("message", x[2])
        self.assertEqual(x[0]["data"]["heading"], "Heading of p001.jpg")

    def test_txt_has_a_section_per_page_with_an_aligned_table(self):
        res = self.roster()
        quiet(am.output, res, "r.txt")
        text = open("r.txt").read()
        self.assertIn("=== p001.jpg ===", text)
        self.assertIn("heading: Heading of p001.jpg", text)
        self.assertIn("row  name    age  sex", text)

    def test_free_text_in_txt_and_md_is_the_pages_text_under_each_heading(self):
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"text": f"Line one of {f}\nline | two"}), codebook=am.codebook(free_text=True))
        quiet(am.output, res, ["t.txt", "t.md"])
        self.assertIn("=== p001.jpg ===\n\nLine one of p001.jpg\nline | two", open("t.txt").read())
        md = open("t.md").read()
        self.assertIn("## p001\\.jpg" if False else "## p001.jpg", md)
        self.assertIn("line \\| two", md)

    def test_md_escapes_pipes_in_tables(self):
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"heading": "h", "records": [{"name": "a|b", "age": 1, "sex": "male"}]}))
        quiet(am.output, res, "r.md")
        md = open("r.md").read()
        self.assertIn("| a\\|b |", md)
        self.assertIn("|---:|---|---:|---|", md)

    def test_docx_opens_and_reads_back(self):
        res = self.roster()
        quiet(am.output, res, "r.docx")
        with zipfile.ZipFile("r.docx") as z:
            self.assertIn("word/document.xml", z.namelist())
        text = read_docx_text("r.docx")
        self.assertIn("p001.jpg", text)
        self.assertIn("Heading of p001.jpg", text)

    def test_free_text_docx_keeps_line_breaks(self):
        res = self.run_with(lambda f, b, n, r: gemini_reply(data={"text": "one\ntwo"}), codebook=am.codebook(free_text=True))
        quiet(am.output, res, "t.docx")
        self.assertIn("one\ntwo", read_docx_text("t.docx"))

    def test_several_formats_in_one_call_are_all_written(self):
        res = self.roster()
        out = quiet(am.output, res, ["a.xlsx", "a.csv", "a.dta", "a.json", "a.txt", "a.md", "a.docx", "a.pkl"])
        self.assertEqual(len(out), 8)
        self.assertTrue(all(os.path.exists(p) for p in out))

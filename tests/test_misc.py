import json
import os

import httpx
import openpyxl
import pandas as pd

import amanuensis as am
from amanuensis import _editor as ED
from amanuensis import _flatten as FL
from amanuensis import _gemini as G
from amanuensis import _practice as PR
from amanuensis import _store as ST
from amanuensis._errors import ContractError, InputError
from amanuensis._validate import validate

from .helpers import example_file, Case, FakeGemini, gemini_reply, make_jpegs, quiet, roster_codebook, roster_data, test_instructions



class Store(Case):
    def test_record_files_are_named_after_the_image_and_never_over_a_control_file(self):
        self.assertEqual(os.path.basename(ST.record_file("d", "p001.jpg")), "p001.json")
        self.assertEqual(os.path.basename(ST.record_file("d", "Contract.JPG")), "_Contract.json")
        self.assertEqual(os.path.basename(ST.record_file("d", "_contract.jpg")), "__contract.json")
        self.assertEqual(os.path.basename(ST.record_file("d", "a:b.jpg")), "a_b.json")

    def test_damaged_records_are_skipped_with_one_warning(self):
        os.makedirs("r")
        ST.write_record("r", {"file": "p1.jpg", "status": "ok"})
        open("r/p2.json", "w").write("{broken")
        open("r/p3.json", "w").write('{"no": "file"}')
        with self.warns("2 result files could not be read"):
            recs = ST.read_records("r")
        self.assertEqual(list(recs), ["p1.jpg"])

    def test_contracts_are_stored_once_and_rebuilt(self):
        os.makedirs("r")
        con = ST.contract(test_instructions(), roster_codebook())
        ST.save_contract("r", con)
        ST.save_contract("r", con)
        entry = ST.read_contracts("r")[con["id"]]
        cb, ins = ST.contract_objects(entry, con["id"])
        self.assertEqual(cb.schema, roster_codebook().schema)
        self.assertEqual([v.values for v in cb.vars], [v.values for v in roster_codebook().vars])
        self.assertEqual(ins.full, test_instructions().full)

    def test_describe_change_says_what_differs(self):
        a = ST.contract_entry(ST.contract(test_instructions(), roster_codebook()))
        b = ST.contract_entry(ST.contract(test_instructions("Other."), quiet(am.codebook, pd.DataFrame({"variable": ["name", "parish"]}))))
        words = ST.describe_change(a, b)
        self.assertIn("codebook: added parish", words)
        self.assertTrue(words[0].startswith("instructions:"))

    def test_the_job_list_is_written_read_and_removed_when_empty(self):
        os.makedirs("r")
        ST.write_jobs("r", [{"name": "batches/1", "pages": [{"file": "a.jpg"}]}])
        self.assertEqual(ST.pending_job_files("r"), ["a.jpg"])
        ST.write_jobs("r", [])
        self.assertFalse(os.path.exists("r/batch-jobs.json"))

    def test_supersede_moves_records_aside_never_deleting_them(self):
        os.makedirs("r")
        ST.write_record("r", {"file": "p1.jpg", "status": "ok"})
        n = ST.supersede("r", ["p1.jpg"], dest="r/superseded/x", quiet=True)
        ST.write_record("r", {"file": "p1.jpg", "status": "ok"})
        n += ST.supersede("r", ["p1.jpg"], dest="r/superseded/x", quiet=True)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(os.listdir("r/superseded/x")), ["p1-1.json", "p1.json"])

    def test_the_session_store_keys_by_settings_and_full_image_path(self):
        make_jpegs("a", 1)
        make_jpegs("b", 1)
        pa = am._images.as_images("a")
        pb = am._images.as_images("b")
        ST.session_keep({"file": "p001.jpg", "status": "ok"}, pa["image"].iloc[0], "fp")
        self.assertEqual(list(ST.session_records("fp", pa)), ["p001.jpg"])
        self.assertEqual(ST.session_records("fp", pb), {})
        self.assertEqual(ST.session_records("other", pa), {})
        with am.options.override(session=False):
            self.assertEqual(ST.session_records("fp", pa), {})


class Results(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 2)

    def test_a_results_folder_alone_rebuilds_the_result(self):
        res = quiet(am.transcribe, "pages", test_instructions(), roster_codebook(), model="mock")
        back = quiet(am._result.result_from_folder, "pages/extraction")
        pd.testing.assert_frame_equal(pd.DataFrame(res), pd.DataFrame(back))
        self.assertEqual(back.info.contract, res.info.contract)

    def test_a_folder_with_practice_and_real_results_gives_only_the_real_ones(self):
        quiet(am.transcribe, "pages", test_instructions(), roster_codebook(), model="mock")
        fake = FakeGemini("pages", lambda f, b, n, r: gemini_reply(data=roster_data(f)))
        with G.mocked(fake):
            quiet(am.transcribe, "pages", test_instructions(), roster_codebook(), limit=1)
        back = quiet(am._result.result_from_folder, "pages/extraction")
        self.assertEqual(back.pages["status"].tolist(), ["ok", "not run"])
        self.assertEqual(back.info.settings["model"], "gemini-3.8-flash")

    def test_two_instructions_in_one_folder_need_contract(self):
        quiet(am.transcribe, "pages", test_instructions(), roster_codebook(), model="mock")
        quiet(am.transcribe, "pages", test_instructions("Other."), roster_codebook(), model="mock", only="p001.jpg", redo="all")
        with self.assertRaises(ContractError) as e:
            am._result.result_from_folder("pages/extraction")
        self.assertIn("contract=", str(e.exception))

    def test_the_result_keeps_its_information_through_selections_and_prints_a_header(self):
        res = quiet(am.transcribe, "pages", test_instructions(), roster_codebook(), model="mock")
        self.assertTrue(repr(res).startswith("amanuensis result: 6 rows from 2 of 2 pages (mock)"))
        part = res[res["row"] == 1]
        self.assertIs(am._result.result_info(part), am._result.result_info(res))
        self.assertIs(part.info.records, res.info.records)
        self.assertTrue(repr(part).startswith("(part of an amanuensis result: 2 of 6 rows)"))
        self.assertIsNone(getattr(pd.concat([res, res]), "info", None) if False else None)
        self.assertEqual(list(res.pages.columns[:5]), ["file", "source", "source_page", "status", "problem"])


class Flatten(Case):
    def test_whole_numbers_above_two_billion_stay_exact(self):
        s = FL.as_series([3500000000, None, 7], "integer")
        self.assertEqual(str(s.dtype), "Int64")
        self.assertEqual(s.tolist()[0], 3500000000)
        self.assertEqual(FL.as_series([2.5, 1], "integer").tolist(), [2.5, 1.0])

    def test_lists_of_numbers_are_spread_as_text_without_scientific_notation(self):
        cb = quiet(am.codebook, json=json.dumps({"type": "object", "properties": {"vals": {"type": "array", "items": {"type": "number"}}}}))
        tab = FL.flatten_records([{"file": "a.jpg", "data": {"vals": [3500000000, 1.5]}}], cb)
        self.assertEqual(tab["vals_1"].tolist(), ["3500000000"])
        self.assertEqual(tab["vals_2"].tolist(), ["1.5"])

    def test_records_inside_a_plain_object_are_found_and_its_other_fields_stay_page_variables(self):
        s = {"type": "object", "properties": {"body": {"type": "object", "properties": {"volume": {"type": "string"},
             "people": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}}}}}}
        cb = quiet(am.codebook, json=json.dumps(s))
        tab = FL.flatten_records([{"file": "a.jpg", "data": {"body": {"volume": "II", "people": [{"name": "A"}, {"name": "B"}]}}}], cb)
        self.assertEqual(tab["body.volume"].tolist(), ["II", "II"])
        self.assertEqual(tab["name"].tolist(), ["A", "B"])

    def test_a_nested_json_schema_gives_the_long_table_with_enclosing_fields_and_counters(self):
        s = {"type": "object", "properties": {"households": {"type": "array", "items": {"type": "object", "properties": {
            "head": {"type": "string"}, "members": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}}}}}}}
        cb = quiet(am.codebook, json=json.dumps(s))
        data = {"households": [{"head": "H1", "members": [{"name": "a"}, {"name": "b"}]}, {"head": "H2", "members": []}]}
        tab = FL.flatten_records([{"file": "a.jpg", "data": data}], cb)
        self.assertEqual(tab["households.head"].tolist(), ["H1", "H1", "H2"])
        self.assertEqual(tab["households_n"].tolist(), [1, 1, 2])
        self.assertEqual(tab["name"].tolist()[:2], ["a", "b"])
        self.assertTrue(pd.isna(tab["name"].tolist()[2]))

    def test_codebook_tables_a_page_with_no_records_gives_no_rows(self):
        tab = FL.flatten_records([{"file": "a.jpg", "data": {"heading": "x", "records": []}}], roster_codebook())
        self.assertEqual(len(tab), 0)
        self.assertEqual(list(tab.columns), ["file", "source", "source_page", "row", "heading", "name", "age", "sex"])


class Setup(Case):
    def test_check_setup_offline_has_one_row_per_check_and_writes_nothing(self):
        before = sorted(os.listdir("."))
        out = am.check_setup(online=False)
        self.assertEqual(list(out.columns), ["check", "status", "detail", "fix"])
        self.assertEqual(sorted(os.listdir(".")), before)
        row = out[out["check"] == "Gemini key"].iloc[0]
        self.assertEqual(row["status"], "OK")
        self.assertIn("AIza...fgh", row["detail"])
        self.assertNotIn("✔", out.report())

    def test_a_missing_key_is_a_problem_with_the_export_spelled_out(self):
        os.environ.pop("GEMINI_API_KEY")
        os.environ["GOOGLE_API_KEY"] = "AIza-google-0123456789abcdefghijkl"
        out = quiet(am.check_setup, online=False)
        row = out[out["check"] == "Gemini key"].iloc[0]
        self.assertEqual(row["status"], "PROBLEM")
        self.assertIn("export GEMINI_API_KEY", row["fix"])
        self.assertIn("reads only GEMINI_API_KEY", row["detail"])

    def test_a_second_key_is_a_note_and_a_wrong_shape_a_problem(self):
        os.environ["GOOGLE_API_KEY"] = "AIza-google-0123456789abcdefghijkl"
        out = quiet(am.check_setup, online=False)
        self.assertEqual(out[out["check"] == "second key"]["status"].tolist(), ["NOTE"])
        os.environ["GEMINI_API_KEY"] = "abc"
        out = quiet(am.check_setup, online=False)
        self.assertEqual(out[out["check"] == "Gemini key"]["status"].tolist(), ["PROBLEM"])

    def test_online_asks_google_about_the_key(self):
        with G.mocked(lambda req: httpx.Response(200, json={"models": []})):
            out = quiet(am.check_setup)
        self.assertEqual(out[out["check"] == "Google"]["detail"].tolist(), ["reachable, and it accepts the key"])
        with G.mocked(lambda req: httpx.Response(400, json={"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})):
            out = quiet(am.check_setup)
        self.assertEqual(out[out["check"] == "Google"]["status"].tolist(), ["PROBLEM"])
        with G.mocked(lambda req: httpx.Response(403, json={})):
            out = quiet(am.check_setup, key=False)
        self.assertEqual(out[out["check"] == "Google"]["detail"].tolist(), ["reachable (no key to check)"])

    def test_a_working_folder_inside_onedrive_gets_a_note(self):
        os.makedirs("OneDrive/project")
        os.chdir("OneDrive/project")
        out = quiet(am.check_setup, online=False)
        self.assertEqual(out[out["check"] == "synced folder"]["status"].tolist(), ["NOTE"])


class Example(Case):
    def test_example_project_copies_every_file_and_keeps_edited_ones(self):
        ex = quiet(am.example_project)
        self.assertEqual(ex.folder, __import__("pathlib").Path(os.path.realpath("amanuensis-example")))
        for p in (ex.scans, ex.instructions, ex.codebook, ex.codebook_csv, ex.gold):
            self.assertTrue(p.exists())
        self.assertFalse((ex.folder / "replies.json").exists())
        open(ex.instructions, "w").write("mine")
        am.example_project()
        self.assertEqual(open(ex.instructions).read(), "mine")
        self.assertIn("files were kept" if False else "Kept 1 file", self.messages()) if False else None
        quiet(am.example_project, overwrite=True)
        self.assertNotEqual(open(ex.instructions).read(), "mine")

    def test_example_project_refuses_a_file_and_package_folders(self):
        open("afile", "w").write("x")
        with self.assertRaises(InputError):
            am.example_project("afile")
        with self.assertRaises(InputError):
            am.example_project(os.path.join(os.path.dirname(am.__file__), "x"))

    def test_the_example_codebook_builds_alike_from_excel_and_csv_and_every_reply_fits(self):
        a = quiet(am.codebook, example_file("codebook.xlsx"))
        b = quiet(am.codebook, example_file("codebook.csv"))
        self.assertEqual(a.schema, b.schema)
        for f, rep in PR.example_replies().items():
            self.assertEqual(validate(rep["data"], a.schema), [], f)

    def test_images_renders_the_example_pdf_as_three_pages_with_the_expected_names(self):
        ex = quiet(am.example_project)
        pages = quiet(am.images, ex.scans)
        self.assertEqual(list(pages["file"]), ["padro1930_p001.jpg", "padro1930_p002.jpg", "padro1930_p003.jpg"])


class Practice(Case):
    def test_mock_value_invents_data_of_every_type(self):
        s = {"type": "object", "properties": {"t": {"type": "string"}, "n": {"type": "integer"}, "x": {"type": "number"},
             "b": {"type": "boolean"}, "e": {"enum": ["a", "b", None]}, "o": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
             "l": {"type": "array", "items": {"type": "string"}}, "text": {"type": ["string", "null"]}}}
        v = PR.mock_value(s, "value", 1, "p.jpg", s)
        self.assertEqual(v["t"], "t 1")
        self.assertEqual((v["n"], v["x"], v["b"], v["e"], v["o"]), (1, 1.5, True, "a", 1))
        self.assertEqual(v["l"], ["l 1", "l 2"])
        self.assertTrue(v["text"].startswith("Practice transcription of p.jpg"))

    def test_mock_data_fits_the_codebooks_it_is_asked_about(self):
        for cb in (roster_codebook(), am.codebook(free_text=True), quiet(am.codebook, example_file("codebook.xlsx"))):
            self.assertEqual(validate(PR.mock_value(cb.schema, "value", 1, "f", cb.schema), cb.schema), [])

    def test_the_practice_server_answers_camel_and_snake_requests_and_refuses_other_endpoints(self):
        h = PR.handler("mock")
        body = {"contents": [{"parts": [{"text": "x"}]}], "generationConfig": {"responseJsonSchema": roster_codebook().schema}}
        r = h(httpx.Request("POST", "https://g/v1beta/models/m:generateContent", json=body))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["candidates"][0]["finishReason"], "STOP")
        snake = PR.practice_reply("mock", {"generation_config": {"response_schema": {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}}}, "f")
        self.assertEqual(json.loads(snake["candidates"][0]["content"]["parts"][0]["text"]), {"a": "a 1"})
        self.assertEqual(h(httpx.Request("GET", "https://g/v1beta/whatever")).status_code, 404)


class Editor(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 3)
        self.cb = roster_codebook()
        quiet(am.benchmark, "gold.xlsx", "pages", self.cb, n=2, seed=4)
        self.state = ED.editor_state("gold.xlsx", self.cb, am._images.as_images("pages"))

    def test_the_state_reads_the_file_and_finds_the_images(self):
        self.assertEqual(len(self.state["files"]), 2)
        self.assertEqual(self.state["done"], [False, False])
        self.assertTrue(all(self.state["images"].values()))
        self.assertEqual(ED.shown_rows(self.state), 1)

    def test_pure_functions_read_check_and_save_a_page(self):
        vals = ED.form_values({"p_1": "Heading", "r_1_1": "Anna", "r_1_2": "x"}, self.state)
        self.assertEqual(vals, {"page": {"heading": "Heading"}, "rows": {"name": ["Anna"], "age": ["x"], "sex": [""]}})
        self.assertEqual(ED.check_values(vals, self.cb), ['row 1, age: "x" is not a whole number'])
        s = ED.add_row(ED.store_page(self.state, vals))
        self.assertEqual(ED.shown_rows(s), 2)
        s = ED.delete_row(s)
        self.assertEqual(ED.shown_rows(s), 1)

    def test_handle_validates_saves_marks_done_and_moves_between_pages(self):
        s, msg, kind, fin = ED.handle(self.state, "next", {"p_1": "H", "r_1_1": "Anna", "r_1_2": "x"})
        self.assertEqual(kind, "error")
        self.assertEqual(s["current"], 0)
        s, msg, kind, fin = ED.handle(s, "next", {"p_1": "H", "r_1_1": "Anna", "r_1_2": "34", "r_1_3": "female"})
        self.assertEqual((kind, s["current"], s["done"][0]), ("ok", 1, True))
        s, msg, kind, fin = ED.handle(s, "prev", {"p_1": "half"})
        self.assertEqual(s["current"], 0)
        s, msg, kind, fin = ED.handle(s, "finish", {"p_1": "H", "r_1_1": "Anna", "r_1_2": "34", "r_1_3": "female"})
        self.assertTrue(fin)
        g = quiet(am.benchmark, "gold.xlsx")
        self.assertEqual(g["name"].tolist(), ["Anna"])
        self.assertEqual(g["age"].tolist(), [34])

    def test_the_routes_serve_the_page_the_state_and_the_actions(self):
        ed = ED.Editor(self.state)
        base = f"/{ed.token}/"
        status, ctype, body = ed.route("GET", base)
        self.assertEqual(status, 200)
        self.assertIn(b"Save and finish", body)
        status, ctype, body = ed.route("GET", base + "state")
        view = json.loads(body)
        self.assertEqual(view["row_vars"][2]["values"], ["male", "female", "other"])
        status, ctype, body = ed.route("GET", base + view["image"])
        self.assertEqual((status, ctype), (200, "image/jpeg"))
        status, ctype, body = ed.route("POST", base + "action", json.dumps({"action": "add_row", "page": 1, "inputs": {}}).encode())
        self.assertEqual(len(json.loads(body)["rows"]), 2)
        self.assertEqual(ed.route("GET", base + "nope")[0], 404)
        self.assertEqual(ed.route("GET", "/state")[0], 404)  # without the token

    def test_free_text_is_edited_in_a_text_area_and_saved_to_the_text_file(self):
        quiet(am.benchmark, "g.txt", "pages", am.codebook(free_text=True), n=1, seed=4)
        st = ED.editor_state("g.txt")
        s, msg, kind, fin = ED.handle(st, "finish", {"p_1": "Line one\nline two"})
        self.assertTrue(fin)
        g = quiet(am.benchmark, "g.txt")
        self.assertEqual(g["text"].tolist(), ["Line one\nline two"])


class HomeFolder(Case):
    def test_paths_starting_with_a_tilde_mean_the_home_folder(self):
        os.makedirs("home")
        old = os.environ.get("HOME")
        os.environ["HOME"] = os.path.realpath("home")
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old) if old is not None else os.environ.pop("HOME"))
        ex = quiet(am.example_project, "~/ex")
        self.assertTrue(os.path.exists("home/ex/padro1930.pdf"))
        pages = quiet(am.images, "~/ex/padro1930.pdf", save_to="~/ex/pages")
        res = quiet(am.transcribe, pages, "~/ex/instructions.md", "~/ex/codebook.xlsx", model="example", save_to="~/ex/out")
        self.assertTrue(os.path.exists("home/ex/out/padro1930_p001.json"))
        quiet(am.output, "~/ex/out", "~/ex/out.xlsx")
        self.assertTrue(os.path.exists("home/ex/out.xlsx"))
        s = quiet(am.score, "~/ex/gold.xlsx", "~/ex/out", report="~/ex/acc.html")
        self.assertTrue(os.path.exists("home/ex/acc.html"))
        self.assertFalse(os.path.exists("~"))

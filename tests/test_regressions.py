"""Regression tests for defects that have been fixed. One class per defect,
named for the behaviour it guards; the F01 ... F33 prefixes are stable
identifiers for those defects. Each test asserts the fixed behaviour, so it
fails if the defect comes back. Gemini is faked at the HTTP level, as
everywhere in the suite. The chmod-based tests need a POSIX file system and a
non-root user."""

import json
import os
import stat
import unittest
import warnings

import httpx
import pandas as pd

import amanuensis as am
from amanuensis import _gemini as G
from amanuensis import _store
from amanuensis import _transcribe as TR
from amanuensis._errors import InputError

from .helpers import (Case, FakeBatch, FakeGemini, error_response, gemini_reply, make_jpegs, make_pdf, quiet,
                      roster_codebook, roster_data, test_instructions)

POSIX = os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0


def ok(f, b, n, r):
    return gemini_reply(data=roster_data(f))


class Base(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 3)
        self.cb = roster_codebook()
        self.ins = test_instructions()

    def tr(self, handler=ok, fake=None, images="pages", instructions=None, **kw):
        fake = fake or FakeGemini("pages", handler)
        with G.mocked(fake):
            res = am.transcribe(images, instructions or self.ins, self.cb, **kw)
        return res, fake

    def record(self, name, folder="pages/extraction"):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def kept(self, folder="pages/extraction/superseded"):
        return sorted(f for _, _, fs in os.walk(folder) for f in fs)


# ------- money ---------------------------------------------------------------------------

@unittest.skipUnless(POSIX, "needs chmod")
class F01ParallelRoundKeepsPaidReplies(Base):
    def test_an_unreadable_image_stops_the_run_before_anything_of_its_round_is_sent(self):
        pages = am.images("pages")
        os.chmod("pages/p003.jpg", 0)
        try:
            with self.assertRaises(InputError) as e:
                self.tr(images=pages, workers=3)
        finally:
            os.chmod("pages/p003.jpg", 0o644)
        self.assertIn("Could not read the page image", str(e.exception))
        self.assertNotIn("Please report", self.messages())
        res, fake = self.tr(images=pages, workers=3)
        self.assertEqual(sorted(fake.files_called()), ["p001.jpg", "p002.jpg", "p003.jpg"])
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)

    def test_an_unexpected_error_of_one_page_keeps_the_other_replies_of_its_round(self):
        def handler(f, b, n, r):
            return RuntimeError("something unexpected") if f == "p002.jpg" and n == 1 else ok(f, b, n, r)
        with self.assertRaises(RuntimeError) as e:
            self.tr(handler=handler, workers=3)
        self.assertEqual(self.record("p001.json")["status"], "ok")
        self.assertEqual(self.record("p003.json")["status"], "ok")
        self.assertIsNone(self.record("p002.json"))
        self.assertEqual(e.exception.result.pages["status"].tolist(), ["ok", "not run", "ok"])
        _, fake = self.tr(workers=3)
        self.assertEqual(fake.files_called(), ["p002.jpg"])

    def test_a_reply_that_cannot_be_decoded_is_a_network_failure_of_that_page_only(self):
        def handler(f, b, n, r):
            if f == "p002.jpg" and n == 1:
                raise httpx.DecodingError("garbled", request=r)
            return ok(f, b, n, r)
        res, fake = self.tr(handler=handler, workers=3, retries=0)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "failed", "ok"])
        self.assertEqual(res.pages["problem"].tolist()[1], "network")
        _, fake = self.tr(workers=3)
        self.assertEqual(fake.files_called(), ["p002.jpg"])


@unittest.skipUnless(POSIX, "needs chmod")
class F02UnreadableIsNotDamaged(Base):
    def test_a_record_that_cannot_be_opened_stops_the_run_and_is_not_paid_again(self):
        self.tr()
        before = open("pages/extraction/p002.json", "rb").read()
        os.chmod("pages/extraction/p002.json", 0)
        try:
            with self.assertRaises(InputError) as e:
                _, fake = self.tr()
        finally:
            os.chmod("pages/extraction/p002.json", 0o644)
        self.assertIn("could not be opened: p002.json", str(e.exception))
        self.assertIn("Nothing was sent", str(e.exception))
        self.assertEqual(open("pages/extraction/p002.json", "rb").read(), before)
        _, fake = self.tr()
        self.assertEqual(fake.calls, [])

    def test_a_contract_json_that_cannot_be_opened_is_never_rewritten(self):
        self.tr()
        with G.mocked(FakeGemini("pages", ok)):
            am.transcribe("pages", test_instructions("Another source."), self.cb, redo=["p001.jpg"], only=["p001.jpg"])
        ids = sorted(json.load(open("pages/extraction/contract.json"))["contracts"])
        os.chmod("pages/extraction/contract.json", 0)
        try:
            with self.assertRaises(InputError) as e:
                self.tr(instructions=test_instructions("Another source."), only=["p001.jpg"])
        finally:
            os.chmod("pages/extraction/contract.json", 0o644)
        self.assertIn("contract.json could not be opened", str(e.exception))
        self.assertEqual(sorted(json.load(open("pages/extraction/contract.json"))["contracts"]), ids)
        self.assertEqual(len(ids), 2)

    def test_a_damaged_contract_json_is_kept_aside_before_a_new_one_is_started(self):
        os.makedirs("r")
        with open("r/contract.json", "w") as f:
            f.write('{"contracts": {"abc": {"instructions": "lost?"')
        con = _store.contract(self.ins, self.cb)
        with self.warns("was damaged"):
            _store.save_contract("r", con)
        aside = [f for f in os.listdir("r") if f.startswith("contract.json.damaged-")]
        self.assertEqual(len(aside), 1)
        self.assertIn("lost?", open(os.path.join("r", aside[0])).read())
        self.assertEqual(list(_store.read_contracts("r")), [con["id"]])
        self.assertEqual(_store.read_records("r"), {})  # the copy is never read as a page record


class F08PaidRepliesSurviveErrors(Base):
    @unittest.skipUnless(POSIX, "needs chmod")
    def test_a_reply_whose_record_cannot_be_written_is_saved_by_the_next_run_not_paid_again(self):
        def handler(f, b, n, r):
            if f == "p002.jpg":
                os.chmod("pages/extraction", stat.S_IRUSR | stat.S_IXUSR)
            return ok(f, b, n, r)
        try:
            with self.assertRaises(InputError) as e:
                self.tr(handler=handler)
        finally:
            os.chmod("pages/extraction", 0o755)
        self.assertIn("Could not write", str(e.exception))
        self.assertEqual(e.exception.result.pages["status"].tolist(), ["ok", "ok", "not run"])
        self.assertIsNone(self.record("p002.json"))
        _, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p003.jpg"])
        self.assertEqual(self.record("p002.json")["status"], "ok")
        self.assertIn("from this Python session", self.messages())

    def test_a_reply_kept_in_the_session_wins_over_an_older_failure_in_the_folder(self):
        self.tr(handler=lambda f, b, n, r: error_response(503) if f == "p002.jpg" else ok(f, b, n, r), retries=0)
        self.assertEqual(self.record("p002.json")["status"], "failed")
        orig = TR.write_record
        calls = {"n": 0}

        def failing(folder, rec):
            if rec["file"] == "p002.jpg" and rec["status"] == "ok" and calls["n"] == 0:
                calls["n"] += 1
                raise InputError("Could not write p002.json.")
            return orig(folder, rec)
        TR.write_record = failing
        try:
            with self.assertRaises(InputError):
                self.tr()
        finally:
            TR.write_record = orig
        _, fake = self.tr()
        self.assertEqual(fake.calls, [])
        self.assertEqual(self.record("p002.json")["status"], "ok")

    def test_a_draft03_required_true_no_longer_loses_every_reply(self):
        s = {"type": "object", "properties": {
            "head": {"type": "object", "required": True, "properties": {"name": {"type": "string"}}},
            "people": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}}}}
        cb = quiet(am.codebook, json=json.dumps(s))
        make_jpegs("scans", 2)
        res = quiet(am.transcribe, "scans", None, cb, model="mock")
        self.assertEqual(res.pages["status"].tolist(), ["ok", "ok"])
        saved = sorted(f for f in os.listdir("scans/extraction") if f.endswith(".json") and f != "contract.json")
        self.assertEqual(saved, ["p001.json", "p002.json"])

    def test_validate_reports_what_it_cannot_check_instead_of_raising(self):
        from amanuensis import _validate as V
        orig = V._node

        def broken(*a, **kw):
            raise TypeError("an odd schema")
        V._node = broken
        try:
            errs = V.validate({"a": 1}, {"type": "object"})
        finally:
            V._node = orig
        self.assertEqual(errs, ["(whole reply): could not be checked against the schema here (TypeError: an odd schema)"])


@unittest.skipUnless(POSIX, "needs chmod")
class F09OldResultNeverOverwrittenWithoutACopy(Base):
    def test_when_the_old_result_cannot_be_moved_or_copied_it_is_not_replaced(self):
        self.tr(handler=lambda f, b, n, r: gemini_reply(data=roster_data("FIRST")))
        orig = TR.stamp
        TR.stamp = lambda: "20261108-120000"
        dest = "pages/extraction/superseded/20261108-120000"
        os.makedirs(dest)
        os.chmod(dest, 0o555)
        try:
            with self.assertRaises(InputError) as e:
                self.tr(handler=lambda f, b, n, r: gemini_reply(data=roster_data("SECOND")), redo="all")
        finally:
            TR.stamp = orig
            os.chmod(dest, 0o755)
        self.assertIn("could not be moved", str(e.exception))
        self.assertEqual(self.record("p001.json")["data"]["heading"], "Heading of FIRST")
        self.assertEqual(e.exception.result.info.records["p001.jpg"]["data"]["heading"], "Heading of SECOND")

    def test_when_the_move_fails_the_old_result_is_copied_aside_first(self):
        self.tr(handler=lambda f, b, n, r: gemini_reply(data=roster_data("FIRST")))
        orig = _store.os.rename

        def no_rename(a, b):
            raise PermissionError(13, "held by another program")
        _store.os.rename = no_rename
        try:
            self.tr(handler=lambda f, b, n, r: gemini_reply(data=roster_data("SECOND")), redo="all")
        finally:
            _store.os.rename = orig
        self.assertEqual(self.record("p001.json")["data"]["heading"], "Heading of SECOND")
        self.assertEqual(self.kept(), ["p001.json", "p002.json", "p003.json"])
        old = [os.path.join(d, f) for d, _, fs in os.walk("pages/extraction/superseded") for f in fs if f == "p001.json"][0]
        self.assertEqual(json.load(open(old))["data"]["heading"], "Heading of FIRST")


class F10FailedRedoKeepsTheResult(Base):
    def test_a_503_on_every_redo_request_keeps_every_earlier_result_in_the_table(self):
        first, _ = self.tr()
        res, _ = self.tr(handler=lambda f, b, n, r: error_response(503), redo="all", retries=0)
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertEqual(len(res), len(first))
        self.assertEqual(self.kept(), [])
        self.assertIn("3 pages to redo failed this time and keep their earlier results", self.messages())

    def test_a_truncated_redo_keeps_the_result_and_its_cost_is_counted(self):
        self.tr()
        c_first = self.record("p001.json")["cost_usd"]

        def cut(f, b, n, r):
            return gemini_reply(text='{"heading": "x", "records": [', finish="MAX_TOKENS", prompt=2500, output=32000)
        res, _ = self.tr(handler=cut, redo=["p001.jpg"], only=["p001.jpg"])
        rec = self.record("p001.json")
        self.assertEqual(rec["status"], "ok")
        self.assertGreater(rec["other_attempts_cost_usd"], 0.1)
        self.assertAlmostEqual(res.pages["total_cost_usd"].iloc[0], c_first + rec["other_attempts_cost_usd"])


class F12BudgetFromTheFirstRound(Base):
    def test_eight_workers_and_a_tiny_budget_send_nothing(self):
        make_jpegs("many", 8, prefix="m")
        res, fake = self.tr(fake=FakeGemini("many", ok), images="many", workers=8, budget=0.001)
        self.assertEqual(fake.calls, [])
        self.assertIn("Stopped before sending anything", self.messages())

    def test_budget_zero_sends_nothing(self):
        res, fake = self.tr(budget=0)
        self.assertEqual(fake.calls, [])

    def test_the_first_round_is_one_page_and_later_rounds_fit_the_budget(self):
        self.freeze_price_date("2026-11-08")
        make_jpegs("many", 8, prefix="m")
        res, fake = self.tr(fake=FakeGemini("many", ok), images="many", workers=8, budget=0.01)
        spent = float(res.pages["cost_usd"].sum())
        self.assertLessEqual(spent, 0.01)
        self.assertEqual(len(fake.calls), 3)  # 1 alone, then 2 that the measured cost allows


class F13ReRenderedImagesAreNotPaidAgain(Case):
    def setUp(self):
        super().setUp()
        make_pdf("book.pdf", 3)
        self.cb, self.ins = roster_codebook(), test_instructions()
        pages = quiet(am.images, "book.pdf")
        with G.mocked(FakeGemini("book_pages", ok)):
            quiet(am.transcribe, pages, self.ins, self.cb)
        self.pages = quiet(am.images, "book.pdf", overwrite=True, quality=84)

    def test_many_changed_images_keep_their_results_unless_redo_is_given(self):
        fake = FakeGemini("book_pages", ok)
        with G.mocked(fake):
            res = am.transcribe(self.pages, self.ins, self.cb)
        self.assertEqual(fake.calls, [])
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertIn("3 page images changed since they were extracted", self.messages())
        with G.mocked(fake):
            am.transcribe(self.pages, self.ins, self.cb, redo="all")
        self.assertEqual(len(fake.calls), 3)

    def test_asked_in_an_interactive_session(self):
        import builtins
        answers = iter(["2"])
        orig = builtins.input
        builtins.input = lambda *a: next(answers)
        fake = FakeGemini("book_pages", ok)
        try:
            with am.options.override(interactive=True, ask=True), G.mocked(fake):
                am.transcribe(self.pages, self.ins, self.cb, confirm=False)
        finally:
            builtins.input = orig
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len([f for _, _, fs in os.walk("book_pages/extraction/superseded") for f in fs]), 3)


class F18EveryPaidAttemptIsCounted(Base):
    def test_a_truncated_then_retried_page_reports_both_costs(self):
        def cut(f, b, n, r):
            if f == "p001.jpg":
                return gemini_reply(text='{"heading": "x", "records": [', finish="MAX_TOKENS", prompt=2500, output=32000)
            return ok(f, b, n, r)
        res1, _ = self.tr(handler=cut)
        c1 = float(res1.pages["cost_usd"].iloc[0])
        res2, fake = self.tr(max_tokens=65536)
        self.assertEqual(fake.files_called(), ["p001.jpg"])
        c2 = float(res2.pages["cost_usd"].iloc[0])
        self.assertAlmostEqual(float(res2.pages["total_cost_usd"].iloc[0]), c1 + c2)
        self.assertIn("for other attempts of the same pages", self.messages())
        am.output(res2, "out.xlsx")
        import openpyxl
        about = {r[0]: r[1] for r in openpyxl.load_workbook("out.xlsx")["about"].iter_rows(min_row=2, values_only=True)}
        total = float(res2.pages["total_cost_usd"].sum())
        self.assertAlmostEqual(float(about["cost_usd"]), total, places=6)

    def test_a_redo_carries_the_cost_of_the_result_it_replaces(self):
        self.tr()
        c_old = self.record("p001.json")["cost_usd"]
        res, _ = self.tr(redo=["p001.jpg"], only=["p001.jpg"])
        self.assertAlmostEqual(self.record("p001.json")["other_attempts_cost_usd"], c_old)


class F23SettingsThatChangeTheRequest(Base):
    def test_api_args_and_a_temperature_are_part_of_what_makes_a_page_done(self):
        self.tr(limit=2, api_args={"systemInstruction": {"parts": [{"text": "Modernise every spelling."}]}}, temperature=1.5)
        with self.assertRaises(am.ContractError) as e:
            self.tr()
        self.assertIn("temperature: 1.5 -> not sent", str(e.exception))
        self.assertIn("api_args", str(e.exception))
        res, fake = self.tr(redo="all")
        self.assertEqual(len(fake.calls), 3)
        self.assertIsNone(res.info.settings["temperature"])
        res, _ = self.tr(save_to="other", temperature=0.5)
        self.assertEqual(res.info.settings["temperature"], 0.5)

    def test_runs_without_them_keep_their_fingerprints(self):
        from amanuensis._result import fingerprint
        self.assertEqual(fingerprint("c", "gemini-3.8-flash", "high", "low"), fingerprint("c", "gemini-3.8-flash", "high", "low", "not sent", None))
        self.assertEqual(fingerprint("c", "m", "high", "low", None, {}), "c|m|high|low")


# ------- smaller things (F33) ----------------------------------------------------------------

class F33SmallerThings(Base):
    def test_a_key_set_with_quotes_in_windows_cmd_is_used_without_them(self):
        from amanuensis._errors import MissingKeyError
        k = "AIzaSyFAKE-test-key-0123456789abcdefgh"
        self.assertEqual(G.check_key_shape(f'"{k}"'), k)
        with self.assertRaises(MissingKeyError):
            G.check_key_shape(f'AIza"{k}')
        os.environ.pop("GEMINI_API_KEY")
        with self.assertRaises(MissingKeyError) as e:
            G.gemini_key()
        self.assertIn("$env:GEMINI_API_KEY", str(e.exception))
        self.assertIn("setx GEMINI_API_KEY", str(e.exception))

    def test_res_info_called_is_the_pandas_method(self):
        import io
        res, _ = self.tr()
        buf = io.StringIO()
        res.info(buf=buf)
        self.assertIn("Data columns", buf.getvalue())
        self.assertEqual(res.info.settings["model"], "gemini-3.8-flash")

    def test_save_to_with_an_output_extension_is_refused(self):
        with self.assertRaises(InputError) as e:
            self.tr(save_to="results.xlsx")
        self.assertIn("output(res, 'results.xlsx')", str(e.exception))
        self.assertFalse(os.path.exists("results.xlsx"))

    def test_a_second_run_on_a_folder_in_use_is_refused(self):
        import socket
        import subprocess
        import sys
        import time
        os.makedirs("pages/extraction")

        def lock(pid):
            with open("pages/extraction/transcribe.lock", "w") as f:
                json.dump({"pid": pid, "host": socket.gethostname(), "started": "now", "time": time.time()}, f)
        lock(os.getppid())  # a process that is running
        with self.assertRaises(InputError) as e:
            self.tr()
        self.assertIn("Another transcribe() is working", str(e.exception))
        self.assertFalse(os.path.exists("pages/extraction/p001.json"))
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        lock(p.pid)  # a process that is gone: its lock is taken over
        res, fake = self.tr()
        self.assertEqual(len(fake.calls), 3)
        self.assertIn("Took over transcribe.lock", self.messages())
        self.assertFalse(os.path.exists("pages/extraction/transcribe.lock"))

    def test_on_windows_the_process_holding_a_lock_is_asked_about_and_never_ended(self):
        # Windows simulated: os.kill(pid, 0) would end the process there, so
        # it must never be called; Windows is asked through kernel32 instead
        import ctypes
        import types
        from unittest import mock
        running, exited, other_user, gone = 4100, 4200, 4300, 4400
        last_error = [0]
        closed = []

        def open_process(access, inherit, pid):
            if pid in (running, exited):
                return pid  # the handle
            last_error[0] = 5 if pid == other_user else 87  # ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER
            return None

        def get_exit_code_process(handle, code):
            code._obj.value = 259 if handle == running else 0  # STILL_ACTIVE, or the code it ended with
            return 1

        kernel32 = types.SimpleNamespace(OpenProcess=open_process, GetExitCodeProcess=get_exit_code_process,
                                         CloseHandle=lambda handle: closed.append(handle) or 1)
        kill = mock.Mock(side_effect=AssertionError("os.kill() ends the process on Windows"))
        with mock.patch.object(_store.os, "name", "nt"), mock.patch.object(_store.os, "kill", kill), \
                mock.patch.object(ctypes, "WinDLL", lambda name, use_last_error=False: kernel32, create=True), \
                mock.patch.object(ctypes, "get_last_error", lambda: last_error[0], create=True):
            alive = {p: _store._alive(p) for p in (running, exited, other_user, gone)}
        self.assertEqual(alive, {running: True, exited: False, other_user: True, gone: False})
        kill.assert_not_called()
        self.assertEqual(closed, [running, exited])

    def test_the_summary_announces_every_page_it_will_send_again(self):
        res, _ = self.tr(handler=lambda f, b, n, r: gemini_reply(no_candidates=True, block_reason="SAFETY") if f == "p001.jpg" else ok(f, b, n, r))
        self.assertIn("1 of them will be tried again", self.messages())
        _, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p001.jpg"])

    def test_check_setup_shows_one_line_when_displayed(self):
        out = quiet(am.check_setup, online=False)
        self.assertTrue(repr(out).startswith("<amanuensis setup check:"))
        self.assertIn("checks", out._repr_html_())
        self.assertIsInstance(out[out["status"] == "OK"], pd.DataFrame)
        self.assertFalse(repr(out[out["status"] == "OK"]).startswith("<amanuensis"))


# ------- what the user hands in --------------------------------------------------------------

def _xlsx(path, rows, title="codebook", hidden_rows=(), formats=None, before=()):
    import openpyxl
    wb = openpyxl.Workbook()
    first = wb.active
    for name, state in before:  # sheets placed before the codebook
        ws = wb.create_sheet(name, 0)
        ws.append(["x"])
        ws.sheet_state = state
    ws = first
    ws.title = title
    for r in rows:
        ws.append(r)
    for r in hidden_rows:
        ws.row_dimensions[r].hidden = True
    for ref, f in (formats or {}).items():
        ws[ref].number_format = f
    wb.save(path)


def _enum(cb, var):
    return cb.schema["properties"]["records"]["items"]["properties"][var]["enum"]


CATALAN = [["full", "text", "Número de full imprès.", None, "pàgina"],
           ["cognoms", "text", "Cognoms tal com estan escrits.", None, "fila"],
           ["nom", "text", "El nom de pila.", None, "fila"],
           ["edat", "enter", "Edat en anys.", None, "fila"],
           ["sexe", "categoria", "H o D.", "H; D", "fila"]]


class F03CodebookHeaderRow(Case):
    def test_a_catalan_header_is_read_and_a_data_row_is_never_the_header(self):
        _xlsx("cb.xlsx", [["Nom de la variable", "Tipus", "Descripció", "Valors", "Nivell"]] + CATALAN)
        cb = quiet(am.codebook, "cb.xlsx")
        self.assertEqual([(v.variable, v.type, v.level) for v in cb.vars],
                         [("full", "text", "page"), ("cognoms", "text", "row"), ("nom", "text", "row"),
                          ("edat", "integer", "row"), ("sexe", "category", "row")])

    def test_an_unknown_spelling_of_the_variable_header_stops_instead_of_reading_a_data_row(self):
        _xlsx("cb.xlsx", [["Nom de la var.", "Tipus", "Descripció", "Valors", "Nivell"]] + CATALAN)
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, "cb.xlsx")
        self.assertIn("needs a column called 'variable'", str(e.exception))

    def test_spanish_nombre_de_variable_and_a_title_row_above_a_one_column_header(self):
        _xlsx("cb.xlsx", [["Nombre de variable", "Tipo"], ["nombre", "text"], ["edad", "entero"]])
        self.assertEqual([v.variable for v in quiet(am.codebook, "cb.xlsx").vars], ["nombre", "edad"])
        _xlsx("cb2.xlsx", [["Variables of the 1930 register"], ["variable", "explicació"], ["nom", "x"], ["edat", "y"]])
        self.assertEqual([v.variable for v in quiet(am.codebook, "cb2.xlsx").vars], ["nom", "edat"])


class F14InstructionsPaths(Base):
    def test_a_path_without_its_extension_stops_with_a_hint_and_nothing_is_sent(self):
        os.makedirs("Tesis doctoral")
        with open("Tesis doctoral/instrucciones.md", "w", encoding="utf-8") as f:
            f.write("Pages of a 1930 municipal register.\n")
        fake = FakeGemini("pages", ok)
        with G.mocked(fake), self.assertRaises(am.NotFoundError) as e:
            am.transcribe("pages", "Tesis doctoral/instrucciones", self.cb)
        self.assertIn("Did you mean 'Tesis doctoral/instrucciones.md'?", str(e.exception))
        self.assertEqual(fake.calls, [])
        with self.assertRaises(am.NotFoundError):
            am.instructions("Tesis doctoral/instrucciones")
        with self.assertRaises(am.NotFoundError) as e:
            am.instructions("Tesis doctoral/instrucciones.md ")
        self.assertIn("ends with a space", str(e.exception))

    def test_text_that_cannot_be_a_path_is_still_taken_as_the_instructions(self):
        x = am.instructions("Transcribe every row of the register. Ignore the stamps.")
        self.assertTrue(x.text.startswith("Transcribe every row"))
        with self.assertRaises(am.NotFoundError):
            quiet(am.transcribe, "pages", "Transcribe every row of the register.", self.cb, model="mock")


class F15CodebookExcelDamage(Case):
    def test_a_value_excel_turned_into_a_date_stops(self):
        import datetime as dt
        _xlsx("cb.xlsx", [["variable", "type", "description", "values"],
                          ["grup_edat", "category", "Age band.", "0-4"],
                          [None, None, None, dt.datetime(2026, 5, 9)],
                          [None, None, None, dt.datetime(2026, 10, 14)]])
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, "cb.xlsx")
        self.assertIn("turned some cells of the codebook into dates", str(e.exception))
        self.assertIn("row 3, column 'values': it reads 46151", str(e.exception))

    def test_a_date_in_a_column_the_codebook_does_not_use_is_fine(self):
        import datetime as dt
        _xlsx("cb.xlsx", [["variable", "type", "added on"], ["nom", "text", dt.datetime(2026, 5, 9)]])
        self.assertEqual([v.variable for v in quiet(am.codebook, "cb.xlsx").vars], ["nom"])

    def test_codes_shown_with_a_number_format_warn(self):
        _xlsx("cb.xlsx", [["variable", "type", "values"], ["sexe", "category", 1], [None, None, 2]],
              formats={"C2": "00", "C3": "00"})
        with self.warns("shown in a number format") as w:
            cb = quiet(am.codebook, "cb.xlsx")
        self.assertIn("(Excel shows 01)", str(w[0].message))
        self.assertEqual(_enum(cb, "sexe")[:2], ["1", "2"])


class F16CategoriesWithCatchAll(Case):
    def test_comma_or_slash_separated_values_stop_even_with_a_catch_all(self):
        for values in ("soltero, casado, viudo; otro", "H/M; otro"):
            with self.assertRaises(InputError) as e:
                quiet(am.codebook, pd.DataFrame({"variable": ["estat"], "type": ["category"], "values": [values]}))
            self.assertIn("did you mean", str(e.exception))

    def test_one_plain_value_with_a_catch_all_is_allowed_as_before(self):
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["lloc"], "type": ["category"], "values": ["Barcelona; altres"]}))
        self.assertEqual(_enum(cb, "lloc"), ["Barcelona", "altres", None])


class F17DataFrameCodebook(Case):
    def test_a_dataframe_read_from_the_file_sends_and_hashes_the_same(self):
        _xlsx("cb.xlsx", [["variable", "type", "description", "values", "level"],
                          ["full", "text", "Printed sheet number.", None, "page"],
                          ["sexe", "category", "Code in column Sexe.", 1, "row"],
                          [None, None, None, 2, None]])
        a, b = quiet(am.codebook, "cb.xlsx"), quiet(am.codebook, pd.read_excel("cb.xlsx"))
        self.assertEqual(_enum(b, "sexe"), ["1", "2", "other", None])
        self.assertEqual(a.sha12, b.sha12)

    def test_pandas_missing_values_are_empty_and_variables_round_trip(self):
        df = pd.DataFrame({"variable": ["nom", "edat"], "type": ["text", "integer"], "description": ["Name.", None],
                           "values": [None, None]}).convert_dtypes()
        cb = am.codebook(df)
        self.assertIn("1 of 2 variables has no description", self.messages())
        self.assertNotIn("<NA>", json.dumps(cb.schema))
        self.assertEqual([v.type for v in cb.vars], ["text", "integer"])
        cat = quiet(am.codebook, pd.DataFrame({"variable": ["sexe"], "type": ["category"], "values": ["H; D"]}))
        again = quiet(am.codebook, cat.variables)
        self.assertEqual(again.sha12, cat.sha12)
        self.assertEqual(_enum(again, "sexe"), ["H", "D", "other", None])

    def test_a_date_in_a_dataframe_codebook_stops(self):
        df = pd.DataFrame({"variable": ["grup"], "type": ["category"], "values": [pd.Timestamp("2026-05-09")]})
        with self.assertRaises(InputError):
            quiet(am.codebook, df)


class F31SmallerCodebookProblems(Case):
    def test_a_cover_sheet_first_names_the_sheet_read_and_sheet_(self):
        import openpyxl
        wb = openpyxl.Workbook()
        wb.active.title = "Cover"
        wb.active.append(["Codebook of the 1930 register"])
        ws = wb.create_sheet("vars")
        ws.append(["variable", "type"])
        ws.append(["nom", "text"])
        wb.save("cb.xlsx")
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, "cb.xlsx")
        self.assertIn("read from the sheet 'Cover'", str(e.exception))
        self.assertIn("sheet='vars'", str(e.exception))

    def test_a_hidden_first_sheet_is_not_read(self):
        _xlsx("cb.xlsx", [["variable", "type"], ["nom", "text"]], title="vars", before=[("lookup", "hidden")])
        self.assertEqual([v.variable for v in quiet(am.codebook, "cb.xlsx").vars], ["nom"])

    def test_hidden_rows_warn(self):
        _xlsx("cb.xlsx", [["variable", "type"], ["nom", "text"], ["edat", "integer"], ["ofici_antic", "text"]], hidden_rows=[4])
        with self.warns("hidden in Excel") as w:
            quiet(am.codebook, "cb.xlsx")
        self.assertIn("row 4 ('ofici_antic')", str(w[0].message))

    def test_a_page_variable_called_records_is_refused(self):
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, pd.DataFrame({"variable": ["records", "nom"], "level": ["page", "row"]}))
        self.assertIn("'records' cannot be the name of a page variable", str(e.exception))

    def test_as_written_keeps_a_twin_the_user_already_has(self):
        df = pd.DataFrame({"variable": ["total_as_written", "total"], "type": ["text", "integer"],
                           "level": ["page", "page"], "description": ["Mine, exactly as printed.", "The total."]})
        cb = quiet(am.codebook, df, as_written=True)
        self.assertEqual([v.variable for v in cb.vars], ["total_as_written", "total"])
        self.assertEqual(cb.schema["properties"]["total_as_written"]["description"], "Mine, exactly as printed.")

    def test_a_levels_header_with_page_and_row_is_the_level_column(self):
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["full", "nom"], "type": ["text", "text"], "Levels": ["page", "row"]}))
        self.assertEqual([v.level for v in cb.vars], ["page", "row"])

    def test_nombre_as_a_type_is_a_number_only_under_a_catalan_header(self):
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, pd.DataFrame({"variable": ["vecino"], "tipo": ["nombre"]}))
        self.assertIn("a number in Catalan and a name in Spanish", str(e.exception))
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["edat"], "tipus": ["nombre"]}))
        self.assertEqual(cb.vars[0].type, "number")

    def test_a_dimension_tag_that_understates_the_sheet_does_not_truncate_it(self):
        import re
        import zipfile
        _xlsx("cb.xlsx", [["variable", "type"]] + [[f"v{i}", "text"] for i in range(1, 6)])
        with zipfile.ZipFile("cb.xlsx") as z:
            data = {n: z.read(n) for n in z.namelist()}
        for n in data:
            if n.startswith("xl/worksheets/"):
                data[n] = re.sub(rb'<dimension ref="[^"]*"\s*/>', b'<dimension ref="A1:B2"/>', data[n])
        with zipfile.ZipFile("cut.xlsx", "w") as z:
            for n, d in data.items():
                z.writestr(n, d)
        self.assertEqual(len(quiet(am.codebook, "cut.xlsx").vars), 5)

    def test_xls_and_protected_files_get_their_own_message(self):
        open("cb.xls", "wb").write(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100)
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, "cb.xls")
        self.assertIn("old Excel (.xls) file", str(e.exception))
        open("locked.xlsx", "wb").write(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100)
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, "locked.xlsx")
        self.assertIn("protected by a password or a sensitivity label", str(e.exception))

    def test_a_csv_codebook_with_a_title_line(self):
        with open("cb.csv", "w", encoding="utf-8") as f:
            f.write("Llibre de codis\nvariable;type;values\nnom;text;\nsexe;category;H; D\n".replace("H; D", '"H; D"'))
        cb = quiet(am.codebook, "cb.csv")
        self.assertEqual([v.variable for v in cb.vars], ["nom", "sexe"])
        self.assertEqual(_enum(cb, "sexe"), ["H", "D", "other", None])

    def test_an_nfd_codebook_hashes_as_the_same_nfc_one(self):
        import unicodedata
        rows = {"variable": ["ofici"], "type": ["category"], "values": ["pagès; jornaler"], "description": ["Ofici tal com és escrit."]}
        nfd = {k: [unicodedata.normalize("NFD", v) for v in vs] for k, vs in rows.items()}
        self.assertEqual(quiet(am.codebook, pd.DataFrame(nfd)).sha12, quiet(am.codebook, pd.DataFrame(rows)).sha12)

    def test_codebooks_and_instructions_cannot_drift_from_their_ids(self):
        import dataclasses
        cb = roster_codebook()
        ins = test_instructions()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cb.vars[0].description = "changed"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ins.full = "changed"
        cb.schema["properties"]["heading"]["description"] = "edited in place"
        make_jpegs("pages", 1)
        with self.assertRaises(InputError) as e:
            quiet(am.transcribe, "pages", ins, cb, model="mock")
        self.assertIn("schema was changed after codebook() made it", str(e.exception))

    def test_a_schema_with_a_field_of_the_page_and_of_each_entry_like_a_newspapers_blocks(self):
        s = {"type": "object", "properties": {
            "block_n": {"type": ["integer", "null"]}, "kind": {"type": ["string", "null"]},
            "parts": {"type": "array", "items": {"type": "object", "properties": {
                "part_n": {"type": "integer"}, "kind": {"type": "string", "enum": ["headline", "body"]},
                "text_raw": {"type": "string"}}, "required": ["part_n", "kind", "text_raw"]}}},
            "required": ["block_n", "kind", "parts"]}
        cb = quiet(am.codebook, json=json.dumps(s))
        self.assertEqual([v.variable for v in cb.vars], ["block_n", "kind", "part_n", "parts.kind", "text_raw"])
        make_jpegs("pages", 1)
        data = {"block_n": 3, "kind": "article", "parts": [{"part_n": 1, "kind": "headline", "text_raw": "LA PAZ"},
                                                          {"part_n": 2, "kind": "body", "text_raw": "Ayer..."}]}
        with G.mocked(FakeGemini("pages", lambda f, b, n, r: gemini_reply(data=data))):
            res = quiet(am.transcribe, "pages", test_instructions(), cb)
        self.assertEqual(res["kind"].tolist(), ["article", "article"])
        self.assertEqual(res["parts.kind"].tolist(), ["headline", "body"])
        gold = pd.DataFrame({"file": ["p001.jpg"] * 2, "row": [1, 2], "block_n": [3, None], "kind": ["article", None],
                             "part_n": [1, 2], "parts.kind": ["headline", "body"], "text_raw": ["LA PAZ", "Ayer..."]})
        sc = quiet(am.score, gold, res, report=False)
        self.assertEqual(sc.summary.set_index("measure").loc["values", "matched"], sc.summary.set_index("measure").loc["values", "compared"])


def _docx(path, body_xml):
    import zipfile
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
           + body_xml + '<w:sectPr/></w:body></w:document>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", doc)


def _p(*runs):
    return "<w:p>" + "".join(runs) + "</w:p>"


def _r(text, hidden=False):
    rpr = "<w:rPr><w:vanish/></w:rPr>" if hidden else ""
    return f"<w:r>{rpr}<w:t xml:space=\"preserve\">{text}</w:t></w:r>"


class F32DocxReadAsWordShowsIt(Case):
    def test_a_tracked_move_is_read_once(self):
        _docx("a.docx", _p(_r("# Rules"))
              + '<w:moveFrom w:id="1" w:author="M"><w:p>' + _r("Ignore the stamps of the archive.") + "</w:p></w:moveFrom>"
              + _p(_r("Copy the surnames as written."))
              + '<w:moveTo w:id="2" w:author="M"><w:p>' + _r("Ignore the stamps of the archive.") + "</w:p></w:moveTo>")
        self.assertEqual(I_text("a.docx"), "# Rules\nCopy the surnames as written.\nIgnore the stamps of the archive.")

    def test_hidden_text_is_left_out(self):
        _docx("a.docx", _p(_r("Transcribe every row."), _r(" NOTE TO SELF: old rule.", hidden=True)))
        self.assertEqual(I_text("a.docx"), "Transcribe every row.")

    def test_footnotes_warn(self):
        _docx("a.docx", _p(_r("Use the 1931 list."), '<w:r><w:footnoteReference w:id="1"/></w:r>'))
        with self.warns("footnotes or endnotes, which are not read"):
            I_text("a.docx")

    def test_a_table_is_one_line_per_row_with_its_cells_separated(self):
        def tc(t):
            return "<w:tc><w:tcPr/>" + _p(_r(t)) + "</w:tc>"
        table = ("<w:tbl><w:tblPr/><w:tr>" + tc("abbreviation") + tc("meaning") + "</w:tr>"
                 "<w:tr>" + tc("id.") + tc("the same as above") + "</w:tr></w:tbl>")
        _docx("a.docx", _p(_r("Abbreviations:")) + table + _p(_r("End.")))
        self.assertEqual(I_text("a.docx"), "Abbreviations:\nabbreviation | meaning\nid. | the same as above\nEnd.")


def I_text(path):
    from amanuensis._instructions import read_docx_text
    return read_docx_text(path)


# ------- the benchmark, its editor and score() -------------------------------------------------

def _gold_cb():
    return quiet(am.codebook, pd.DataFrame({
        "variable": ["heading", "name", "age", "sex", "literate"],
        "type": ["text", "text", "integer", "category", "yes/no"],
        "level": ["page", "row", "row", "row", "row"],
        "values": ["", "", "", "male; female", ""],
        "description": ["Page heading.", "Name as written.", "Age.", "Sex.", "Can read."]}))


class GoldBase(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 3)
        self.cb = _gold_cb()
        quiet(am.benchmark, "gold.xlsx", "pages", self.cb, n=2, seed=4)

    def fill(self, cells, pages_done=None, path="gold.xlsx"):
        import openpyxl
        wb = openpyxl.load_workbook(path)
        ws = wb["benchmark"]
        for ref, v in cells.items():
            ws[ref] = v
        for ref, v in (pages_done or {}).items():
            wb["pages"][ref] = v
        wb.save(path)

    def editor(self, path="gold.xlsx", cb=None):
        from amanuensis import _editor as ED
        from amanuensis._images import as_images
        return ED, ED.Editor(ED.editor_state(path, cb or self.cb, as_images("pages")))


def _parse_form(markup):
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.inputs, self.areas, self.selects, self.attrs = {}, {}, {}, set()
            self.area = self.select = None

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            self.attrs |= set(a)
            if tag == "input" and a.get("type") != "checkbox":
                self.inputs[a.get("id")] = a.get("value")
            elif tag == "textarea":
                self.area = a.get("id")
                self.areas[self.area] = ""
            elif tag == "select":
                self.select = a.get("id")
                self.selects[self.select] = ""
            elif tag == "option" and "selected" in a:
                self.selects[self.select] = a.get("value")

        def handle_endtag(self, tag):
            if tag == "textarea":
                self.area = None

        def handle_data(self, data):
            if self.area is not None:
                self.areas[self.area] += data

    p = P()
    p.feed(markup)
    return p


_HARNESS = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const state = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const script = html.split("<script>")[1].split("</script>")[0];
const els = {};
function el(id) { if (!els[id]) els[id] = {id: id, innerHTML: "", textContent: "", className: "", style: {}, value: "100", checked: false}; return els[id]; }
const buttons = ["prev", "next", "save", "finish"].map(function (a) { return {disabled: false, onclick: null, getAttribute: function () { return a; }}; });
const posts = []; let pending = [];
global.document = {getElementById: el, querySelectorAll: function (s) { return s.indexOf("button") === 0 ? buttons : []; }, body: {innerHTML: ""}};
global.fetch = function (url, opts) {
  if (url === "state") return Promise.resolve({json: function () { return Promise.resolve(state); }});
  posts.push({url: url, body: JSON.parse(opts.body)});
  return new Promise(function (res) { pending.push(function () { res({json: function () { return Promise.resolve(state); }}); }); });
};
eval(script);
setTimeout(function () {
  const form = el("form").innerHTML;
  buttons[1].onclick(); buttons[1].onclick();          // a double-click on Next page
  const during = posts.length;
  pending.forEach(function (f) { f(); }); pending = [];
  setTimeout(function () {
    buttons[1].onclick();                              // the second click landing after the re-render
    console.log(JSON.stringify({form: form, during: during, after: posts.length, post: posts[0]}));
  }, 50);
}, 50);
"""


def _node():
    import shutil
    return shutil.which("node")


@unittest.skipUnless(_node(), "needs Node.js to run the editor page's script")
class F04EditorPageEscapesValues(GoldBase):
    def test_quotes_line_breaks_and_model_text_reach_the_form_as_they_are(self):
        import subprocess
        hostile = 'Puig" autofocus onfocus="fetch(\'/action\')'
        self.fill({"C2": "Can\nPuig", "D2": 'Puig "el Vell"', "E2": "34", "F2": "male", "G2": "Sí",
                   "A3": None, "D3": hostile, "G3": "maybe"})
        import openpyxl
        wb = openpyxl.load_workbook("gold.xlsx")
        first = wb["benchmark"]["A2"].value
        wb["benchmark"]["A3"] = first
        wb.save("gold.xlsx")
        ED, ed = self.editor()
        page = ed.route("GET", f"/{ed.token}/")[2].decode("utf-8")
        state = ed.route("GET", f"/{ed.token}/state")[2].decode("utf-8")
        open("page.html", "w", encoding="utf-8").write(page)
        open("state.json", "w", encoding="utf-8").write(state)
        open("harness.js", "w", encoding="utf-8").write(_HARNESS)
        out = subprocess.run([_node(), "harness.js", "page.html", "state.json"], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        got = json.loads(out.stdout)
        form = _parse_form(got["form"])
        self.assertEqual(form.inputs["r_1_1"], 'Puig "el Vell"')
        self.assertEqual(form.inputs["r_2_1"], hostile)
        self.assertNotIn("onfocus", form.attrs)
        self.assertNotIn("autofocus", form.attrs)
        self.assertEqual(form.areas["p_1"], "\nCan\nPuig")  # a browser drops the first line break of a textarea
        self.assertEqual(form.selects["r_1_4"], "yes")      # "Sí" is yes (see F07EditorYesNo)
        self.assertEqual(form.selects["r_2_4"], "maybe")    # not yes or no: shown, never emptied
        # a double-click sends one request, and so does a click right after the re-render
        self.assertEqual((got["during"], got["after"]), (1, 1))
        self.assertEqual(got["post"]["body"]["page"], json.loads(state)["current"])


class F05EditorKeepsWhatTheUserAdded(GoldBase):
    def test_one_save_keeps_a_comment_column_and_a_sheet_the_user_added_and_a_backup(self):
        import openpyxl
        wb = openpyxl.load_workbook("gold.xlsx")
        ws = wb["benchmark"]
        ws.cell(row=1, column=8).value = "comment"
        ws.cell(row=2, column=8).value = "age hard to read, checked with the 1931 list"
        wb.create_sheet("my notes").append(["Archive visit 12 March."])
        wb.save("gold.xlsx")
        ED, ed = self.editor()
        s, msg, kind, _ = ED.handle(ed.state, "save", {"p_1": "Heading 1", "r_1_1": "Anna Puig", "r_1_2": "34", "r_1_3": "female",
                                                       "_done": True})
        self.assertEqual(kind, "ok", msg)
        wb = openpyxl.load_workbook("gold.xlsx")
        self.assertIn("my notes", wb.sheetnames)
        self.assertEqual(wb["my notes"]["A1"].value, "Archive visit 12 March.")
        header = [c.value for c in wb["benchmark"][1]]
        self.assertIn("comment", header)
        col = header.index("comment") + 1
        self.assertEqual(wb["benchmark"].cell(row=2, column=col).value, "age hard to read, checked with the 1931 list")
        self.assertEqual(len([f for f in os.listdir(".") if "-backup-" in f]), 1)
        self.assertIn("a copy of the file as it was", msg[0])
        g = quiet(am.benchmark, "gold.xlsx")
        self.assertEqual(g["name"].tolist(), ["Anna Puig"])

    def test_a_file_saved_in_excel_meanwhile_is_never_written_over(self):
        ED, ed = self.editor()
        self.fill({"C3": "Heading typed in Excel"})
        s, msg, kind, _ = ED.handle(ed.state, "next", {"p_1": "Heading ONE", "r_1_1": "Anna", "r_1_2": "34", "r_1_3": "female"})
        self.assertEqual(kind, "error")
        self.assertIn("changed outside the editor", msg[0])
        import openpyxl
        self.assertEqual(openpyxl.load_workbook("gold.xlsx")["benchmark"]["C3"].value, "Heading typed in Excel")
        self.assertEqual(len([f for f in os.listdir(".") if "-from-editor-" in f]), 1)

    def test_lines_of_a_page_outside_the_pages_sheet_and_lines_without_a_file_name_are_kept(self):
        import openpyxl
        wb = openpyxl.load_workbook("gold.xlsx")
        ws = wb["benchmark"]
        ws["D2"] = "First person"
        ws.insert_rows(3)
        ws["D3"] = "Second person"  # typed without its file name: it belongs to the page above, as the reader takes it
        ws.append(["p999.jpg", None, None, "Someone else"])  # a page the pages sheet does not list
        wb.save("gold.xlsx")
        ED, ed = self.editor()
        self.assertEqual(ED.shown_rows(ed.state), 2)
        ED.handle(ed.state, "save", {"p_1": "H", "r_1_1": "First person", "r_2_1": "Second person", "_done": False})
        names = [r[3] for r in openpyxl.load_workbook("gold.xlsx")["benchmark"].iter_rows(min_row=2, values_only=True)]
        self.assertIn("Someone else", names)
        self.assertIn("Second person", names)

    def test_notes_above_the_first_page_of_a_text_benchmark_are_kept(self):
        quiet(am.benchmark, "g.txt", "pages", am.codebook(free_text=True), n=1, seed=3)
        text = open("g.txt").read()
        with open("g.txt", "w") as f:
            f.write("My note: compare with the 1931 copy.\n" + text)
        from amanuensis import _editor as ED
        st = ED.editor_state("g.txt")
        ED.handle(st, "finish", {"p_1": "Line one", "_done": True})
        self.assertIn("My note: compare with the 1931 copy.", open("g.txt").read())


class F06EditorClickForAnotherPage(GoldBase):
    def test_the_second_click_of_a_double_click_is_refused(self):
        ED, ed = self.editor()
        page1 = {"p_1": "Heading ONE", "r_1_1": "Anna Puig", "r_1_2": "34", "r_1_3": "female"}
        for _ in (1, 2):
            status, _, body = ed.route("POST", f"/{ed.token}/action", json.dumps({"action": "next", "page": 1, "inputs": page1}).encode())
        out = json.loads(body)
        self.assertEqual(out["kind"], "error")
        self.assertIn("meant for page 1", out["message"][0])
        g = quiet(am.benchmark, "gold.xlsx")
        pages = g.gold["pages"]
        self.assertEqual(pages["done"].tolist(), [True, False])
        self.assertEqual(g["heading"].tolist(), ["Heading ONE"])


class F07EditorYesNo(GoldBase):
    def test_every_spelling_the_reader_accepts_is_shown_as_yes_or_no(self):
        from amanuensis import _editor as ED
        import openpyxl
        wb = openpyxl.load_workbook("gold.xlsx")
        ws = wb["benchmark"]
        first = ws["A2"].value
        for r, v in enumerate(["Sí", "S", "t", "oui", "No", "maybe"], start=2):
            ws[f"A{r}"], ws[f"B{r}"], ws[f"D{r}"], ws[f"G{r}"] = first, r - 1, f"P{r}", v
        wb.save("gold.xlsx")
        ED, ed = self.editor()
        view = ED.view(ed.state)
        self.assertEqual([r[3] for r in view["rows"]], ["yes", "yes", "yes", "yes", "no", "maybe"])
        # what the menu then sends is read as the reader reads the file
        inputs = {f"r_{i}_{j}": v for i, r in enumerate(view["rows"], 1) for j, v in enumerate(r, 1)}
        inputs.update({"p_1": "H"})
        s, msg, kind, _ = ED.handle(ed.state, "save", {**inputs, "_done": True})
        self.assertEqual(kind, "error")  # "maybe" is not yes or no: the page is not marked finished
        inputs["r_6_4"] = "no"
        ED.handle(ed.state, "save", {**inputs, "_done": True})
        lit = pd.DataFrame(quiet(am.benchmark, "gold.xlsx"))["literate"].tolist()
        self.assertEqual(lit, [True, True, True, True, False, False])


class F19UnfinishedPagesAreNotScored(GoldBase):
    def test_a_pages_sheet_without_its_file_header_stops(self):
        self.fill({"C2": "Heading 1", "D2": "Anna Puig", "E2": "34", "F2": "female", "D3": "Pe"}, {"B2": "yes", "B3": "no", "A1": "fitxer"})
        with self.assertRaises(InputError) as e:
            quiet(am.benchmark, "gold.xlsx")
        self.assertIn("has no `file` column", str(e.exception))

    def test_an_unknown_status_word_in_a_text_benchmark_stops(self):
        quiet(am.benchmark, "g.txt", "pages", am.codebook(free_text=True), n=3, seed=3)
        text = open("g.txt").read()
        heads = [l for l in text.split("\n") if l.startswith("=== ")]
        for h, st, b in zip(heads, [" done", " not done yet", " to do"], ["Whole first page.", "Half of the sec", "First line"]):
            text = text.replace(h + "\n", h + st + "\n" + b + "\n", 1)
        open("g.txt", "w").write(text)
        with self.assertRaises(InputError) as e:
            quiet(am.benchmark, "g.txt")
        self.assertIn("write done or not done", str(e.exception))
        self.assertIn('says "not done yet"', str(e.exception))


class F20ScoreRefusesASelection(Case):
    def test_a_selection_of_rows_is_refused_and_a_selection_of_the_benchmark_works(self):
        ex = quiet(am.example_project, "ex")
        pages = quiet(am.images, ex.scans)
        res = quiet(am.transcribe, pages, ex.instructions, ex.codebook, model="example")
        gold = quiet(am.benchmark, ex.gold)
        with self.assertRaises(InputError) as e:
            quiet(am.score, gold, res[res.file != "padro1930_p002.jpg"], report=False)
        self.assertIn("this is a selection", str(e.exception))
        part = quiet(am.score, gold[gold.file != "padro1930_p002.jpg"], res, report=False)
        self.assertNotIn("padro1930_p002.jpg", part.by_page.loc[part.by_page.status != "not run", "file"].tolist())


class F21DateCellsStopARewrite(GoldBase):
    def test_a_date_in_the_age_column_stops_extending_the_benchmark(self):
        import datetime as dt
        make_jpegs("pages", 4)
        self.fill({"D2": "Anna Puig", "E2": dt.datetime(2026, 3, 12)}, {"B2": "yes"})
        before = open("gold.xlsx", "rb").read()
        with self.assertRaises(InputError):
            quiet(am.benchmark, "gold.xlsx")
        with self.assertRaises(InputError) as e:
            quiet(am.benchmark, "gold.xlsx", "pages", n=3)
        self.assertIn("cannot be rewritten", str(e.exception))
        self.assertEqual(open("gold.xlsx", "rb").read(), before)


class F22EditorServerAnswersOnlyItsPage(GoldBase):
    def test_other_hosts_origins_content_types_and_addresses_are_refused(self):
        ED, ed = self.editor()
        ed.host = "127.0.0.1:5000"
        base = f"/{ed.token}/"
        mine = {"Host": "127.0.0.1:5000", "Content-Type": "application/json"}
        body = json.dumps({"action": "finish", "page": 1, "inputs": {}}).encode()
        self.assertEqual(ed.route("GET", base + "state", b"", {"Host": "rebind.attacker.example"})[0], 403)
        self.assertEqual(ed.route("POST", base + "action", body, {**mine, "Origin": "https://evil.example"})[0], 403)
        self.assertEqual(ed.route("POST", base + "action", body, {**mine, "Content-Type": "text/plain"})[0], 415)
        self.assertEqual(ed.route("POST", "/action", body, mine)[0], 404)
        self.assertFalse(ed.finished.is_set())
        self.assertEqual(ed.route("GET", base + "state", b"", {"Host": "127.0.0.1:5000"})[0], 200)
        self.assertEqual(ed.route("POST", base + "action", body, {**mine, "Origin": "http://127.0.0.1:5000"})[0], 200)


class _FakeServer:
    def __init__(self, handler):
        import threading
        self.server_address = ("127.0.0.1", 54321)
        self._stop = threading.Event()

    def serve_forever(self):
        self._stop.wait()

    def shutdown(self):
        self._stop.set()

    def server_close(self):
        pass


class F28SmallerEditorProblems(GoldBase):
    def test_without_a_browser_it_says_so_even_quietly_and_stops_waiting(self):
        from amanuensis import _editor as ED
        import time
        t = time.monotonic()
        with self.warns("No browser could be opened") as w, am.options.override(quiet=True):
            ok = ED.run_editor("gold.xlsx", self.cb, server=_FakeServer, opener=lambda url: False, first_wait=0.2)
        self.assertFalse(ok)
        self.assertLess(time.monotonic() - t, 10)
        texts = " ".join(str(x.message) for x in w)
        self.assertIn("was not opened within", texts)
        self.assertIn("app=False", texts)

    def test_a_failed_save_says_to_click_save_again_not_to_run_the_line(self):
        from amanuensis import _gold
        from amanuensis._text import unwritable
        ED, ed = self.editor()
        orig = _gold.book_write

        def locked(book):
            raise unwritable(book["file"])
        _gold.book_write = locked
        try:
            s, msg, kind, _ = ED.handle(ed.state, "save", {"p_1": "H", "r_1_1": "Anna", "_done": False})
        finally:
            _gold.book_write = orig
        self.assertEqual(kind, "error")
        self.assertIn("click Save again", msg[0])
        self.assertNotIn("run the line again", msg[0])

    def test_save_and_finish_keep_a_half_typed_page_out_of_the_score(self):
        ED, ed = self.editor()
        s, msg, kind, fin = ED.handle(ed.state, "finish", {"p_1": "Half", "r_1_1": "An", "_done": False})
        self.assertTrue(fin)
        with self.assertRaises(am.GoldEmptyError):
            quiet(am.benchmark, "gold.xlsx")
        ED, ed = self.editor()
        ED.handle(ed.state, "next", {"p_1": "Whole", "r_1_1": "Anna", "r_1_2": "34", "r_1_3": "female"})
        self.assertEqual(quiet(am.benchmark, "gold.xlsx").gold["pages"]["done"].tolist(), [True, False])
        ed.state["current"] = 0
        ED.handle(ed.state, "save", {"p_1": "Whole", "r_1_1": "Anna", "r_1_2": "34", "r_1_3": "female", "_done": False})
        with self.assertRaises(am.GoldEmptyError):
            quiet(am.benchmark, "gold.xlsx")


class F29ComparedValues(GoldBase):
    def test_a_point_that_may_separate_thousands_warns(self):
        cb = quiet(am.codebook, pd.DataFrame({"variable": ["amount"], "type": ["number"], "level": ["page"]}))
        quiet(am.benchmark, "money.xlsx", "pages", cb, n=1, seed=4)
        self.fill({"B2": "1.250"}, {"B2": "yes"}, path="money.xlsx")
        with self.warns("could be thousands written with a point"):
            g = quiet(am.benchmark, "money.xlsx")
        self.assertEqual(g["amount"].tolist(), [1.25])

    def test_a_long_number_excel_kept_to_15_digits_in_a_text_column_warns(self):
        self.fill({"D2": 1.23456789012346e17, "C2": "H"}, {"B2": "yes"})
        with self.warns("loses digits"):
            quiet(am.benchmark, "gold.xlsx")

    def test_whole_numbers_are_compared_exactly(self):
        from amanuensis._score import score_values
        self.assertNotEqual(score_values([1234567890123456], "integer")[0], score_values([1234567890123457], "integer")[0])
        self.assertEqual(score_values([1234567890123457], "integer")[0], score_values([1234567890123457.0], "integer")[0])
        self.assertEqual(score_values(["1234567890123457"], "integer")[0], ["1234567890123457"])
        from amanuensis._gold import gold_convert
        self.assertEqual(gold_convert(["12345678901234567890"], "integer")["value"], [12345678901234567890])

    def test_a_yes_no_value_the_reader_would_not_take_is_never_false(self):
        from amanuensis._flatten import convert
        self.assertEqual(convert(["S", "sí", "no", "maybe", True, 0], "boolean"), [True, True, False, None, True, False])


class F30DrawWithSemicolons(Case):
    def test_page_names_with_a_semicolon_survive_the_recorded_draw(self):
        from PIL import Image
        os.makedirs("pages")
        names = ["Llibre 1; f. 1.jpg", "Llibre 1; f. 2.jpg", "Llibre 2; f. 1.jpg", "Llibre 2; f. 2.jpg", "Llibre 3; f. 1.jpg"]
        for i, n in enumerate(names):
            Image.new("RGB", (40, 30), (i * 40, 10, 10)).save(os.path.join("pages", n), "JPEG")
        cb = roster_codebook()
        quiet(am.benchmark, "gold.xlsx", "pages", cb, n=2, seed=7)
        quiet(am.benchmark, "gold.xlsx", "pages", n=3)
        import openpyxl
        about = {r[0]: r[1] for r in openpyxl.load_workbook("gold.xlsx")["about"].iter_rows(min_row=2, values_only=True)}
        drawn = json.loads(about["drawn"])
        self.assertEqual(len(drawn), 3)
        self.assertEqual(len(set(drawn)), 3)
        self.assertTrue(all(d in names for d in drawn))
        pages = [r[0] for r in openpyxl.load_workbook("gold.xlsx")["pages"].iter_rows(min_row=2, values_only=True) if r[0]]
        self.assertEqual(sorted(pages), sorted(drawn))

    def test_names_without_a_semicolon_are_recorded_as_the_r_package_writes_them(self):
        from amanuensis._gold import join_drawn, split_drawn
        self.assertEqual(join_drawn(["p001.jpg", "p002.jpg"]), "p001.jpg; p002.jpg")
        self.assertEqual(split_drawn("p001.jpg; p002.jpg"), ["p001.jpg", "p002.jpg"])
        self.assertEqual(split_drawn(join_drawn(["a; b.jpg", "c.jpg"])), ["a; b.jpg", "c.jpg"])


# ------- smaller things ------------------------------------------------------------------------

class F24PatternCheck(Case):
    def test_dollar_is_the_end_of_the_text_as_in_json_schema(self):
        from amanuensis._validate import validate
        s = {"type": "string", "pattern": "^[a-z]+(-[a-z]+)*$"}
        self.assertEqual(validate("fabregas-puig", s), [])
        self.assertEqual(len(validate("fabregas-puig\n", s)), 1)
        self.assertEqual(validate("costs $5", {"type": "string", "pattern": "[$][0-9]"}), [])

    @unittest.skipUnless(hasattr(__import__("signal"), "setitimer"), "needs a timer signal")
    def test_a_pattern_that_backtracks_is_stopped_and_reported_not_hung_on(self):
        import time
        from amanuensis import _validate as V
        old = V.PATTERN_SECONDS
        V.PATTERN_SECONDS = 0.2
        try:
            t = time.monotonic()
            errs = V.validate("Puigferrervidalsolerjoanmarti?", {"type": "string", "pattern": "^([A-Za-zÀ-ÿ]+ ?)+$"})
        finally:
            V.PATTERN_SECONDS = old
        self.assertLess(time.monotonic() - t, 5)
        self.assertEqual(len(errs), 1)
        self.assertIn("could not be checked against the pattern", errs[0])


class F25LargeScans(Case):
    def test_a_scan_above_pillows_limit_is_reduced_not_refused(self):
        from PIL import Image
        from amanuensis import _images as IM
        Image.new("L", (100, 100), 200).save("map.tif", compression="tiff_lzw")
        old_limit, old_target = Image.MAX_IMAGE_PIXELS, IM.TARGET_PIXELS
        Image.MAX_IMAGE_PIXELS, IM.TARGET_PIXELS = 1000, 2500  # Pillow's limit and the target, scaled down
        try:
            pages = quiet(am.images, "map.tif")
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1000)  # put back afterwards
        finally:
            Image.MAX_IMAGE_PIXELS, IM.TARGET_PIXELS = old_limit, old_target
        self.assertEqual(pages["status"].tolist(), ["converted"])
        self.assertEqual((int(pages["width"].iloc[0]), int(pages["height"].iloc[0])), (50, 50))
        self.assertIn("a very large scan", pages["note"].iloc[0])


class F26DpiProvenance(Case):
    def test_a_new_dpi_without_overwrite_is_said_and_the_index_keeps_the_dpi_used(self):
        make_pdf("book.pdf", 2)
        a = quiet(am.images, "book.pdf", dpi=100)
        b = am.images("book.pdf", dpi=300)
        self.assertEqual(a["width"].tolist(), b["width"].tolist())
        self.assertIn("rendered earlier at 100 dpi", self.messages())
        self.assertIn("add overwrite=True", self.messages())
        self.assertNotIn("rendering at 300 dpi makes", self.messages())
        idx = open("book_pages/images.csv").read().splitlines()
        self.assertTrue(all(",100," in line for line in idx[1:]), idx)
        self.assertEqual(b.dpi, 100)
        c = quiet(am.images, "book.pdf", dpi=150, overwrite=True)
        self.assertEqual(c.dpi, 150)
        self.assertTrue(all(",150," in line for line in open("book_pages/images.csv").read().splitlines()[1:]))


@unittest.skipUnless(POSIX, "needs POSIX permissions")
class F27FilePermissions(Base):
    def test_files_get_the_usual_permissions_and_keep_their_own(self):
        old = os.umask(0o022)
        try:
            res, _ = self.tr()
            am.output(res, "out.xlsx")
            am.output(res, "out.csv")
            from PIL import Image
            Image.new("RGB", (40, 30), (9, 9, 9)).save("one.png")
            conv = quiet(am.images, "one.png")
            paths = ["pages/extraction/p001.json", "pages/extraction/contract.json", "out.xlsx", "out.csv", conv["image"].iloc[0]]
            modes = {os.path.basename(p): oct(os.stat(p).st_mode & 0o777) for p in paths}
            self.assertEqual(set(modes.values()), {"0o644"}, modes)
            os.chmod("out.csv", 0o640)
            am.output(res, "out.csv", overwrite=True)
            self.assertEqual(oct(os.stat("out.csv").st_mode & 0o777), "0o640")
        finally:
            os.umask(old)


class F33PdfFromDisk(Case):
    def test_a_pdf_is_rendered_and_its_scans_measured_without_reading_it_whole(self):
        from amanuensis import _images as IM
        make_pdf("book.pdf", 2)
        reads = []  # every read() of book.pdf through Python: the old code read it whole (read(-1))
        import builtins
        real_open = builtins.open

        def watching(path, mode="r", *a, **kw):
            f = real_open(path, mode, *a, **kw)
            if str(path).endswith("book.pdf") and "b" in mode:
                read = f.read

                def tracked(n=-1):
                    reads.append(n)
                    return read(n)
                f.read = tracked
            return f
        builtins.open = watching
        try:
            pages = quiet(am.images, "book.pdf")
        finally:
            builtins.open = real_open
        self.assertEqual(pages["status"].tolist(), ["rendered", "rendered"])
        self.assertNotIn(-1, reads)  # never f.read() of the whole file
        self.assertEqual(IM.pdf_native_dpi("book.pdf", [(300 * 72 / 100, 400 * 72 / 100)]), 100)

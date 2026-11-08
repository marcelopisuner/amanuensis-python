import _thread
import json
import os
import pickle
import time

import httpx
import pandas as pd

import amanuensis as am
from amanuensis import _gemini as G
from amanuensis._errors import ContractError, InputError, Interrupted
from amanuensis._version import __version__

from .helpers import (Case, FakeGemini, error_response, gemini_reply, make_jpegs, quiet, roster_codebook, roster_data,
                      test_instructions)


def ok(f, b, n, r):
    return gemini_reply(data=roster_data(f))


class Base(Case):
    def setUp(self):
        super().setUp()
        self.paths = make_jpegs("pages", 3)
        self.cb = roster_codebook()
        self.ins = test_instructions()

    def tr(self, handler=ok, **kw):
        fake = kw.pop("fake", None) or FakeGemini("pages", handler)
        with G.mocked(fake):
            res = am.transcribe(kw.pop("images", "pages"), kw.pop("instructions", self.ins), kw.pop("codebook", self.cb), **kw)
        return res, fake

    def records(self, folder="pages/extraction"):
        return sorted(f for f in os.listdir(folder) if f.endswith(".json") and f != "contract.json")


class Saving(Base):
    def test_each_page_is_saved_by_default_next_to_the_images_with_the_contract_and_logs(self):
        res, fake = self.tr()
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(self.records(), ["p001.json", "p002.json", "p003.json"])
        rec = json.load(open("pages/extraction/p001.json"))
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["contract"], f"{self.ins.sha12}-{self.cb.sha12}")
        contracts = json.load(open("pages/extraction/contract.json"))["contracts"]
        self.assertEqual(list(contracts), [rec["contract"]])
        self.assertEqual(sorted(f for f in os.listdir("pages/extraction") if not f.endswith(".json")),
                         ["run_manifest.jsonl", "token_usage.txt"])
        self.assertEqual(len(res), 6)
        self.assertEqual(list(res.columns[:4]), ["file", "source", "source_page", "row"])
        self.assertEqual(res.info.folder, os.path.realpath("pages/extraction"))

    def test_a_second_run_sends_nothing_and_returns_the_same_table(self):
        first, _ = self.tr()
        again, fake = self.tr()
        self.assertEqual(fake.calls, [])
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(again))

    def test_with_save_to_false_nothing_is_written_and_the_session_continues(self):
        res, fake = self.tr(save_to=False, limit=2)
        self.assertEqual(sorted(os.listdir("pages")), ["p001.jpg", "p002.jpg", "p003.jpg"])
        self.assertEqual(len(fake.calls), 2)
        again, fake2 = self.tr(save_to=False)
        self.assertEqual(fake2.files_called(), ["p003.jpg"])
        self.assertEqual(again.pages["status"].tolist(), ["ok"] * 3)
        self.assertIsNone(again.info.folder)
        with am.options.override(session=False):
            third, fake3 = self.tr(save_to=False)
        self.assertEqual(len(fake3.calls), 3)

    def test_pages_read_without_a_folder_are_saved_into_one_given_later(self):
        self.tr(save_to=False)
        res, fake = self.tr(save_to="results")
        self.assertEqual(fake.calls, [])
        self.assertEqual(self.records("results"), ["p001.json", "p002.json", "p003.json"])

    def test_continue_from_carries_the_pages_of_an_earlier_result_even_through_a_pickle(self):
        res, _ = self.tr(save_to=False, limit=2)
        data = pickle.dumps(res)
        am.session_reset()
        again, fake = self.tr(save_to=False, continue_from=pickle.loads(data))
        self.assertEqual(fake.files_called(), ["p003.jpg"])
        other = test_instructions("Other instructions.")
        with self.assertRaises(ContractError):
            self.tr(save_to=False, continue_from=res, instructions=other)

    def test_logs_keep_the_manifest_token_usage_and_errors_and_can_be_switched_off(self):
        self.tr(handler=lambda f, b, n, r: error_response(500) if f == "p002.jpg" else ok(f, b, n, r))
        man = [json.loads(l) for l in open("pages/extraction/run_manifest.jsonl")]
        self.assertEqual(man[0]["model"], "gemini-3.8-flash")
        self.assertEqual(man[0]["media_resolution"], "MEDIA_RESOLUTION_ULTRA_HIGH")
        self.assertEqual(man[0]["prompt_sha256"], self.ins.sha12)
        self.assertEqual(man[0]["schema_sha256"], self.cb.sha12)
        self.assertTrue(man[0]["script"].startswith("amanuensis "))
        tok = open("pages/extraction/token_usage.txt").read().splitlines()
        self.assertEqual(tok[0], "p001.jpg\tinput=2500\tthinking=None\toutput=200\ttotal=2700")
        self.assertIn("p002.jpg", open("pages/extraction/errors.log").read())
        self.tr(save_to="nolog", log=False)
        self.assertFalse(os.path.exists("nolog/run_manifest.jsonl"))
        with self.assertRaises(InputError):
            self.tr(save_to=False, log=True)


class Retrying(Base):
    def test_pages_that_failed_on_googles_side_are_retried_next_run(self):
        self.tr(handler=lambda f, b, n, r: error_response(500) if f == "p002.jpg" else ok(f, b, n, r), retries=0)
        res, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p002.jpg"])
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)

    def test_a_reply_cut_off_is_retried_only_with_a_larger_max_tokens(self):
        cut = lambda f, b, n, r: gemini_reply(text='{"x"', finish="MAX_TOKENS") if f == "p001.jpg" else ok(f, b, n, r)
        self.tr(handler=cut, max_tokens=1000)
        res, fake = self.tr(handler=cut, max_tokens=1000)
        self.assertEqual(fake.calls, [])
        self.assertIn("cut off at max_tokens = 1000", self.messages() + str(res.pages["message"].iloc[0]))
        res, fake = self.tr(max_tokens=2000)
        self.assertEqual(fake.files_called(), ["p001.jpg"])

    def test_a_blocked_page_is_retried_once_then_held_until_redo_failed(self):
        blocked = lambda f, b, n, r: gemini_reply(no_candidates=True, block_reason="SAFETY") if f == "p001.jpg" else ok(f, b, n, r)
        self.tr(handler=blocked)
        _, fake = self.tr(handler=blocked)
        self.assertEqual(fake.files_called(), ["p001.jpg"])
        self.assertEqual(json.load(open("pages/extraction/p001.json"))["attempts"], 2)
        _, fake = self.tr(handler=blocked)
        self.assertEqual(fake.calls, [])
        _, fake = self.tr(redo="failed")
        self.assertEqual(fake.files_called(), ["p001.jpg"])

    def test_an_image_google_cannot_use_is_not_retried_but_one_from_another_version_is(self):
        bad = lambda f, b, n, r: error_response(400, "Unable to process input image") if f == "p001.jpg" else ok(f, b, n, r)
        self.tr(handler=bad)
        _, fake = self.tr()
        self.assertEqual(fake.calls, [])
        rec = json.load(open("pages/extraction/p001.json"))
        rec["amanuensis"] = "0.1.0"
        open("pages/extraction/p001.json", "w").write(json.dumps(rec))
        _, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p001.jpg"])

    def test_a_failed_record_stays_small_when_google_quotes_the_request_back(self):
        quote = "Invalid image: " + "QUJD" * 100000
        self.tr(handler=lambda f, b, n, r: error_response(400, quote) if f == "p001.jpg" else ok(f, b, n, r))
        self.assertLess(os.path.getsize("pages/extraction/p001.json"), 5000)

    def test_the_same_image_error_on_three_pages_in_a_row_stops_the_run(self):
        make_jpegs("pages", 6)
        res, fake = self.tr(handler=lambda f, b, n, r: error_response(400, "Invalid image data"))
        self.assertEqual(len(fake.calls), 3)
        self.assertIn("3 pages in a row failed the same way", res.info.stopped)

    def test_five_server_failures_in_a_row_stop_the_run(self):
        make_jpegs("pages", 8)
        res, fake = self.tr(handler=lambda f, b, n, r: error_response(500), retries=0)
        self.assertEqual(len(fake.calls), 5)


class Settings(Base):
    def test_changed_instructions_stop_a_non_interactive_run_naming_both_ways_forward(self):
        self.tr()
        with self.assertRaises(ContractError) as e:
            self.tr(instructions=test_instructions("Different instructions."))
        msg = str(e.exception)
        self.assertIn("instructions: 2 lines changed", msg)
        self.assertIn("save_to=", msg)
        self.assertIn('redo="all"', msg)

    def test_redo_all_moves_the_old_records_to_superseded_and_extracts_again(self):
        self.tr()
        _, fake = self.tr(redo="all")
        self.assertEqual(len(fake.calls), 3)
        sup = os.path.join("pages/extraction", "superseded")
        moved = [f for d in os.listdir(sup) for f in os.listdir(os.path.join(sup, d))]
        self.assertEqual(sorted(moved), ["p001.json", "p002.json", "p003.json"])

    def test_pages_to_check_are_kept_apart_and_redone_with_redo_check(self):
        bad = lambda f, b, n, r: gemini_reply(data={"heading": "x", "records": [{"name": "a", "age": "x", "sex": "male"}]}) if f == "p002.jpg" else ok(f, b, n, r)
        res, _ = self.tr(handler=bad)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "check", "ok"])
        self.assertEqual(sorted(set(res["file"])), ["p001.jpg", "p003.jpg"])
        allrows = am.output(res, include="all")
        self.assertEqual(allrows["page_status"].tolist().count("check"), 1)
        _, fake = self.tr(redo="check")
        self.assertEqual(fake.files_called(), ["p002.jpg"])

    def test_only_picks_pages_by_file_name_stem_or_table(self):
        _, fake = self.tr(only="p002.jpg")
        self.assertEqual(fake.files_called(), ["p002.jpg"])
        _, fake = self.tr(only=["p003"], save_to="b")
        self.assertEqual(fake.files_called(), ["p003.jpg"])
        pages = am._images.as_images("pages")
        _, fake = self.tr(only=pages.iloc[:1], save_to="c")
        self.assertEqual(fake.files_called(), ["p001.jpg"])
        with self.assertRaises(InputError):
            self.tr(only=[1, 2])
        with self.assertRaises(InputError):
            self.tr(only="p009.jpg")

    def test_limit_sends_at_most_that_many_pages_and_limit_0_none(self):
        res, fake = self.tr(limit=2)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "ok", "not run"])
        res, fake = self.tr(limit=0)
        self.assertEqual(fake.calls, [])
        self.assertEqual(len(res), 4)

    def test_budget_stops_the_run_before_going_over_it(self):
        make_jpegs("pages", 10)
        res, fake = self.tr(budget=0.01)
        cost = res.pages["cost_usd"].sum()
        self.assertLessEqual(cost, 0.01)
        self.assertLess(len(fake.calls), 10)
        self.assertIn("budget", res.info.stopped)

    def test_settings_are_checked_before_anything_is_sent(self):
        for kw in ({"resolution": "huge"}, {"thinking": "max"}, {"workers": 9}, {"batch": True, "flex": True},
                   {"budget": -1}, {"max_tokens": 0}, {"api_args": "x"}, {"prices": {"input": 1}},
                   {"model": "gemini-9", "budget": 1}, {"thinking": "minimal"}):
            with self.assertRaises(InputError, msg=str(kw)):
                self.tr(**kw)

    def test_max_tokens_is_capped_at_the_models_limit(self):
        _, fake = self.tr(model="gemini-4-argon", max_tokens=60000)
        self.assertEqual(fake.calls[0]["body"]["generationConfig"]["maxOutputTokens"], 32768)

    def test_an_old_model_name_and_the_model_it_routes_to_count_as_the_same(self):
        self.tr(model="gemini-3.7-flash")
        _, fake = self.tr(model="gemini-3.8-flash")
        self.assertEqual(fake.calls, [])

    def test_unknown_models_run_without_a_cost(self):
        res, _ = self.tr(model="gemini-9-future")
        self.assertTrue(res.pages["cost_usd"].isna().all())

    def test_a_page_image_that_changed_is_extracted_again(self):
        self.tr()
        from PIL import Image
        Image.new("RGB", (61, 40), (1, 2, 3)).save("pages/p002.jpg", "JPEG", quality=80)
        _, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p002.jpg"])

    def test_practice_results_are_extracted_again_for_real_and_never_mixed(self):
        quiet(am.transcribe, "pages", self.ins, self.cb, model="mock")
        res, fake = self.tr(limit=1)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "not run", "not run"])
        self.assertEqual(len(res), 2)

    def test_results_kept_apart_in_a_second_folder_are_continued_there(self):
        self.tr()
        other = test_instructions("Other instructions.")
        self.tr(instructions=other, save_to="pages/extraction_2", limit=1)
        _, fake = self.tr(instructions=other)
        self.assertEqual(fake.files_called(), ["p002.jpg", "p003.jpg"])
        self.assertIn("Continuing in", self.messages())

    def test_images_can_be_given_as_images_output_a_folder_or_files(self):
        _, fake = self.tr(images=self.paths[:2], save_to="a")
        self.assertEqual(len(fake.calls), 2)
        _, fake = self.tr(images=quiet(am.images, "pages"), save_to="b")
        self.assertEqual(len(fake.calls), 3)

    def test_free_text_by_default_announced(self):
        res, fake = self.tr(handler=lambda f, b, n, r: gemini_reply(data={"text": f"Text of {f}"}), codebook=None)
        self.assertIn("transcribe each page as text", self.messages())
        self.assertEqual(res["text"].tolist(), ["Text of p001.jpg", "Text of p002.jpg", "Text of p003.jpg"])

    def test_batch_needs_a_folder_and_a_long_run_without_one_suggests_it(self):
        with self.assertRaises(InputError) as e:
            self.tr(batch=True, save_to=False)
        self.assertIn("batch", str(e.exception))
        make_jpegs("pages", 4)
        with am.options.override(long_run=3):
            fake = FakeGemini("pages", ok)
            with G.mocked(fake):
                am.transcribe("pages", self.ins, self.cb, save_to=False)
        self.assertIn("A long run", self.messages())


class Parallel(Base):
    def test_workers_send_pages_in_parallel_rounds_retry_transient_failures_and_slow_down(self):
        make_jpegs("pages", 6)
        res, fake = self.tr(handler=lambda f, b, n, r: error_response(429) if (f == "p002.jpg" and n == 1) else ok(f, b, n, r), workers=3)
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 6)
        self.assertEqual(fake.count["p002.jpg"], 2)
        self.assertIn("slow down", self.messages())

    def test_in_a_parallel_round_a_stop_keeps_the_replies_already_received(self):
        res, fake = self.tr(handler=lambda f, b, n, r: error_response(400, "API key not valid") if f == "p002.jpg" else ok(f, b, n, r), workers=3)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "not run", "ok"])


class Stopping(Base):
    def test_ctrl_c_keeps_the_pages_read_and_the_same_line_reads_only_the_rest(self):
        def handler(f, b, n, r):
            if f == "p002.jpg":
                raise KeyboardInterrupt
            return ok(f, b, n, r)
        with self.assertRaises(Interrupted) as e:
            self.tr(handler=handler)
        self.assertIsInstance(e.exception, KeyboardInterrupt)
        part = e.exception.result
        self.assertEqual(part.pages["status"].tolist(), ["ok", "not run", "not run"])
        self.assertEqual(self.records(), ["p001.json"])
        _, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p002.jpg", "p003.jpg"])

    def test_ctrl_c_without_a_folder_is_kept_for_the_session(self):
        def handler(f, b, n, r):
            if f == "p003.jpg":
                raise KeyboardInterrupt
            return ok(f, b, n, r)
        with self.assertRaises(Interrupted):
            self.tr(handler=handler, save_to=False)
        _, fake = self.tr(save_to=False)
        self.assertEqual(fake.files_called(), ["p003.jpg"])

    def test_ctrl_c_during_a_parallel_round_keeps_that_rounds_replies(self):
        make_jpegs("pages", 6)

        def handler(f, b, n, r):
            if f == "p002.jpg":
                _thread.interrupt_main()
                time.sleep(0.2)
            return ok(f, b, n, r)
        with self.assertRaises(Interrupted) as e:
            self.tr(handler=handler, workers=3)
        st = e.exception.result.pages["status"].tolist()
        self.assertEqual(st[:3], ["ok", "ok", "ok"])
        self.assertEqual(st[3:], ["not run"] * 3)

    def test_an_unexpected_error_keeps_the_pages_already_read_and_is_raised(self):
        import amanuensis._transcribe as TR
        orig = TR.record_from_reply
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("disk full")
            return orig(*a, **kw)
        TR.record_from_reply = flaky
        try:
            with self.assertRaises(RuntimeError) as e:
                self.tr()
        finally:
            TR.record_from_reply = orig
        self.assertIn("Please report the error", self.messages())
        # the reply that could not be processed was paid for: it is kept, with
        # its cost, and not sent again by this version
        self.assertEqual(self.records(), ["p001.json", "p002.json"])
        rec = json.load(open("pages/extraction/p002.json"))
        self.assertEqual((rec["status"], rec["problem"]), ("failed", "internal"))
        self.assertIn("Heading of p002.jpg", rec["reply"])
        self.assertGreater(rec["cost_usd"], 0)
        self.assertEqual(e.exception.result.pages["status"].tolist(), ["ok", "failed", "not run"])
        _, fake = self.tr()
        self.assertEqual(fake.files_called(), ["p003.jpg"])
        self.assertIn('use redo="failed" to try again', self.messages())


class Practice(Case):
    def test_mock_needs_no_key_and_says_it_is_practice(self):
        make_jpegs("pages", 2)
        os.environ.pop("GEMINI_API_KEY")
        res = am.transcribe("pages", test_instructions(), roster_codebook(), model="mock")
        self.assertEqual(res.pages["status"].tolist(), ["ok", "ok"])
        self.assertIn("practice model: nothing was charged", self.messages())
        self.assertTrue(os.path.exists("pages/extraction/p001.json"))

    def test_the_example_model_knows_only_the_example_pages(self):
        make_jpegs("pages", 1)
        with self.assertRaises(InputError):
            quiet(am.transcribe, "pages", model="example")


class Interactive(Base):
    def answers(self, *replies):
        import builtins
        it = iter(replies)
        orig = builtins.input
        builtins.input = lambda prompt="": next(it)
        self.addCleanup(setattr, builtins, "input", orig)

    def test_changed_settings_offer_a_menu_keep_apart_redo_or_stop(self):
        self.tr()
        other = test_instructions("Different instructions.")
        with am.options.override(interactive=True, ask=True):
            self.answers("1")
            res, fake = self.tr(instructions=other)
            self.assertEqual(res.info.folder, os.path.realpath("pages/extraction_2"))
            self.assertEqual(len(fake.calls), 3)
            self.answers("3")
            with self.assertRaises(ContractError):
                self.tr(instructions=test_instructions("Third version."))
            self.answers("2")
            res, fake = self.tr(instructions=test_instructions("Third version."))
            self.assertEqual(res.info.folder, os.path.realpath("pages/extraction"))
            self.assertEqual(len(fake.calls), 3)
        self.assertTrue(os.path.isdir("pages/extraction/superseded"))

    def test_with_25_pages_the_projected_cost_is_shown_after_two_and_a_no_stops(self):
        make_jpegs("pages", 26)
        with am.options.override(interactive=True, ask=True):
            self.answers("n")
            res, fake = self.tr()
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("Stopped by you after the first pages", res.info.stopped)


class Partial(Base):
    def test_redo_stopped_by_the_budget_keeps_every_old_result_until_it_is_replaced(self):
        make_jpegs("pages", 6)
        self.tr()
        res, fake = self.tr(redo="all", budget=0.008)
        self.assertLess(len(fake.calls), 6)
        self.assertEqual(len(self.records()), 6)
        self.assertEqual(res.pages["status"].tolist().count("ok"), 6)

    def test_after_a_codebook_change_pages_not_yet_redone_are_left_out_of_the_table(self):
        self.tr()
        cb2 = quiet(am.codebook, pd.DataFrame({"variable": ["heading", "name", "age", "sex", "parish"],
                                               "type": ["text", "text", "integer", "category", "text"],
                                               "level": ["page", "row", "row", "row", "row"],
                                               "values": ["", "", "", "male; female", ""]}))
        def with_parish(f, b, n, r):
            d = roster_data(f)
            for rec in d["records"]:
                rec["parish"] = "Sant Pere"
            return gemini_reply(data=d)
        res, fake = self.tr(handler=with_parish, codebook=cb2, redo="all", limit=1)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "not run", "not run"])
        self.assertIn("other settings", res.pages["message"].iloc[1])
        self.assertEqual(set(res["file"]), {"p001.jpg"})

    def test_the_summary_says_what_this_run_did(self):
        self.tr(handler=lambda f, b, n, r: error_response(400, "API key not valid"))
        self.assertIn("x No page was read in this run.", self.messages())
        self.clear_messages()
        self.tr()
        self.assertIn("v Sent 3 pages to gemini-3.8-flash", self.messages())
        self.clear_messages()
        self.tr()
        self.assertIn("Nothing to send: every page is already done.", self.messages())

import json
import os

import httpx

import amanuensis as am
from amanuensis import _batch as B
from amanuensis import _gemini as G
from amanuensis import _store
from amanuensis._errors import ApiError, InputError

from .helpers import Case, FakeBatch, error_response, gemini_reply, make_jpegs, quiet, roster_codebook, roster_data, test_instructions


class Base(Case):
    def setUp(self):
        super().setUp()
        make_jpegs("pages", 3)
        self.cb = roster_codebook()
        self.ins = test_instructions()

    def tr(self, fake=None, **kw):
        if fake is None:
            return am.transcribe("pages", self.ins, self.cb, batch=True, **kw)
        with G.mocked(fake):
            return am.transcribe("pages", self.ins, self.cb, batch=True, **kw)

    def jobs(self, folder="pages/extraction"):
        return _store.read_jobs(folder)


class MockCycle(Base):
    def test_the_mock_model_runs_the_whole_batch_cycle(self):
        res = self.tr(model="mock")
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertEqual(res.pages["engine"].tolist(), ["batch"] * 3)
        self.assertEqual(self.jobs(), [])
        self.assertIn("Collected batch job batches/mock-1: 3 of 3 pages returned", self.messages())
        rec = json.load(open("pages/extraction/p001.json"))
        self.assertEqual(rec["batch_job"], "batches/mock-1")
        self.assertEqual(rec["settings"]["schema_channel"], "openapi")

    def test_a_job_sent_with_wait_false_is_collected_by_a_later_call_even_without_batch(self):
        res = self.tr(model="mock", wait=False)
        self.assertIn("Run the same line again later to collect", self.messages())
        self.assertEqual(res.pages["status"].tolist(), ["not run"] * 3)
        jobs = self.jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["state"], "submitted")
        self.assertEqual([p["file"] for p in jobs[0]["pages"]], ["p001.jpg", "p002.jpg", "p003.jpg"])
        self.assertEqual(jobs[0]["contract"], f"{self.ins.sha12}-{self.cb.sha12}")
        # still running: its pages are not sent again
        self.clear_messages()
        res = am.transcribe("pages", self.ins, self.cb, model="mock")
        self.assertIn("1 batch job recorded", self.messages())
        self.assertIn("3 in a pending batch job", self.messages())
        self.assertFalse(os.path.exists("pages/extraction/p001.json"))
        # finished: collected, and stamped as batch results
        res = am.transcribe("pages", self.ins, self.cb, model="mock")
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertEqual(res.pages["engine"].tolist(), ["batch"] * 3)
        self.assertIn("Collected 3 pages from a batch job sent earlier", self.messages())
        self.assertFalse(os.path.exists("pages/extraction/batch-jobs.json"))

    def test_batch_size_splits_the_pages_into_several_jobs(self):
        self.tr(model="mock", batch_size=2)
        self.assertIn("Batch job 2 created", self.messages())

    def test_practice_jobs_never_reach_the_network(self):
        res = self.tr(model="mock")
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)


class RealModel(Base):
    def test_requests_are_snake_case_lines_keyed_by_file_and_the_input_file_is_deleted(self):
        fake = FakeBatch()
        self.tr(fake)
        lines = [json.loads(l) for l in fake.uploads["files/input-1"].split("\n") if l]
        self.assertEqual([l["key"] for l in lines], ["p001.jpg", "p002.jpg", "p003.jpg"])
        req = lines[0]["request"]
        self.assertIn("inline_data", req["contents"][0]["parts"][0])
        self.assertIn("response_schema", req["generation_config"])
        self.assertEqual(req["generation_config"]["response_schema"]["type"], "OBJECT")
        self.assertEqual(fake.deleted, ["files/input-1"])
        self.assertEqual(fake.created[0]["batch"]["displayName"][:11], "amanuensis-")

    def test_the_json_schema_channel_can_be_chosen(self):
        fake = FakeBatch()
        with am.options.override(batch_schema="json"):
            self.tr(fake)
        req = json.loads(fake.uploads["files/input-1"].split("\n")[0])["request"]
        self.assertEqual(req["generation_config"]["response_json_schema"], self.cb.schema)

    def test_a_json_schema_you_wrote_goes_to_batch_as_it_is(self):
        cb = quiet(am.codebook, json=json.dumps({"type": "object", "properties": {"t": {"type": ["string", "null"]}}}))
        fake = FakeBatch()
        with G.mocked(fake):
            am.transcribe("pages", self.ins, cb, batch=True)
        req = json.loads(fake.uploads["files/input-1"].split("\n")[0])["request"]
        self.assertEqual(req["generation_config"]["response_json_schema"], cb.schema)

    def test_results_are_matched_by_key_when_they_come_back_in_another_order(self):
        fake = FakeBatch(shuffle=True, reply=lambda k, req: {"key": k, "response": gemini_reply(data=roster_data(k))})
        res = self.tr(fake)
        self.assertEqual(res.loc[res["row"] == 1, "heading"].tolist(), ["Heading of p001.jpg", "Heading of p002.jpg", "Heading of p003.jpg"])

    def test_a_per_line_error_fails_that_page_only_and_it_is_sent_again(self):
        fake = FakeBatch(reply=lambda k, req: {"key": k, "error": {"message": "boom"}} if k == "p002.jpg" else None)
        res = self.tr(fake)
        self.assertEqual(res.pages["status"].tolist(), ["ok", "failed", "ok"])
        fake2 = FakeBatch()
        self.tr(fake2)
        lines = [json.loads(l)["key"] for l in fake2.uploads["files/input-1"].split("\n") if l]
        self.assertEqual(lines, ["p002.jpg"])

    def test_an_expired_or_failed_job_turns_its_pages_into_failed_records(self):
        for state in ("EXPIRED", "FAILED"):
            res = self.tr(FakeBatch(states=[state]), save_to=f"r_{state}")
            self.assertEqual(res.pages["status"].tolist(), ["failed"] * 3)
            self.assertEqual(res.pages["problem"].tolist(), ["batch"] * 3)

    def test_a_job_is_waited_for_until_it_ends(self):
        fake = FakeBatch(states=["RUNNING", "RUNNING", "SUCCEEDED"])
        res = self.tr(fake)
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertEqual(fake.polls, 3)

    def test_a_full_batch_queue_stops_submitting_and_removes_the_job_entry(self):
        fake = FakeBatch(create_error=error_response(429, "Too many requests"))
        res = self.tr(fake)
        self.assertIn("batch queue for your account is full", res.info.stopped)
        self.assertEqual(self.jobs(), [])
        self.assertEqual(fake.deleted, ["files/input-1"])

    def test_an_unclear_answer_at_creation_keeps_the_job_and_the_next_run_finds_it(self):
        fake = FakeBatch(create_error=error_response(500, "internal"))
        with self.assertRaises(ApiError):
            self.tr(fake)
        jobs = self.jobs()
        self.assertEqual(len(jobs), 1)
        self.assertIsNone(jobs[0]["name"])
        fake2 = FakeBatch()
        fake2.display = jobs[0]["display_name"]
        fake2.input = jobs[0]["input_file"]
        fake2.uploads[jobs[0]["input_file"]] = fake.uploads["files/input-1"]
        res = self.tr(fake2)
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertEqual(fake2.created, [])
        self.assertIn("Found batch job batches/test-1", self.messages())

    def test_a_submitting_entry_google_never_created_is_dropped_and_its_pages_sent(self):
        _store.write_jobs(os.path.realpath("pages/extraction") if os.path.isdir("pages/extraction") else (os.makedirs("pages/extraction") or os.path.realpath("pages/extraction")),
                          [{"name": None, "display_name": "amanuensis-lost", "pages": [{"file": "p001.jpg"}], "settings": {"model": "gemini-3.8-flash"}, "input_file": "files/old"}])
        fake = FakeBatch()
        res = self.tr(fake)
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertIn("Google has no record of it", self.messages())
        self.assertIn("files/old", fake.deleted)

    def test_when_google_cannot_be_asked_the_entry_stays_and_its_pages_wait(self):
        os.makedirs("pages/extraction")
        _store.write_jobs(os.path.realpath("pages/extraction"),
                          [{"name": None, "display_name": "amanuensis-lost", "pages": [{"file": "p001.jpg"}], "settings": {"model": "gemini-3.8-flash"}}])
        fake = FakeBatch(list_error=error_response(503, "down"))
        res = self.tr(fake)
        self.assertEqual(res.pages["status"].tolist()[0], "not run")
        self.assertEqual(len(self.jobs()), 1)

    def test_google_forgetting_a_job_older_than_its_retention_means_its_pages_are_sent_again(self):
        os.makedirs("pages/extraction")
        _store.write_jobs(os.path.realpath("pages/extraction"),
                          [{"name": "batches/gone", "display_name": "x", "submitted": "2026-01-02T10:00:00Z",
                            "pages": [{"file": "p001.jpg"}], "settings": {"model": "gemini-3.8-flash"}}])
        fake = FakeBatch()
        res = self.tr(fake)
        self.assertIn("no longer knows batch job batches/gone", self.messages())
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)

    def test_a_404_for_a_job_sent_recently_keeps_the_job_and_never_sends_its_pages_again(self):
        # another project's key (or a passing 404) once dropped the job and
        # paid for its pages a second time
        quiet(self.tr, FakeBatch(), wait=False)

        class OtherProject(FakeBatch):
            def __call__(self, req):
                if req.method == "GET" and str(req.url).split("?")[0].endswith("/v1beta/batches/test-1"):
                    return error_response(404, "Requested entity was not found.")
                return super().__call__(req)

        other = OtherProject()
        res = self.tr(other, wait=False)
        self.assertEqual(other.created, [])
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual(res.pages["status"].tolist(), ["not run"] * 3)
        self.assertIn("may now hold a key of another Google project", self.messages())
        self.assertNotIn("its pages will be sent again", self.messages())
        # with the right key again, the job is collected and nothing is paid twice
        res = self.tr(FakeBatch())
        self.assertEqual(res.pages["status"].tolist(), ["ok"] * 3)
        self.assertEqual(self.jobs(), [])

    def test_a_404_is_said_once_while_waiting_for_another_job(self):
        os.makedirs("pages/extraction")
        _store.write_jobs(os.path.realpath("pages/extraction"),
                          [{"name": "batches/elsewhere", "display_name": "x", "submitted": _store.utc_now(),
                            "pages": [{"file": "p001.jpg"}], "settings": {"model": "gemini-3.8-flash"}}])

        class Unknown(FakeBatch):
            def __call__(self, req):
                if str(req.url).split("?")[0].endswith("/v1beta/batches/elsewhere"):
                    return error_response(404, "Requested entity was not found.")
                return super().__call__(req)

        res = self.tr(Unknown(states=("RUNNING", "SUCCEEDED")))
        self.assertEqual(self.messages().count("Google does not know batch job batches/elsewhere"), 1)
        self.assertEqual(res.pages["status"].tolist(), ["not run", "ok", "ok"])
        self.assertEqual([j["name"] for j in self.jobs()], ["batches/elsewhere"])

    def test_two_jobs_made_in_the_same_second_get_different_display_names(self):
        from amanuensis._transcribe import Run
        run = Run({}, None, "/some/folder", None, None, "k")
        self.assertNotEqual(B.job_display_name(run, 1), B.job_display_name(run, 1))


class Budget(Base):
    def test_before_any_page_is_measured_the_budget_covers_the_most_the_batch_could_cost(self):
        with self.assertRaises(InputError) as e:
            self.tr(FakeBatch(), budget=0.05)
        self.assertIn("could cost up to", str(e.exception))
        self.assertEqual(self.jobs(), [])

    def test_a_batch_that_would_cost_more_than_measured_pages_allow_is_not_sent(self):
        est = {"cost": 0.5, "ceiling": None, "measured": True, "note": ", measured on earlier pages"}
        with self.assertRaises(InputError) as e:
            B.check_batch_budget(est, 0.1)
        self.assertIn("more than your budget", str(e.exception))
        B.check_batch_budget(est, 1.0)

    def test_batch_estimate_measures_earlier_pages_when_it_can(self):
        from amanuensis._transcribe import Run
        pages = am._images.as_images("pages")
        con = _store.contract(self.ins, self.cb)
        s = {"model": "gemini-3.8-flash", "resolution": "ultra_high", "max_tokens": 1000, "prices": None}
        run = Run(s, con, None, None, pages, "k")
        est = B.batch_estimate(run, 10)
        self.assertFalse(est["measured"])
        self.assertGreater(est["ceiling"], est["cost"])
        run.records = {k: {"model": "gemini-3.8-flash", "tokens": {"input": 1000, "billed_output": 1000}} for k in ("a", "b")}
        est = B.batch_estimate(run, 10)
        self.assertTrue(est["measured"])
        self.assertAlmostEqual(est["cost"], 10 * (1000 * 0.375 + 1000 * 1.875) / 1e6)

    def test_a_batch_that_costs_more_than_its_budget_says_so(self):
        job = {"name": "batches/x", "budget": {"total": 1.0, "share": 0.5}}
        with self.warns("more than its share of your budget"):
            B.warn_over_budget(job, 0.6)

    def test_prices_stored_without_names_still_give_a_cost(self):
        self.assertEqual(B.job_prices([1, 2]), {"input": 1, "output": 2})
        self.assertEqual(B.job_prices({"input": 1, "output": 2}), {"input": 1, "output": 2})
        self.assertIsNone(B.job_prices([1]))

    def test_your_own_prices_are_used_for_every_page(self):
        res = self.tr(FakeBatch(), prices={"input": 2.0, "output": 4.0})
        rec = json.load(open("pages/extraction/p001.json"))
        self.assertEqual(rec["price"]["input"], 1.0)
        self.assertFalse(res.pages["cost_usd"].isna().any())


class Sizes(Base):
    def test_pages_are_split_into_jobs_whose_files_stay_under_the_limit(self):
        from amanuensis._transcribe import Run
        pages = am._images.as_images("pages")
        run = Run({"model": "m"}, _store.contract(self.ins, self.cb), None, None, pages, "k")
        size = os.path.getsize("pages/p001.jpg")
        with am.options.override(batch_max_bytes=2 * (size * 4 // 3 + 4000) + 100):
            chunks = B.batch_chunks(run, [0, 1, 2], 500)
        self.assertEqual([len(c) for c in chunks], [2, 1])

    def test_a_batch_file_over_the_limit_is_refused_before_anything_is_recorded(self):
        with am.options.override(batch_max_bytes=10):
            with self.assertRaises(InputError):
                self.tr(FakeBatch())
        self.assertEqual(self.jobs(), [])


class Cancel(Base):
    def test_cancel_batch_cancels_recorded_jobs_and_their_pages_are_sent_again(self):
        fake = FakeBatch(states=["RUNNING"])
        with G.mocked(fake):
            am.transcribe("pages", self.ins, self.cb, batch=True, wait=False)
            done = am.cancel_batch("pages/extraction")
        self.assertEqual(done, ["batches/test-1"])
        self.assertEqual(fake.cancelled, ["batches/test-1"])
        fake.states = ["CANCELLED"]
        with G.mocked(fake):
            res = am.transcribe("pages", self.ins, self.cb)
        self.assertEqual(res.pages["status"].tolist(), ["failed"] * 3)
        self.assertIn("cancelled", res.pages["message"].iloc[0])

    def test_cancel_batch_with_nothing_pending(self):
        os.makedirs("empty")
        self.assertEqual(am.cancel_batch("empty"), [])
        with self.assertRaises(InputError):
            am.cancel_batch("nope")

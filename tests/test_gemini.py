"""The Gemini layer: what google-genai is asked to send, the package's own
retry loop, what each failure means, and how a reply becomes a record."""

import json
import os

import httpx

import amanuensis as am
from amanuensis import _gemini as G
from amanuensis import _transcribe as TR
from amanuensis._errors import MissingKeyError, NetworkError

from .helpers import Case, FakeGemini, error_response, gemini_reply, make_jpegs, quiet, roster_codebook, roster_data, test_instructions


def run(images, handler, **kw):
    fake = FakeGemini(images, handler)
    with G.mocked(fake):
        res = quiet(am.transcribe, images, kw.pop("instructions", test_instructions()), kw.pop("codebook", roster_codebook()), **kw)
    return res, fake


class RequestBodies(Case):
    def test_the_sync_body_puts_the_image_first_then_the_text_with_the_schema_and_settings(self):
        make_jpegs("pages", 1)
        res, fake = run("pages", lambda f, b, n, r: gemini_reply(data=roster_data(f)))
        body = fake.calls[0]["body"]
        self.assertEqual(list(body), ["contents", "generationConfig"])
        parts = body["contents"][0]["parts"]
        self.assertIn("inlineData", parts[0])
        self.assertEqual(parts[0]["mediaResolution"], {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"})
        self.assertTrue(parts[1]["text"].startswith("These are pages of a 1930 municipal register."))
        gc = body["generationConfig"]
        self.assertEqual(gc["responseMimeType"], "application/json")
        self.assertEqual(gc["responseJsonSchema"], roster_codebook().schema)
        self.assertEqual(gc["maxOutputTokens"], 32768)
        self.assertEqual(gc["thinkingConfig"], {"include_thoughts": False, "thinking_level": "LOW"})
        self.assertNotIn("mediaResolution", gc)
        self.assertNotIn("temperature", gc)
        self.assertNotIn("systemInstruction", body)
        self.assertEqual(fake.calls[0]["headers"]["x-goog-api-key"], os.environ["GEMINI_API_KEY"])
        self.assertIn("models/gemini-3.8-flash:generateContent", fake.calls[0]["url"])

    def test_other_resolutions_go_in_the_config_temperature_only_when_given_flex_sets_the_tier(self):
        make_jpegs("pages", 1)
        res, fake = run("pages", None, resolution="high", temperature=0.2, flex=True,
                        api_args={"generationConfig": {"seed": 7}})
        body = fake.calls[0]["body"]
        self.assertNotIn("mediaResolution", body["contents"][0]["parts"][0])
        self.assertEqual(body["generationConfig"]["mediaResolution"], "MEDIA_RESOLUTION_HIGH")
        self.assertEqual(body["generationConfig"]["temperature"], 0.2)
        self.assertEqual(body["generationConfig"]["seed"], 7)
        self.assertEqual(body["serviceTier"], "flex")

    def test_the_image_reaches_google_byte_for_byte(self):
        p = make_jpegs("pages", 1)[0]
        res, fake = run("pages", None)
        self.assertEqual(fake.calls[0]["file"], "p001.jpg")

    def test_the_batch_body_is_snake_case_with_the_same_content(self):
        s = {"resolution": "ultra_high", "max_tokens": 100, "thinking_level": "LOW", "schema_channel": "openapi",
             "api_args": {"generationConfig": {"responseSchema": {"x": 1}, "topK": 3}}}
        body = G.batch_body("Read.", "QUJD", "image/jpeg", roster_codebook().schema, s)
        part = body["contents"][0]["parts"][0]
        self.assertEqual(part, {"inline_data": {"mime_type": "image/jpeg", "data": "QUJD"}, "media_resolution": {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}})
        gc = body["generation_config"]
        # api_args are deep-merged last, as R's modifyList does
        self.assertEqual(gc["response_schema"]["x"], 1)
        self.assertEqual(gc["response_schema"]["type"], "OBJECT")
        self.assertEqual(gc["top_k"], 3)
        self.assertEqual(gc["thinking_config"], {"thinking_level": "LOW", "include_thoughts": False})

    def test_the_openapi_channel_keeps_enum_nullable_and_property_order(self):
        out = G.to_response_schema(roster_codebook().schema)
        self.assertEqual(out["type"], "OBJECT")
        self.assertEqual(out["propertyOrdering"], ["heading", "records"])
        sex = out["properties"]["records"]["items"]["properties"]["sex"]
        self.assertEqual(sex, {"description": sex["description"], "type": "STRING", "nullable": True, "enum": ["male", "female", "other"]})

    def test_to_response_schema_resolves_refs_and_gives_every_node_a_type(self):
        s = {"$defs": {"n": {"type": "integer", "minimum": 1}}, "type": "object",
             "properties": {"a": {"$ref": "#/$defs/n"}, "b": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "c": {"const": "x"}, "d": {"allOf": [{"properties": {"p": {"type": "string"}}}, {"required": ["p"]}]},
                            "e": {"properties": {}}, "f": {}}}
        out = G.to_response_schema(s)
        p = out["properties"]
        self.assertEqual(p["a"], {"type": "INTEGER", "minimum": 1})
        self.assertEqual(p["b"], {"type": "STRING", "nullable": True})
        self.assertEqual(p["c"], {"type": "STRING", "enum": ["x"]})
        self.assertEqual(p["d"]["required"], ["p"])
        self.assertEqual(p["e"], {"properties": {}, "propertyOrdering": [], "type": "OBJECT"})
        self.assertEqual(p["f"], {"type": "STRING"})

    def test_snake_keys_converts_field_names_only(self):
        x = G.snake_keys({"generationConfig": {"responseJsonSchema": {"properties": {"firstName": {}}}, "topP": 1}})
        self.assertEqual(x, {"generation_config": {"response_json_schema": {"properties": {"firstName": {}}}, "top_p": 1}})

    def test_schema_for_api_drops_only_dollar_schema(self):
        self.assertEqual(G.schema_for_api({"$schema": "x", "type": "object"}), {"type": "object"})

    def test_mime_types(self):
        self.assertEqual(G.mime_of("a.JPG"), "image/jpeg")
        self.assertEqual(G.mime_of("a.heic"), "image/heic")
        with self.assertRaises(am.InputError):
            G.mime_of("a.tif")


class Key(Case):
    def test_the_key_comes_only_from_gemini_api_key(self):
        os.environ.pop("GEMINI_API_KEY")
        os.environ["GOOGLE_API_KEY"] = "AIza-other-key-0123456789abcdefghij"
        with self.assertRaises(MissingKeyError) as e:
            G.gemini_key()
        self.assertIn("export GEMINI_API_KEY", str(e.exception))
        self.assertIn("Never write the key in a script", str(e.exception))

    def test_the_key_shape_is_checked(self):
        os.environ["GEMINI_API_KEY"] = "short"
        with self.assertRaises(MissingKeyError):
            G.gemini_key()
        os.environ["GEMINI_API_KEY"] = "AIza with a space 0123456789012345"
        with self.assertRaises(MissingKeyError):
            G.gemini_key()
        self.assertEqual(G.mask_key("AIzaSyFAKE-test-key-0123456789abcdefgh"), "AIza...fgh (38 characters)")

    def test_a_real_model_without_a_key_stops_before_anything_is_sent(self):
        make_jpegs("pages", 1)
        os.environ.pop("GEMINI_API_KEY")
        with self.assertRaises(MissingKeyError):
            quiet(am.transcribe, "pages", model="gemini-3.8-flash")


class Retries(Case):
    def test_a_503_is_retried_and_the_200_that_follows_is_kept(self):
        make_jpegs("pages", 1)
        waits = []
        with am.options.override(sleep=waits.append):
            res, fake = run("pages", lambda f, b, n, r: error_response(503) if n == 1 else gemini_reply(data=roster_data(f)))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(res.pages["status"].tolist(), ["ok"])
        self.assertEqual(len(waits), 1)
        self.assertTrue(5 <= waits[0] <= 6)

    def test_google_retry_advice_is_honoured_and_capped(self):
        e = type("E", (), {})()
        ok = error_response(429, retry_delay="17s")
        from google.genai import errors
        err = errors.APIError(429, ok.json(), ok)
        self.assertEqual(G.retry_wait(1, err), 17)
        ra = httpx.Response(503, headers={"retry-after": "300"}, json={"error": {"message": "x"}})
        self.assertEqual(G.retry_wait(1, errors.APIError(503, ra.json(), ra)), 120)
        self.assertTrue(10 <= G.retry_wait(2) <= 11)

    def test_server_errors_are_retried_and_refusals_are_not(self):
        for status, tries in ((500, 2), (502, 2), (504, 2), (408, 2), (400, 1), (403, 1), (404, 1)):
            os.makedirs(f"p{status}")
            make_jpegs(f"p{status}", 1)
            res, fake = run(f"p{status}", lambda f, b, n, r, s=status: error_response(s) if n == 1 else gemini_reply(data=roster_data(f)))
            self.assertEqual(len(fake.calls), tries, status)

    def test_retries_run_out_and_the_last_reply_is_kept(self):
        make_jpegs("pages", 1)
        waits = []
        with am.options.override(sleep=waits.append):
            res, fake = run("pages", lambda f, b, n, r: error_response(503), retries=3)
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual([int(w) for w in waits][:3], [5, 10, 20])
        self.assertEqual(res.pages["problem"].tolist(), ["server"])

    def test_network_failures_are_retried_then_become_a_failed_page(self):
        make_jpegs("pages", 1)
        res, fake = run("pages", lambda f, b, n, r: httpx.ConnectError("Could not resolve host"), retries=2)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(res.pages["problem"].tolist(), ["network"])
        self.assertIn("Could not reach Google", res.pages["message"].iloc[0])

    def test_call_raises_network_error_after_the_last_try(self):
        def boom():
            raise httpx.ConnectError("down")
        with self.assertRaises(NetworkError):
            G.call(boom, retries=1)


class Failures(Case):
    def test_classify_failure_stops_the_run_or_skips_the_page(self):
        c = G.classify_failure
        self.assertEqual(c(400, "API key not valid")["action"], "stop")
        self.assertEqual(c(400, "User location is not supported")["action"], "stop")
        self.assertEqual(c(400, "x", details={"reason": "FAILED_PRECONDITION", "quota": []})["action"], "stop")
        self.assertEqual(c(403, "denied")["action"], "stop")
        self.assertEqual(c(404, "nope", model="gemini-9")["message"].count("gemini-9"), 1)
        self.assertNotIn("model", c(404, "nope")["message"])
        self.assertEqual(c(400, "Base64 decoding failed")["action"], "stop")
        self.assertEqual(c(400, "Invalid image data"), {"action": "skip", "problem": "image", "message": "Google could not use this image (HTTP 400): Invalid image data"})
        self.assertEqual(c(500, "boom")["problem"], "server")
        self.assertEqual(c(400, "weird")["problem"], "request")
        self.assertEqual(c(400, "bad response_json_schema")["action"], "stop")

    def test_a_429_says_which_limit_ran_out(self):
        c = G.classify_failure
        self.assertIn("daily quota", c(429, "Quota exceeded", details={"reason": None, "quota": ["GenerateRequestsPerDayPerProjectPerModel"]})["message"])
        self.assertIn("flex tier", c(429, "busy", engine="flex")["message"])
        self.assertIn("per-minute", c(429, "x", details={"reason": None, "quota": ["RequestsPerMinute"]})["message"])
        self.assertIn("daily quotas reset", c(429, "x")["message"])

    def test_messages_are_shortened_and_quoted_base64_dropped(self):
        msg = "bad " + "A" * 200 + " end"
        self.assertEqual(G.short_message(msg), "bad [...] end")
        self.assertEqual(len(G.short_message("word " * 300)), 500)
        self.assertTrue(G.short_message("word " * 300).endswith("..."))

    def test_a_base64_error_stops_the_run_without_writing_records(self):
        make_jpegs("pages", 3)
        res, fake = run("pages", lambda f, b, n, r: error_response(400, "Base64 decoding failed for x"))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(res.pages["status"].tolist(), ["not run"] * 3)
        self.assertIn("problem in the package", res.info.stopped)

    def test_a_refused_key_stops_the_run_and_earlier_pages_are_kept(self):
        make_jpegs("pages", 3)
        res, fake = run("pages", lambda f, b, n, r: error_response(400, "API key not valid") if f == "p002.jpg" else gemini_reply(data=roster_data(f)))
        self.assertEqual(res.pages["status"].tolist(), ["ok", "not run", "not run"])
        self.assertEqual(len(res), 2)


class Replies(Case):
    def test_read_reply_joins_non_thought_parts_and_reads_the_token_counts(self):
        r = G.read_reply(gemini_reply(text="abc", thoughts=50, cached=10, thought_part=True))
        self.assertEqual(r["text"], "abc")
        self.assertEqual(r["tokens"]["thinking_raw"], 50)
        self.assertEqual(r["tokens"]["cached"], 10)
        self.assertEqual(r["image_tokens"], 2240)
        self.assertEqual(r["finish_reason"], "STOP")

    def test_absent_counts_and_snake_case_replies(self):
        r = G.read_reply(gemini_reply(text="x", image=None))
        self.assertIsNone(r["tokens"]["thinking_raw"])
        self.assertEqual(r["tokens"]["thinking"], 0)
        self.assertIsNone(r["image_tokens"])
        snake = {"candidates": [{"content": {"parts": [{"text": "y"}]}, "finish_reason": "STOP"}],
                 "usage_metadata": {"prompt_token_count": 3, "total_token_count": 5}, "model_version": "m"}
        r = G.read_reply(snake)
        self.assertEqual((r["text"], r["finish_reason"], r["tokens"]["input"], r["model_version"]), ("y", "STOP", 3, "m"))

    def test_no_candidates_gives_the_block_reason(self):
        self.assertEqual(G.read_reply(gemini_reply(no_candidates=True, block_reason="SAFETY"))["block_reason"], "SAFETY")
        self.assertEqual(G.read_reply({})["block_reason"], "no reply")

    def test_finish_messages(self):
        self.assertIn("max_tokens = 100", G.finish_message("MAX_TOKENS", 100))
        self.assertIn("content filter", G.finish_message("SAFETY", 1))
        self.assertIn("RECITATION", G.finish_message("RECITATION", 1))
        self.assertIn("no finish reason", G.finish_message(None, 1))

    def _record(self, reply, raw=None, **settings):
        make_jpegs("pages", 1)
        pages = am._images.as_images("pages")
        con = am._store.contract(test_instructions(), roster_codebook())
        s = {"model": "gemini-3.8-flash", "resolution": "ultra_high", "thinking": "low", "thinking_level": "LOW",
             "max_tokens": 32768, "temperature": None, "engine": "sync", "flex": False, "api_args": {}, "timeout": 600,
             "retries": 4, "prices": None, "schema_channel": None}
        s.update(settings)
        run = TR.Run(s, con, None, None, pages, "k")
        return TR.record_from_reply(reply, raw if raw is not None else json.dumps(reply), pages.iloc[0], run)

    def test_a_good_reply_becomes_an_ok_record_with_tokens_price_and_cost(self):
        rec = self._record(gemini_reply(data=roster_data(), prompt=2500, output=200, thoughts=100))
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["tokens"]["billed_output"], 300)
        self.assertAlmostEqual(rec["cost_usd"], (2500 * 0.75 + 300 * 3.75) / 1e6)
        self.assertEqual(rec["data"], roster_data())
        self.assertEqual(rec["implementation"], "python")

    def test_batch_and_flex_cost_half_and_unknown_models_have_no_cost(self):
        full = self._record(gemini_reply(data=roster_data()))["cost_usd"]
        self.assertAlmostEqual(self._record(gemini_reply(data=roster_data()), engine="batch")["cost_usd"], full / 2)
        self.assertIsNone(self._record(gemini_reply(data=roster_data()), model="gemini-9")["cost_usd"])

    def test_cut_off_blocked_nul_empty_and_non_json_replies_fail_with_their_own_problem(self):
        cases = [(gemini_reply(text='{"a"', finish="MAX_TOKENS"), "truncated"),
                 (gemini_reply(data=roster_data(), finish="SAFETY"), "blocked"),
                 (gemini_reply(no_candidates=True), "blocked"),
                 (gemini_reply(text="   "), "empty"),
                 (gemini_reply(text="not json"), "not_json")]
        for reply, problem in cases:
            rec = self._record(reply)
            self.assertEqual((rec["status"], rec["problem"]), ("failed", problem))
        nul = gemini_reply(text='{"heading": "a\\u0000b", "records": []}')
        self.assertEqual(self._record(nul, raw=json.dumps(nul))["problem"], "control_chars")
        cut = self._record(gemini_reply(text='{"a"', finish="MAX_TOKENS"))
        self.assertEqual(cut["reply"], '{"a"')
        self.assertIn("larger max_tokens", cut["message"])

    def test_a_reply_that_does_not_match_the_codebook_is_kept_as_check(self):
        bad = {"heading": "x", "records": [{"name": "a", "age": "old", "sex": "male"}]}
        rec = self._record(gemini_reply(data=bad))
        self.assertEqual((rec["status"], rec["problem"]), ("check", "invalid"))
        self.assertIn("records[1].age", rec["message"])
        self.assertEqual(rec["data"], bad)

    def test_mechanical_checks_become_notes(self):
        data = {"heading": " padded", "records": [{"name": None, "age": None, "sex": None}]}
        rec = self._record(gemini_reply(data=data, model_version="gemini-2-old", image=900))
        notes = " ".join(rec["notes"])
        self.assertIn("start or end with a space", notes)
        self.assertIn("completely empty", notes)
        self.assertIn("Google answered with gemini-2-old", notes)
        self.assertIn("Only 900 image tokens", notes)

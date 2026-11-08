import json
import os
from datetime import date

from amanuensis import _models as M
from amanuensis import _validate as V
from amanuensis._errors import InputError
from amanuensis._rng import r_sample_int

from .helpers import FIXTURES, Case, quiet, roster_codebook


class Validate(Case):
    def test_types_with_null_unions_and_integers_as_numbers(self):
        s = {"type": ["number", "null"]}
        self.assertEqual(V.validate(3, s), [])
        self.assertEqual(V.validate(None, s), [])
        self.assertEqual(V.validate("3", s), ['(whole reply): expected a number or empty (null), got "3"'])
        self.assertEqual(V.validate(2.0, {"type": "integer"}), [])
        self.assertTrue(V.validate(True, {"type": "integer"}))

    def test_enum_names_the_allowed_values(self):
        err = V.validate("X", {"enum": ["H", "D", None]})
        self.assertEqual(err, ['(whole reply): "X" is not one of the allowed values (H, D, empty)'])
        self.assertEqual(V.validate(None, {"enum": ["H", None]}), [])

    def test_required_and_additional_properties_with_paths(self):
        s = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": False}
        err = V.validate({"a": "x", "c": 1}, s)
        self.assertEqual(err, ["b: is missing", "c: is not a field in the codebook"])

    def test_items_with_positions_counted_from_one(self):
        s = {"type": "object", "properties": {"records": {"type": "array", "minItems": 3, "items": {"type": "object", "properties": {"age": {"type": "integer"}}}}}}
        err = V.validate({"records": [{"age": 1}, {"age": "x"}]}, s)
        self.assertEqual(err, ["records: has 2 item(s), needs at least 3", 'records[2].age: expected a whole number, got "x"'])

    def test_ranges_lengths_and_patterns(self):
        self.assertEqual(V.validate(0, {"minimum": 1}), ["(whole reply): 0 is below the minimum 1"])
        self.assertEqual(V.validate(5, {"exclusiveMaximum": 5}), ["(whole reply): 5 must be below 5"])
        self.assertEqual(V.validate("ab", {"minLength": 3}), ["(whole reply): is shorter than 3 characters"])
        self.assertEqual(V.validate("ab", {"pattern": "^[0-9]+$"}), ['(whole reply): "ab" does not match the pattern ^[0-9]+$'])
        self.assertEqual(V.validate("ab", {"pattern": "(unclosed"}), [])

    def test_refs_and_combinators(self):
        s = {"definitions": {"n": {"type": "integer"}}, "anyOf": [{"$ref": "#/definitions/n"}, {"type": "string"}]}
        self.assertEqual(V.validate(3, s), [])
        self.assertTrue(V.validate(2.5, s))
        self.assertTrue(V.validate(1, {"oneOf": [{"type": "integer"}, {"type": "number"}]}))
        self.assertTrue(V.validate(1, {"not": {"type": "integer"}}))
        self.assertTrue(V.validate(1, {"allOf": [{"minimum": 2}]}))

    def test_the_number_of_problems_is_capped(self):
        s = {"type": "array", "items": {"type": "string"}}
        self.assertEqual(len(V.validate(list(range(50)), s)), 20)

    def test_a_codebooks_schema_catches_wrong_replies(self):
        cb = roster_codebook()
        good = {"heading": "x", "records": [{"name": "a", "age": 3, "sex": "male"}]}
        self.assertEqual(V.validate(good, cb.schema), [])
        bad = {"heading": "x", "records": [{"name": "a", "age": 3, "sex": "dona"}]}
        self.assertEqual(len(V.validate(bad, cb.schema)), 1)

    def test_unchecked_keywords_are_not_property_names(self):
        s = {"type": "object", "properties": {"format": {"type": "string", "x-weird": 1}}}
        self.assertEqual(V.unchecked_keywords(s), ["x-weird"])

    def test_resolve_ref(self):
        root = {"$defs": {"a/b": {"x": 1}, "c": {"y": 2}}}
        self.assertEqual(V.resolve_ref("#/$defs/a~1b", root), {"x": 1})
        self.assertIsNone(V.resolve_ref("#/$defs/zzz", root))
        self.assertIsNone(V.resolve_ref("http://x", root))


class Models(Case):
    def test_the_table_lists_models_with_a_checked_date(self):
        m = M.gemini_models()
        self.assertIn("gemini-3.8-flash", list(m["model"]))
        self.assertEqual(m.checked, "2026-10-23")
        self.assertIn("checked 2026-10-23", repr(m))

    def test_model_info_and_thinking(self):
        self.assertIsNone(M.model_info("mock"))
        self.assertIsNone(M.model_info("gemini-9"))
        self.assertEqual(M.thinking_for("gemini-3.8-flash", "low"), "LOW")
        with self.assertRaises(InputError):
            M.thinking_for("gemini-3.8-flash", "minimal")
        self.assertEqual(M.thinking_for("gemini-9", "minimal"), "MINIMAL")
        with self.assertRaises(InputError):
            M.thinking_for("gemini-3.8-flash", "max")

    def test_prices_follow_announced_changes_discounts_and_overrides(self):
        p = M.price_for("gemini-3.8-flash", when=date(2026, 11, 8))
        self.assertEqual(p, {"input": 0.75, "output": 3.75, "cached": 0.075})
        self.assertEqual(M.price_for("gemini-3.8-flash", when=date(2027, 1, 1))["input"], 1.5)
        self.assertEqual(M.price_for("gemini-3.8-flash", when=date(2026, 11, 8), discount=True)["output"], 1.875)
        self.assertEqual(M.price_for("x", prices={"input": 1, "output": 2}), {"input": 1.0, "output": 2.0, "cached": 1.0})
        self.assertIsNone(M.price_for("x", prices={"input": 1}))
        self.assertIsNone(M.price_for("gemini-9"))

    def test_cost_bills_cached_input_at_the_cached_price_and_thinking_as_output(self):
        p = {"input": 1.0, "output": 10.0, "cached": 0.1}
        self.assertAlmostEqual(M.cost({"input": 1000, "cached": 500, "output": 100, "thinking": 100}, p), (500 + 50 + 2000) / 1e6)
        self.assertIsNone(M.cost({"input": 1}, None))

    def test_check_model_explains_aliases_previews_and_unknown_models(self):
        self.assertEqual(M.check_model("models/gemini-3.8-flash"), "gemini-3.8-flash")
        M.check_model("gemini-3.7-flash")
        M.check_model("gemini-3.1-pro-preview")
        M.check_model("gemini-9")
        text = self.messages()
        self.assertIn("is an old name", text)
        self.assertIn("preview model", text)
        self.assertIn("not in this package's model table", text)
        self.clear_messages()
        M.check_model("example")
        self.assertEqual(self.messages(), "")

    def test_canonical_models_follow_the_routes(self):
        self.assertEqual(M.canonical_model("gemini-3.7-flash"), "gemini-3.8-flash")
        self.assertEqual(M.canonical_model("models/gemini-4-argon"), "gemini-4-argon")


class SameDrawAsR(Case):
    def test_sample_int_reproduces_base_r(self):
        # set.seed(seed, kind = "Mersenne-Twister", sample.kind = "Rejection");
        # head(sample.int(n), 12) in R 4.3.1 (tests/fixtures/r_sample_int.json)
        ref = json.load(open(os.path.join(FIXTURES, "r_sample_int.json")))
        for key, want in ref.items():
            n, seed = key.split("_")
            self.assertEqual(r_sample_int(int(n), int(seed))[:12], want, key)

import math
import os

import pandas as pd

from amanuensis import _text as T
from amanuensis._errors import InputError, InternalError

from .helpers import Case


class TextFiles(Case):
    def test_write_text_is_utf8_lf_and_leaves_no_temporary_file(self):
        T.write_text(["Fàbregas", "Ul·ldemolins"], "a.txt")
        self.assertEqual(open("a.txt", "rb").read(), "Fàbregas\nUl·ldemolins".encode("utf-8"))
        self.assertEqual([f for f in os.listdir(".") if f.startswith(".amanuensis-")], [])

    def test_read_text_strips_bom_crlf_and_falls_back_to_windows_1252(self):
        open("bom.txt", "wb").write(b"\xef\xbb\xbfa\r\nb\rc")
        self.assertEqual(T.read_text("bom.txt"), "a\nb\nc")
        open("cp.txt", "wb").write("Fàbregas".encode("cp1252"))
        with self.warns("Windows-1252"):
            self.assertEqual(T.read_text("cp.txt"), "Fàbregas")

    def test_read_text_decodes_utf16_and_utf32_with_a_bom(self):
        for enc, bom in (("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff"), ("utf-32-le", b"\xff\xfe\x00\x00")):
            open("u.txt", "wb").write(bom + "Antònia\r\nx".encode(enc))
            self.assertEqual(T.read_text("u.txt"), "Antònia\nx", enc)

    def test_a_binary_file_gets_the_callers_advice(self):
        open("b.bin", "wb").write(b"PK\x00\x00stuff")
        with self.assertRaises(InputError) as e:
            T.read_text("b.bin", hint="Save it as .md.")
        self.assertIn("Save it as .md.", str(e.exception))

    def test_append_text_adds_lines_with_a_final_newline(self):
        T.append_text("one", "log.txt")
        T.append_text(["two", "three"], "log.txt")
        self.assertEqual(open("log.txt").read(), "one\ntwo\nthree\n")

    def test_the_old_file_stays_until_the_new_one_is_in_place(self):
        T.write_text("old", "f.txt")
        orig = T._rename
        T._rename = lambda a, b: False
        orig_copy = T._copy
        T._copy = lambda a, b: False
        try:
            with self.assertRaises(InputError):
                T.write_text("new", "f.txt")
        finally:
            T._rename, T._copy = orig, orig_copy
        self.assertEqual(open("f.txt").read(), "old")


class Csv(Case):
    def test_quotes_only_when_needed_and_writes_missing_as_empty(self):
        df = pd.DataFrame({"a": ["x", 'say "hi"', "1,2", None], "n": [1.5, 3500000000.0, math.nan, 2.0]})
        T.write_csv(df, "a.csv", bom=False)
        self.assertEqual(open("a.csv").read(), 'a,n\nx,1.5\n"say ""hi""",3500000000\n"1,2",\n,2\n')

    def test_bom_and_decimal_comma(self):
        df = pd.DataFrame({"a": ["2,5"], "n": [2.5], "k": [3]})
        T.write_csv(df, "b.csv", sep=";", dec=",")
        data = open("b.csv", "rb").read()
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(data[3:].decode(), "a;n;k\n2,5;2,5;3\n")
        with self.assertRaises(InternalError):
            T.write_csv(df, "c.csv", sep=",", dec=",")

    def test_values_that_look_like_formulas_are_kept_byte_for_byte(self):
        df = pd.DataFrame({"a": ["= id.", "+3", "-5", "@x"]})
        T.write_csv(df, "f.csv", bom=False)
        self.assertEqual(open("f.csv").read(), "a\n= id.\n+3\n-5\n@x\n")

    def test_read_delim_text_sniffs_the_separator(self):
        open("s.csv", "wb").write("﻿variable;type\r\nedat ; integer\r\nnom;\r\n".encode("utf-8"))
        df = T.read_delim_text("s.csv")
        self.assertEqual(df.attrs["sep"], ";")
        self.assertEqual(list(df.columns), ["variable", "type"])
        self.assertEqual(df["variable"].tolist(), ["edat", "nom"])
        self.assertEqual(df["type"].tolist(), ["integer", ""])

    def test_as_number_reads_decimal_commas_only_when_told(self):
        self.assertEqual(T.as_number("2,5", dec=","), 2.5)
        self.assertTrue(math.isnan(T.as_number("2,5")))
        self.assertTrue(math.isnan(T.as_number("1.250,5", dec=",")))
        self.assertEqual(T.as_number(" 12 "), 12.0)
        self.assertTrue(math.isnan(T.as_number("12a")))
        self.assertTrue(math.isnan(T.as_number(True)))


class Json(Case):
    def test_compact_json_is_jsonlites(self):
        x = {"type": ["string", "null"], "required": ["a"], "enum": ["H", None], "min": 1.0, "max": 2.5,
             "small": 1e-05, "big": 1e20, "ok": False, "text": "Antònia \"x\"\n"}
        self.assertEqual(T.to_json(x), '{"type":["string","null"],"required":["a"],"enum":["H",null],"min":1,"max":2.5,'
                                       '"small":1e-05,"big":1e+20,"ok":false,"text":"Antònia \\"x\\"\\n"}')

    def test_missing_values_become_null(self):
        self.assertEqual(T.to_json({"a": math.nan, "b": None, "c": pd.NA}), '{"a":null,"b":null,"c":null}')

    def test_json_round_trips_non_ascii(self):
        T.write_json({"nom": "Marcel·lí"}, "x.json")
        self.assertEqual(T.read_json("x.json"), {"nom": "Marcel·lí"})
        self.assertIn("Marcel·lí", open("x.json", encoding="utf-8").read())

    def test_sha12(self):
        self.assertEqual(T.sha12("abc"), "ba7816bf8f01")
        self.assertEqual(len(T.sha12("Fàbregas")), 12)


class Names(Case):
    def test_natural_order_puts_p2_before_p10_ignoring_case(self):
        names = ["p10.jpg", "P2.jpg", "p1.jpg", "scan (12).tif", "scan (3).tif"]
        self.assertEqual(T.natural_sorted(names), ["p1.jpg", "P2.jpg", "p10.jpg", "scan (3).tif", "scan (12).tif"])

    def test_page_key_ignores_case_and_unicode_form(self):
        self.assertEqual(T.page_key("FÀbregas.JPG"), T.page_key("fàbregas.jpg"))

    def test_stem_ext_and_safe_stem(self):
        self.assertEqual(T.stem("dir/a.b.jpg"), "a.b")
        self.assertEqual(T.ext("A.JPG"), "jpg")
        self.assertEqual(T.ext("noext"), "")
        self.assertEqual(T.safe_stem('a:b*c?'), "a_b_c_")
        self.assertEqual(T.safe_stem("CON"), "CON_")
        self.assertEqual(T.safe_stem("name. "), "name")

    def test_fold_accents_and_norm_header(self):
        self.assertEqual(T.fold_accents("Descripció Categorías"), "Descripcio Categorias")
        self.assertEqual(T.norm_header("  Field_Name "), "field name")


class Formatting(Case):
    def test_number_formatting(self):
        self.assertEqual(T.fmt_int(1234567), "1,234,567")
        self.assertEqual(T.fmt_int(None), "-")
        self.assertEqual(T.fmt_usd(0.01234), "$0.0123")
        self.assertEqual(T.fmt_usd(12.345), "$12.35")
        self.assertEqual(T.fmt_usd(None), "unknown")
        self.assertEqual(T.fmt_pct(0.5), "50.00%")
        self.assertEqual(T.chr_value(3500000000.0), "3500000000")
        self.assertEqual(T.chr_value(1 / 3), "0.333333333333333")
        self.assertEqual(T.chr_value(True), "TRUE")
        self.assertEqual(T.chr_value(2.5, dec=","), "2,5")
        self.assertIsNone(T.chr_value(math.nan))

    def test_merge_dicts_merges_nested_replaces_and_deletes(self):
        old = {"a": {"b": 1, "c": 2}, "d": 3}
        self.assertEqual(T.merge_dicts(old, {"a": {"b": 9}, "d": None, "e": 5}), {"a": {"b": 9, "c": 2}, "e": 5})

    def test_argument_checks_say_what_was_expected(self):
        with self.assertRaises(InputError) as e:
            T.check_choice("x", ("a", "b"), "align")
        self.assertIn("'a', 'b'", str(e.exception))
        with self.assertRaises(InputError):
            T.check_number(-1, "budget", minimum=0)
        self.assertIsNone(T.check_number(None, "budget", none_ok=True))

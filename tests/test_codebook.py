import json
import os

import pandas as pd

import amanuensis as am
from amanuensis import _codebook as C
from amanuensis._errors import InputError, NotFoundError

from .helpers import example_file, Case, quiet


# Contract hashes computed by the R package (amanuensis 0.2.0, its
# codebook.R/instructions.R/utils.R sourced read-only in R 4.3.1, 2026-11-08):
# the same codebook hashes the same in both packages, so a results folder
# can be continued by either.
R_HASHES = {"cb_xlsx": "841697f7aea2", "cb_csv": "841697f7aea2", "cb_free": "37562e05b7b2",
            "cb_aswritten": "5036c9c1f516", "cb_json": "27346f8e4069", "cb_df": "89514c9fe3c8"}


def frame(**cols):
    return pd.DataFrame(cols)


class FromTable(Case):
    def test_a_frame_becomes_variables_and_a_draft07_schema_every_field_required_and_nullable(self):
        cb = quiet(am.codebook, frame(variable=["page_no", "name", "age"], type=["text", "text", "integer"],
                                      level=["page", "row", "row"], description=["Printed page.", "Name.", "Age."]))
        s = cb.schema
        self.assertEqual(list(s["properties"]), ["page_no", "records"])
        self.assertEqual(s["required"], ["page_no", "records"])
        self.assertFalse(s["additionalProperties"])
        item = s["properties"]["records"]["items"]
        self.assertEqual(item["required"], ["name", "age"])
        self.assertEqual(item["properties"]["age"]["type"], ["integer", "null"])
        self.assertIn("Give a whole number.", item["properties"]["age"]["description"])
        self.assertEqual(cb.kind, "table")
        self.assertEqual([v.variable for v in cb.vars], ["page_no", "name", "age"])

    def test_every_required_list_is_a_json_array_even_with_one_field(self):
        cb = quiet(am.codebook, frame(variable=["name"]))
        txt = C.to_json(cb.schema)
        self.assertIn('"required":["records"]', txt)
        self.assertIn('"required":["name"]', txt)

    def test_header_and_type_synonyms_in_english_spanish_and_catalan(self):
        cb = quiet(am.codebook, frame(Nombre=["edat", "sexe", "casat"], Tipus=["entero", "categoría", "sí/no"],
                                      Descripció=["Edat.", "Sexe.", "Casat."], Valors=["", "H; D", ""], Nivell=["fila", "fila", "registre"]))
        self.assertEqual([v.type for v in cb.vars], ["integer", "category", "boolean"])
        self.assertEqual(cb.vars[1].values, ["H", "D", "other"])

    def test_unknown_columns_are_mentioned_and_ignored(self):
        quiet(am.codebook, frame(variable=["a"], comentari=["mine"]))
        cb = am.codebook(frame(variable=["a"], comentari=["mine"]))
        self.assertIn("Ignoring codebook column 'comentari'", self.messages())
        self.assertEqual(len(cb.vars), 1)

    def test_an_excel_codebook_may_have_a_title_row_and_a_sheet_called_codebook(self):
        from amanuensis._xlsx import write_xlsx
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "notes"
        ws.append(["nothing here"])
        cbs = wb.create_sheet("Codebook")
        cbs.append(["Codebook of the 1930 register"])
        cbs.append(["variable", "type", "description"])
        cbs.append(["edat", "integer", "Age."])
        wb.save("cb.xlsx")
        cb = quiet(am.codebook, "cb.xlsx")
        self.assertEqual([v.variable for v in cb.vars], ["edat"])
        with self.assertRaises(InputError):
            quiet(am.codebook, "cb.xlsx", sheet="nope")

    def test_a_csv_codebook_with_semicolons_a_bom_and_crlf_is_read(self):
        open("cb.csv", "wb").write("﻿variable;type;values\r\nsexe;category;H; D\r\n".encode("utf-8"))
        # the ';' inside values splits the field: write it quoted instead
        open("cb.csv", "wb").write('﻿variable;type;values\r\nsexe;category;"H; D"\r\n'.encode("utf-8"))
        cb = quiet(am.codebook, "cb.csv")
        self.assertEqual(cb.vars[0].values, ["H", "D", "other"])

    def test_other_is_added_only_when_no_such_value_is_listed(self):
        cb = quiet(am.codebook, frame(variable=["a", "b"], type=["category", "category"], values=["x; y", "x; altres"]))
        self.assertEqual(cb.vars[0].values, ["x", "y", "other"])
        self.assertEqual(cb.vars[1].values, ["x", "altres"])
        cb = quiet(am.codebook, frame(variable=["a"], type=["category"], values=["x; y"]), other=False)
        self.assertEqual(cb.vars[0].values, ["x", "y"])

    def test_values_split_on_semicolons_bars_and_lines_never_commas(self):
        self.assertEqual(C.split_values("Barcelona, ciutat; Girona|Lleida\nTarragona"), ["Barcelona, ciutat", "Girona", "Lleida", "Tarragona"])

    def test_a_category_with_a_single_value_stops_with_a_suggestion(self):
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, frame(variable=["sexe"], type=["category"], values=["H/M"]))
        self.assertIn("did you mean 'H; M'", str(e.exception))

    def test_values_one_per_row_below_their_variable_are_kept(self):
        cb = quiet(am.codebook, frame(variable=["estat", "", "", "edat"], type=["category", "", "", "integer"],
                                      values=["S", "C", "V", ""]))
        self.assertEqual(cb.vars[0].values, ["S", "C", "V", "other"])
        self.assertEqual([v.variable for v in cb.vars], ["estat", "edat"])

    def test_values_on_a_row_of_their_own_away_from_a_category_stop_with_the_row(self):
        with self.assertRaises(InputError) as e:
            quiet(am.codebook, frame(variable=["edat", ""], type=["integer", ""], values=["", "x"]))
        self.assertIn("Row 3", str(e.exception))

    def test_an_exact_column_name_beats_a_synonym_and_loose_synonyms_never_replace_a_real_column(self):
        cb = quiet(am.codebook, frame(name=["x"], variable=["real"], notes=["checked by MP"], description=["What."]))
        self.assertEqual(cb.vars[0].variable, "real")
        self.assertEqual(cb.vars[0].description, "What.")
        cb = quiet(am.codebook, frame(variable=["amount"], unit=["pesetas"]))
        self.assertEqual(cb.vars[0].level, "row")

    def test_as_written_adds_a_text_twin_just_before_each_number(self):
        cb = quiet(am.codebook, frame(variable=["total", "name"], type=["number", "text"]), as_written=True)
        self.assertEqual([v.variable for v in cb.vars], ["total_as_written", "total", "name"])

    def test_reserved_duplicate_and_missing_names_stop(self):
        for f in (frame(variable=["file"]), frame(variable=["a", "a"]), frame(variable=["a", ""], description=["x", "y"])):
            with self.assertRaises(InputError):
                quiet(am.codebook, f)

    def test_missing_descriptions_are_pointed_out(self):
        am.codebook(frame(variable=["a", "b"], description=["x", ""]))
        self.assertIn("1 of 2 variables has no description", self.messages())

    def test_dates_are_kept_as_text(self):
        cb = quiet(am.codebook, frame(variable=["d"], type=["fecha"]))
        self.assertEqual(cb.vars[0].type, "text")


class FreeTextAndJson(Case):
    def test_free_text_alone_and_with_a_table(self):
        cb = am.codebook(free_text=True)
        self.assertEqual(cb.kind, "text")
        self.assertEqual(list(cb.schema["properties"]), ["text"])
        cb = quiet(am.codebook, frame(variable=["name"]), free_text=True)
        self.assertEqual(list(cb.schema["properties"]), ["records", "text"])
        self.assertTrue(cb.free_text)
        self.assertEqual(cb.vars[-1].variable, "text")
        with self.assertRaises(InputError):
            quiet(am.codebook, frame(variable=["text"]), free_text=True)

    def test_a_json_schema_is_kept_as_it_is(self):
        s = {"type": "object", "properties": {"title": {"type": "string"}, "rows": {"type": "array", "items": {
            "type": "object", "properties": {"n": {"type": "integer"}, "who": {"type": "object", "properties": {"name": {"type": "string"}}}}}}}}
        cb = quiet(am.codebook, json=json.dumps(s))
        self.assertEqual(cb.schema, s)
        self.assertEqual(cb.kind, "json")
        self.assertEqual([(v.variable, v.level) for v in cb.vars], [("title", "page"), ("n", "row"), ("who.name", "row")])
        open("s.json", "w").write(json.dumps(s))
        self.assertEqual(quiet(am.codebook, "s.json").schema, s)
        self.assertEqual(quiet(am.codebook, json.dumps(s)).kind, "json")

    def test_json_schemas_are_checked(self):
        for bad in ('[1, 2]', '{"type": "string"}', '{"type": "object", "properties": {"file": {"type": "string"}}}'):
            with self.assertRaises(InputError):
                quiet(am.codebook, json=bad)
        with self.assertRaises(InputError):
            quiet(am.codebook, json="{not json")

    def test_a_field_of_the_page_and_of_each_entry_get_a_column_each_and_provenance_is_protected(self):
        # refused before, as a newspaper's blocks schema was
        s = {"type": "object", "properties": {"notes": {"type": "string"}, "people": {"type": "array", "items": {
            "type": "object", "properties": {"notes": {"type": "string"}}}}}}
        cb = quiet(am.codebook, json=json.dumps(s))
        self.assertEqual([(v.variable, v.level) for v in cb.vars], [("notes", "page"), ("people.notes", "row")])
        tab = am._flatten.flatten_records([{"file": "a.jpg", "data": {"notes": "on the page", "people": [{"notes": "a"}, {"notes": "b"}]}}], cb)
        self.assertEqual(tab["notes"].tolist(), ["on the page"] * 2)
        self.assertEqual(tab["people.notes"].tolist(), ["a", "b"])
        s = {"type": "object", "properties": {"people": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}}},
                                              "notes": {"type": "array", "items": {"type": "object", "properties": {"row": {"type": "string"}}}}}}
        with self.assertRaises(InputError):
            quiet(am.codebook, json=json.dumps(s))

    def test_the_main_array_is_found_depth_first_and_through_refs(self):
        s = {"type": "object", "properties": {"tables": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "rows": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}}}}}}}}
        self.assertEqual(C.main_array_path(s)[0], ["tables", "rows"])
        s = {"type": "object", "$defs": {"person": {"type": "object", "properties": {"name": {"type": "string"}}}},
             "properties": {"people": {"type": "array", "items": {"$ref": "#/$defs/person"}}}}
        self.assertEqual(C.main_array_path(s)[0], ["people"])
        self.assertEqual([v.variable for v in C.vars_from_schema(s)], ["name"])

    def test_unchecked_keywords_are_named(self):
        am.codebook(json='{"type":"object","properties":{"a":{"type":"string","format":"email","x-custom":1}}}')
        self.assertIn("'x-custom'", self.messages())


class FilesAndTemplates(Case):
    def test_missing_files_and_wrong_types_are_refused(self):
        with self.assertRaises(NotFoundError):
            quiet(am.codebook, "nope.xlsx")
        open("cb.doc", "w").write("x")
        with self.assertRaises(InputError):
            quiet(am.codebook, "cb.doc")
        with self.assertRaises(InputError):
            am.codebook()

    def test_codebook_template_writes_a_starter_that_codebook_reads(self):
        for name in ("cb.xlsx", "cb.csv"):
            quiet(am.codebook_template, name)
            cb = quiet(am.codebook, name)
            self.assertEqual(cb.vars[0].variable, "printed_page")
            self.assertEqual(cb.vars[4].values, ["male", "female", "other"])
            with self.assertRaises(InputError):
                am.codebook_template(name)

    def test_as_codebook_defaults_to_free_text(self):
        self.assertEqual(quiet(C.as_codebook, None).kind, "text")
        with self.assertRaises(InputError):
            C.as_codebook(3)

    def test_printing(self):
        cb = quiet(am.codebook, example_file("codebook.xlsx"))
        text = repr(cb)
        self.assertIn("9 variables - 1 per page, 8 per row", text)
        self.assertEqual(repr(am.codebook(free_text=True)), "Codebook: free text - the model transcribes each page as text.")
        self.assertEqual(list(cb.variables.columns), ["variable", "type", "level", "description", "values"])


class SameHashAsR(Case):
    def test_codebooks_hash_as_in_the_r_package(self):
        got = {
            "cb_xlsx": quiet(am.codebook, example_file("codebook.xlsx")).sha12,
            "cb_csv": quiet(am.codebook, example_file("codebook.csv")).sha12,
            "cb_free": am.codebook(free_text=True).sha12,
            "cb_aswritten": quiet(am.codebook, example_file("codebook.csv"), as_written=True, free_text=True, record="person listed").sha12,
            "cb_json": quiet(am.codebook, json='{"type":"object","properties":{"title":{"type":["string","null"]},"rows":{"type":"array","items":{"type":"object","properties":{"n":{"type":"integer","minimum":1},"x":{"type":"number","maximum":2.5}}}}}}').sha12,
            "cb_df": quiet(am.codebook, frame(variable=["heading", "name", "age", "sex"], type=["text", "text", "integer", "category"],
                                              level=["page", "row", "row", "row"], values=["", "", "", "male; female"],
                                              description=["Page heading.", "Name as written.", "Age in years.", "Sex as recorded."])).sha12,
        }
        self.assertEqual(got, R_HASHES)

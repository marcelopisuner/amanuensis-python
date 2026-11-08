import os
import unicodedata

import amanuensis as am
from amanuensis import _instructions as I
from amanuensis._docx import write_docx
from amanuensis._errors import InputError, NotFoundError

from .helpers import example_file, Case, quiet



class Instructions(Case):
    def test_text_gets_the_standard_rules_appended(self):
        x = am.instructions(text="A 1930 register.")
        self.assertTrue(x.full.startswith("A 1930 register.\n\n# Rules\n- Transcribe exactly as written"))
        self.assertTrue(x.rules)
        self.assertEqual(x.rules_id, "rules-1")
        self.assertEqual(x.source, "text")

    def test_rules_false_sends_exactly_the_text_and_rules_can_name_a_version(self):
        x = am.instructions(text="Only this.", rules=False)
        self.assertEqual(x.full, "Only this.")
        self.assertEqual(am.instructions(text="A.", rules="rules-1").full, am.instructions(text="A.").full)
        with self.assertRaises(InputError):
            am.instructions(text="A.", rules="rules-9")

    def test_no_file_and_no_text_gives_the_default(self):
        x = am.instructions()
        self.assertEqual(x.source, "default")
        self.assertIn("# Role", x.full)

    def test_text_given_where_a_file_was_expected_is_used(self):
        x = am.instructions("Transcribe every row of the table.")
        self.assertEqual(x.text, "Transcribe every row of the table.")
        self.assertIn("Using this text as the instructions", self.messages())

    def test_text_files_are_read_as_utf8_with_bom_and_crlf_handled(self):
        for name in ("a.txt", "a.md", "a.tex"):
            open(name, "wb").write("﻿# Role\r\nFàbregas\r\n".encode("utf-8"))
            self.assertEqual(am.instructions(name).text, "# Role\nFàbregas")

    def test_text_is_nfc_so_the_same_file_hashes_the_same_everywhere(self):
        nfd = unicodedata.normalize("NFD", "Antònia")
        self.assertEqual(am.instructions(text=nfd).sha12, am.instructions(text="Antònia").sha12)

    def test_a_docx_is_read_paragraph_by_paragraph_with_tabs_breaks_and_entities(self):
        write_docx([{"type": "heading", "text": "Role"}, {"type": "para", "text": "a\tb\nc & <d>"}], "a.docx")
        self.assertEqual(am.instructions("a.docx").text, "Role\na\tb\nc & <d>")

    def test_a_docx_written_by_the_package_reads_back(self):
        quiet(am.instructions_template, "t.docx")
        x = quiet(am.instructions, "t.docx")
        self.assertIn("Role", x.text)

    def test_unreadable_or_unsupported_files_say_what_to_do(self):
        with self.assertRaises(NotFoundError):
            am.instructions("missing.md")
        open("a.doc", "w").write("x")
        with self.assertRaises(InputError) as e:
            am.instructions("a.doc")
        self.assertIn(".docx", str(e.exception))
        open("b.docx", "w").write("not a zip")
        with self.assertRaises(InputError):
            am.instructions("b.docx")

    def test_placeholders_and_long_instructions_warn(self):
        with self.warns("placeholder"):
            am.instructions(text="You are transcribing <what the source is>.")
        with self.no_warnings():
            am.instructions(text="Ages < 20 and > 60 are rare; <b>bold</b> is fine.")
        with self.warns("long"):
            am.instructions(text="word " * 1300)

    def test_every_placeholder_of_the_template_is_found(self):
        slots = [s for s in I.PLACEHOLDER_RE.findall("\n".join(I.TEMPLATE_LINES))]
        self.assertEqual(len(slots), 7)

    def test_bad_combinations_and_empty_instructions_stop(self):
        with self.assertRaises(InputError):
            am.instructions(file="a.md", text="x")
        with self.assertRaises(InputError):
            am.instructions(text="   ")

    def test_instructions_template_writes_md_and_txt(self):
        for name in ("t.md", "t.txt"):
            quiet(am.instructions_template, name)
            self.assertTrue(open(name).read().startswith("# Role\n"))
        with self.assertRaises(InputError):
            am.instructions_template("t.md")
        with self.assertRaises(InputError):
            am.instructions_template("t.rtf")

    def test_as_instructions_accepts_objects_and_files(self):
        x = am.instructions(text="A.")
        self.assertIs(I.as_instructions(x), x)
        self.assertEqual(quiet(I.as_instructions, None).source, "default")
        with self.assertRaises(NotFoundError):
            I.as_instructions("nope.md")

    def test_the_same_instructions_hash_as_in_the_r_package(self):
        # computed by the R package's instructions() (see test_codebook.R_HASHES)
        self.assertEqual(am.instructions(example_file("instructions.md")).sha12, "df9ccd6b7a7c")
        self.assertEqual(am.instructions().sha12, "17b87bd9de11")
        self.assertEqual(am.instructions(text="These are pages of a 1930 municipal register.", rules=False).sha12, "773d924113aa")

    def test_printing_shows_what_is_sent(self):
        text = repr(am.instructions(text="A 1930 register."))
        self.assertIn("Instructions (typed in Python", text)
        self.assertIn("# Rules", text)

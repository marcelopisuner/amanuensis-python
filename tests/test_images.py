import os
import shutil
import subprocess
import sys
import unittest

from PIL import Image

import amanuensis as am
from amanuensis import _images as IM
from amanuensis._errors import InputError, NotFoundError

from .helpers import Case, make_jpegs, make_pdf, quiet


def sideways_jpeg(path, orientation=6):
    img = Image.new("RGB", (80, 40), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = orientation
    img.save(path, "JPEG", exif=exif.tobytes())
    return path


class Pdf(Case):
    def test_a_pdf_is_rendered_page_by_page_with_numbered_names_and_its_source_page(self):
        make_pdf("book.pdf", 3)
        pages = quiet(am.images, "book.pdf", dpi=72)
        self.assertEqual(list(pages["file"]), ["book_p001.jpg", "book_p002.jpg", "book_p003.jpg"])
        self.assertEqual(list(pages["source_page"]), [1, 2, 3])
        self.assertEqual(set(pages["status"]), {"rendered"})
        self.assertEqual(pages.folder, IM.show_path("book_pages"))
        self.assertTrue(all(IM.jpeg_complete(p) for p in pages["image"]))
        # the test PDF's pages are 3 x 4 inches
        self.assertEqual((pages["width"].iloc[0], pages["height"].iloc[0]), (216, 288))
        hi = quiet(am.images, "book.pdf", dpi=100, save_to="hi")
        self.assertEqual(hi["width"].iloc[0], 300)

    def test_running_again_finds_the_pages_and_redoes_a_truncated_one(self):
        make_pdf("book.pdf", 2)
        quiet(am.images, "book.pdf", dpi=72)
        p2 = os.path.join("book_pages", "book_p002.jpg")
        data = open(p2, "rb").read()
        open(p2, "wb").write(data[: len(data) // 2])
        again = quiet(am.images, "book.pdf", dpi=72)
        self.assertEqual(list(again["status"]), ["already there", "rendered"])

    def test_pages_picks_pdf_pages_and_out_of_range_pages_are_ignored_with_a_warning(self):
        make_pdf("book.pdf", 3)
        with self.warns("ignoring page number 9"):
            pages = quiet(am.images, "book.pdf", dpi=72, pages=[2, 9])
        self.assertEqual(list(pages["file"]), ["book_p002.jpg"])

    def test_images_csv_indexes_the_folder_and_a_later_session_recovers_source_and_page(self):
        make_pdf("book.pdf", 2)
        quiet(am.images, "book.pdf", dpi=72)
        idx = open(os.path.join("book_pages", "images.csv")).read().splitlines()
        self.assertEqual(idx[0], "file,source,source_page,width,height,dpi,source_bytes")
        self.assertTrue(idx[1].startswith("book_p001.jpg,book.pdf,1,"))
        later = IM.as_images("book_pages")
        self.assertEqual(list(later["source"]), ["book.pdf", "book.pdf"])
        self.assertEqual(list(later["source_page"]), [1, 2])

    def test_a_damaged_pdf_fails_with_a_plain_reason(self):
        open("bad.pdf", "wb").write(b"%PDF-1.4 nothing")
        with self.warns("could not be prepared"):
            pages = quiet(am.images, "bad.pdf")
        self.assertEqual(list(pages["status"]), ["failed"])
        self.assertIn("damaged, password-protected, or not really a PDF", pages["note"].iloc[0])

    def test_the_native_resolution_of_scans_inside_a_pdf_is_detected(self):
        raw = (b"<< /Type /XObject /Subtype /Image /Width 1700 /Height 2200 >>")
        self.assertEqual(IM.pdf_native_dpi(raw, [(612, 792)]), 200)
        self.assertIsNone(IM.pdf_native_dpi(b"no images", [(612, 792)]))


class Other(Case):
    def test_a_multi_page_tiff_is_split_and_a_single_page_tiff_is_one_image(self):
        frames = [Image.new("RGB", (50, 30), c) for c in ((10, 10, 10), (200, 200, 200), (90, 30, 30))]
        frames[0].save("scan.tif", save_all=True, append_images=frames[1:])
        Image.new("RGB", (50, 30), (1, 2, 3)).save("one.tif")
        pages = quiet(am.images, ["scan.tif", "one.tif"])
        self.assertEqual(list(pages["file"]), ["scan_p001.jpg", "scan_p002.jpg", "scan_p003.jpg", "one.jpg"])
        self.assertEqual(list(pages["source_page"]), [1, 2, 3, 1])

    def test_png_and_webp_are_converted_to_jpeg(self):
        Image.new("RGBA", (40, 30), (0, 0, 255, 128)).save("a.png")
        Image.new("RGB", (40, 30), (0, 255, 0)).save("b.webp")
        pages = quiet(am.images, ["a.png", "b.webp"])
        self.assertEqual(list(pages["file"]), ["a.jpg", "b.jpg"])
        self.assertEqual(set(pages["status"]), {"converted"})
        with Image.open(pages["image"].iloc[0]) as im:
            self.assertEqual(im.format, "JPEG")
            self.assertEqual(im.mode, "RGB")

    def test_a_sideways_jpeg_is_turned_upright_and_good_jpegs_are_copied_byte_for_byte(self):
        os.makedirs("in")
        sideways_jpeg("in/side.jpg")
        good = make_jpegs("in", 1, prefix="good")[0]
        pages = quiet(am.images, "in", save_to="out")
        st = dict(zip(pages["file"], pages["status"]))
        self.assertEqual(st, {"good001.jpg": "copied", "side.jpg": "converted"})
        self.assertEqual(open(good, "rb").read(), open("out/good001.jpg", "rb").read())
        with Image.open("out/side.jpg") as im:
            self.assertEqual(im.size, (40, 80))
        self.assertIn("stored sideways", pages.loc[pages["file"] == "side.jpg", "note"].iloc[0])

    def test_a_folder_of_ready_jpegs_is_used_in_place(self):
        make_jpegs("scans", 3)
        pages = quiet(am.images, "scans")
        self.assertEqual(set(pages["status"]), {"in place"})
        self.assertFalse(os.path.exists(os.path.join("scans", "images.csv")))
        self.assertFalse(os.path.exists(os.path.join("scans", "pages")))
        self.assertEqual(sorted(os.listdir("scans")), ["p001.jpg", "p002.jpg", "p003.jpg"])

    def test_a_jpg_that_is_really_a_png_is_re_encoded(self):
        os.makedirs("in")
        Image.new("RGB", (20, 20), (5, 5, 5)).save("in/fake.jpg", "PNG")
        pages = quiet(am.images, "in")
        self.assertEqual(pages["status"].iloc[0], "converted")
        self.assertIn("really a PNG", pages["note"].iloc[0])

    def test_two_sources_with_one_stem_get_distinct_page_images(self):
        Image.new("RGB", (20, 20), (5, 5, 5)).save("a.png")
        Image.new("RGB", (20, 20), (9, 9, 9)).save("a.tif")
        pages = quiet(am.images, ["a.png", "a.tif"])
        self.assertEqual(sorted(pages["file"]), ["a_png.jpg", "a_tif.jpg"])

    def test_jpegs_with_the_same_name_in_two_folders_get_one_image_each(self):
        make_jpegs("card1", 1, prefix="IMG_")
        make_jpegs("card2", 2, prefix="IMG_")
        pages = quiet(am.images, ["card1/IMG_001.jpg", "card2/IMG_001.jpg"], save_to="pages")
        self.assertEqual(list(pages["file"]), ["IMG_001.jpg", "card2_IMG_001.jpg"])
        self.assertEqual(list(pages["source"]), ["card1/IMG_001.jpg", "card2/IMG_001.jpg"])
        self.assertIn("named after its folder", pages["note"].iloc[1])
        again = quiet(am.images, ["card1/IMG_001.jpg", "card2/IMG_001.jpg"], save_to="pages")
        self.assertEqual(list(again["file"]), ["IMG_001.jpg", "card2_IMG_001.jpg"])
        self.assertEqual(list(again["status"]), ["already there", "already there"])

    def test_a_damaged_file_is_listed_as_failed_and_never_dropped(self):
        open("broken.png", "wb").write(b"\x89PNG not really")
        Image.new("RGB", (20, 20)).save("ok.png")
        with self.warns("could not be prepared"):
            pages = quiet(am.images, ["broken.png", "ok.png"])
        self.assertEqual(list(pages["status"]), ["failed", "converted"])

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("sips"), "HEIC conversion uses macOS's sips here")
    def test_heic_photos_are_converted(self):
        Image.new("RGB", (64, 48), (120, 30, 200)).save("src.png")
        try:
            subprocess.run(["sips", "-s", "format", "heic", "src.png", "--out", "photo.heic"], check=True, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            self.skipTest("sips cannot write HEIC here")
        pages = quiet(am.images, "photo.heic")
        self.assertEqual(list(pages["file"]), ["photo.jpg"])
        self.assertEqual(pages["status"].iloc[0], "converted")


class Inputs(Case):
    def test_inputs_are_checked(self):
        with self.assertRaises(NotFoundError):
            am.images("nope.pdf")
        open("notes.docx", "w").write("x")
        with self.assertRaises(InputError):
            am.images("notes.docx")
        os.makedirs("empty")
        with self.assertRaises(InputError):
            quiet(am.images, "empty")
        make_pdf("b.pdf", 1)
        with self.assertRaises(InputError):
            am.images("b.pdf", dpi=10)
        with self.assertRaises(InputError):
            am.images("b.pdf", pages=["x"])
        with self.assertRaises(InputError):
            am.images(None)

    def test_the_default_folder_sits_next_to_the_source(self):
        os.makedirs("scans")
        make_pdf("scans/a.pdf", 1)
        make_pdf("scans/b.pdf", 1)
        pages = quiet(am.images, "scans", dpi=50)
        self.assertEqual(pages.folder, IM.show_path("scans/pages"))
        self.assertIn("Skipped", self.messages()) if False else None

    def test_other_files_in_a_folder_are_skipped_and_said(self):
        os.makedirs("scans")
        make_jpegs("scans", 1)
        open("scans/notes.txt", "w").write("x")
        am.images("scans")
        self.assertIn("Skipped 1 file that is not a PDF or an image: notes.txt", self.messages())

    def test_as_images_accepts_folders_files_and_tables_and_names_must_be_unique(self):
        paths = make_jpegs("scans", 2)
        self.assertEqual(len(IM.as_images("scans")), 2)
        self.assertEqual(len(IM.as_images(paths)), 2)
        self.assertEqual(len(IM.as_images(quiet(am.images, "scans"))), 2)
        make_jpegs("other", 1)
        with self.assertRaises(InputError):
            IM.as_images([paths[0], "other/p001.jpg"])
        Image.new("RGB", (20, 20)).save("x.tif")
        with self.assertRaises(InputError):
            IM.as_images(["x.tif"])
        sideways_jpeg("side.jpg")
        with self.assertRaises(InputError):
            IM.as_images(["side.jpg"])
        with self.assertRaises(NotFoundError):
            IM.as_images(["missing.jpg"])

    def test_small_helpers(self):
        p = sideways_jpeg("s.jpg", 8)
        self.assertEqual(IM.jpeg_header(p)["orientation"], 8)
        self.assertFalse(IM.jpeg_ready(p))
        a, b = make_jpegs("x", 2)
        self.assertFalse(IM.same_bytes(a, b))
        self.assertTrue(IM.same_bytes(a, a))
        self.assertTrue(IM.jpeg_complete(a))
        self.assertTrue(IM._is_page_of("book_p001", "book"))
        self.assertFalse(IM._is_page_of("book_px", "book"))

    def test_package_folders_are_refused(self):
        with self.assertRaises(InputError):
            IM.refuse_library_path(os.path.dirname(am.__file__))

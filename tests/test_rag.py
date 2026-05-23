import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import chunk_text
from rag_core import _guess_instrument_from_query, _query_intents
from rag_structured import Norm, find_norm_by_instrument


class TestChunking(unittest.TestCase):
    def test_short_text_not_split(self):
        out = chunk_text("Короткий абзац про установку.", chunk=900)
        self.assertEqual(len(out), 1)

    def test_long_text_split(self):
        long_text = "А" * 2500
        out = chunk_text(long_text, chunk=900, overlap=150)
        self.assertGreater(len(out), 1)

    def test_paragraphs_kept_separate(self):
        text = "Первый абзац.\n\nВторой абзац."
        out = chunk_text(text)
        self.assertEqual(len(out), 2)


class TestInstrumentGuess(unittest.TestCase):
    def test_finds_lic(self):
        self.assertEqual(_guess_instrument_from_query("уставки LIC-31050?"), "LIC-31050")

    def test_finds_without_dash(self):
        self.assertEqual(_guess_instrument_from_query("Что по FT310452"), "FT310452")

    def test_no_instrument(self):
        self.assertEqual(_guess_instrument_from_query("Какие продукты установки 300?"), "")


class TestIntents(unittest.TestCase):
    def test_norms_intent(self):
        self.assertIn("norms", _query_intents("какие уставки и диапазон"))

    def test_alarms_intent(self):
        self.assertIn("alarms", _query_intents("действия при срабатывании сигнализации"))

    def test_reglament_intent(self):
        self.assertIn("reglament", _query_intents("какие продукты установки"))


class TestFindByInstrument(unittest.TestCase):
    def test_find_norm(self):
        n = Norm(instrument="LIC-31050", param="LIC-31050", unit="%",
                 range_min=10, range_max=60, work_min=30, work_max=50,
                 page=1, equipment="G-304")
        self.assertEqual(len(find_norm_by_instrument([n], "LIC-31050")), 1)

    def test_not_found(self):
        n = Norm(instrument="LIC-31050", param="LIC-31050", unit="%",
                 range_min=0, range_max=100, work_min=0, work_max=100,
                 page=1, equipment="G-304")
        self.assertEqual(find_norm_by_instrument([n], "PDT-99999"), [])


if __name__ == "__main__":
    unittest.main()

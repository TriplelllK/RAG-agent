import unittest

from parsers.parse_norms import parse_norm_row
from parsers.parse_alarms import parse_alarm_row


class TestParsers(unittest.TestCase):
    def test_parse_norm_row_standard(self):
        row = ["LIC-31050", "Уровень", "%", "10", "60", "30", "50"]
        out = parse_norm_row(row, "G-304", 2)
        self.assertIsNotNone(out)
        self.assertEqual(out.instrument, "LIC-31050")
        self.assertEqual(out.param, "Уровень")
        self.assertEqual(out.range_min, 10.0)
        self.assertEqual(out.range_max, 60.0)
        self.assertEqual(out.work_min, 30.0)
        self.assertEqual(out.work_max, 50.0)
        self.assertEqual(out.equipment, "G-304")

    def test_parse_norm_row_shifted_columns(self):
        row = ["1", "PDT-31016", "Перепад давления", "мбар", "0", "700", "200"]
        out = parse_norm_row(row, "D-301", 5)
        self.assertIsNotNone(out)
        self.assertEqual(out.instrument, "PDT-31016")
        self.assertEqual(out.equipment, "D-301")

    def test_parse_alarm_row_standard(self):
        row = [
            "FAL-31002",
            "Расход амина",
            "м3/ч",
            "80",
            "-",
            "-",
            "Остановка насоса G-303 A/B",
            "Примечание",
        ]
        out = parse_alarm_row(row, "G-303", 3)
        self.assertIsNotNone(out)
        self.assertEqual(out.instrument, "FAL-31002")
        self.assertEqual(out.setpoint, "80")
        self.assertIn("Остановка", out.action)
        self.assertEqual(out.equipment, "G-303")

    def test_parse_alarm_row_without_instrument(self):
        row = ["Строка", "без", "прибора"]
        out = parse_alarm_row(row, "G-303", 3)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()

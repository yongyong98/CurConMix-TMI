"""Keep code, clinical data, and third-party license scopes separate."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LicensingTests(unittest.TestCase):
    def test_author_owned_code_uses_mit(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("The CurConMix contributors", license_text)
        self.assertIn("Permission is hereby granted, free of charge", license_text)
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', license_text)
        self.assertIn('license = {file = "LICENSE"}', (ROOT / "pyproject.toml").read_text())

    def test_data_and_weights_have_separate_noncommercial_terms(self):
        text = (ROOT / "DATA_AND_MODEL_LICENSE.md").read_text(encoding="utf-8")
        self.assertIn("CC BY-NC 4.0", text)
        self.assertIn("separate from the software license", text)
        self.assertIn("No dataset, annotation table, features, or weights", text)
        self.assertTrue((ROOT / "third_party/licenses/CC-BY-NC-4.0.txt").is_file())

    def test_upstream_bsd_notices_are_retained(self):
        for name, holder in (
            ("ivtmetrics", "Research Group CAMMA"),
            ("SupContrast", "Yonglong Tian"),
        ):
            text = (ROOT / "third_party/licenses" / f"{name}.txt").read_text()
            self.assertIn("BSD 2-Clause License", text)
            self.assertIn(holder, text)
        self.assertIn(
            "does not replace third-party licenses",
            (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()

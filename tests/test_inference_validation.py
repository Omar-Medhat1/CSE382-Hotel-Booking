import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.inference import predict


class InferenceValidationTests(unittest.TestCase):

    def test_negative_guest_counts(self):
        original = json.loads(
            Path("examples/booking.json").read_text()
        )

        for field in ("Adults", "Children", "Babies"):
            with self.subTest(field=field):
                booking = original.copy()
                booking[field] = -2

                with TemporaryDirectory() as folder:
                    path = Path(folder) / "invalid.json"
                    path.write_text(json.dumps(booking))

                    with self.assertRaisesRegex(
                        ValueError, f"{field} cannot be negative"
                    ):
                        predict("regression", path)

    def test_infinite_guest_count(self):
        original = json.loads(
            Path("examples/booking.json").read_text()
        )
        original["Adults"] = "Infinity"

        with TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text(json.dumps(original))

            with self.assertRaisesRegex(
                ValueError, "Adults must be finite"
            ):
                predict("regression", path)
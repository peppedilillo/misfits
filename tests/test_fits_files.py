import asyncio
from pathlib import Path
import unittest

from misfits.app import Misfits
from misfits.data import _validate_fits
from misfits.data import get_fits_content


class TestFitsLoading(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = Path("tests/data")
        self.fits_files = list(self.test_dir.glob("*.fits"))

    def test_fits_files_exist(self):
        self.assertTrue(
            len(self.fits_files) > 0, "No FITS files found in tests directory"
        )

    def test_validate_all_fits(self):
        for fits_file in self.fits_files:
            with self.subTest(file=fits_file.name):
                self.assertTrue(
                    _validate_fits(fits_file), f"Failed to validate {fits_file.name}"
                )

    async def test_load_all_fits(self):
        for fits_file in self.fits_files:
            with self.subTest(file=fits_file.name):
                try:
                    content = await get_fits_content(fits_file)
                    self.assertIsInstance(content, tuple)
                    self.assertGreater(
                        len(content), 0, f"No HDUs found in {fits_file.name}"
                    )

                    for i, hdu in enumerate(content):
                        self.assertIn("name", hdu, f"HDU {i} missing 'name' field")
                        self.assertIn("type", hdu, f"HDU {i} missing 'type' field")
                        self.assertIn("header", hdu, f"HDU {i} missing 'header' field")
                        self.assertIn(
                            "is_table", hdu, f"HDU {i} missing 'is_table' field"
                        )
                        self.assertIn("data", hdu, f"HDU {i} missing 'data' field")

                except Exception as e:
                    self.fail(f"Failed to load {fits_file.name}: {str(e)}")

    async def test_app_load_all_fits(self):
        for fits_file in self.fits_files:
            with self.subTest(file=fits_file.name):
                try:
                    app = Misfits(filepath=fits_file)
                    async with app.run_test() as pilot:
                        await pilot.press("r")
                except Exception as e:
                    self.fail(
                        f"Failed to start application with file {fits_file.name}: {str(e)}"
                    )


if __name__ == "__main__":
    unittest.main()

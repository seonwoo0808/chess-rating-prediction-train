import tempfile
import unittest
from pathlib import Path

from train.main import collect_parquet_files, build_parser


class MainTests(unittest.TestCase):
    def test_directory_files_are_sorted_and_non_parquet_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "2024-10.parquet").touch()
            (root / "2024-09.parquet").touch()
            (root / "README.txt").touch()
            self.assertEqual(
                [path.name for path in collect_parquet_files([root])],
                ["2024-09.parquet", "2024-10.parquet"],
            )

    def test_explicit_file_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.parquet"
            second = root / "second.parquet"
            first.touch()
            second.touch()
            self.assertEqual(collect_parquet_files([second, first]),
                             (second.resolve(), first.resolve()))

    def test_cli_accepts_a_directory(self):
        args = build_parser().parse_args(["/data/monthly", "--epochs", "3"])
        self.assertEqual(args.parquet, [Path("/data/monthly")])
        self.assertEqual(args.epochs, 3)

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, r"No \.parquet files"):
                collect_parquet_files([directory])


if __name__ == "__main__":
    unittest.main()

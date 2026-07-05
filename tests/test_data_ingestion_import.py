import importlib
import sys
from pathlib import Path
import unittest


class DataIngestionImportTest(unittest.TestCase):
    def test_data_ingestion_module_imports(self):
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        module = importlib.import_module("src.components.data_ingestion")
        self.assertTrue(hasattr(module, "DataIngestion"))


if __name__ == "__main__":
    unittest.main()

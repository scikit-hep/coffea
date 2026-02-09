import importlib
import sys


def test_suite_operates_without_topcoffea():
    """Ensure accidental imports of topcoffea fail fast during testing."""

    class BlockTopcoffeaImporter:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "topcoffea":
                raise ModuleNotFoundError("topcoffea intentionally unavailable")
            return None

    blocker = BlockTopcoffeaImporter()
    sys.meta_path.insert(0, blocker)
    try:
        # Touch commonly used modules to ensure coffea remains functional
        hist = importlib.import_module("coffea.hist")
        processor = importlib.import_module("coffea.processor")
        assert hist is not None
        assert processor is not None
    finally:
        sys.meta_path.remove(blocker)

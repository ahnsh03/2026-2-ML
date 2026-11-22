"""Repeated pinned-model loading must accept namespaces and reject collisions."""
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

from twinlite_morai.external_model import (
    ExternalTwinLiteError, PINNED_TWINLITE_COMMIT, load_external_twinlite,
)


class ExternalModelOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.saved_path = list(sys.path)
        self.saved_modules = {k: v for k, v in sys.modules.items()
                              if k == "model" or k.startswith("model.")}
        for name in self.saved_modules:
            del sys.modules[name]
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "reference"
        (self.root / "model").mkdir(parents=True)
        (self.root / ".git").mkdir()
        (self.root / ".git/HEAD").write_text(PINNED_TWINLITE_COMMIT + "\n")
        (self.root / "LICENSE").write_text("Fixture license\n")
        (self.root / "model/model.py").write_text(
            "import torch\nclass TwinLiteNetPlus(torch.nn.Module):\n"
            "    def __init__(self, args):\n"
            "        super().__init__()\n        self.config = args.config\n")
        importlib.invalidate_caches()

    def tearDown(self):
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                del sys.modules[name]
        sys.modules.update(self.saved_modules)
        sys.path[:] = self.saved_path
        importlib.invalidate_caches()
        self.directory.cleanup()

    def test_same_namespace_can_load_two_distinct_models(self):
        first, commit = load_external_twinlite(root=str(self.root))
        self.assertIsNone(sys.modules["model"].__file__)
        second, repeated_commit = load_external_twinlite(root=str(self.root))
        self.assertIsNot(first, second)
        self.assertEqual((first.config, second.config), ("medium", "medium"))
        self.assertEqual((commit, repeated_commit), (PINNED_TWINLITE_COMMIT,) * 2)

    def test_regular_package_can_also_reload(self):
        (self.root / "model/__init__.py").write_text("")
        first, _ = load_external_twinlite(root=str(self.root))
        second, _ = load_external_twinlite(root=str(self.root))
        self.assertIsNot(first, second)

    def test_foreign_namespace_is_rejected(self):
        foreign = ModuleType("model")
        foreign.__path__ = [str(Path(self.directory.name) / "foreign")]
        sys.modules["model"] = foreign
        with self.assertRaises(ExternalTwinLiteError):
            load_external_twinlite(root=str(self.root))

    def test_namespace_with_extra_foreign_location_is_rejected(self):
        foreign = ModuleType("model")
        foreign.__path__ = [str(self.root / "model"),
                            str(Path(self.directory.name) / "foreign")]
        sys.modules["model"] = foreign
        with self.assertRaises(ExternalTwinLiteError):
            load_external_twinlite(root=str(self.root))

    def test_orphaned_foreign_child_is_rejected(self):
        child = ModuleType("model.model")
        child.__file__ = str(Path(self.directory.name) / "foreign/model.py")
        sys.modules["model.model"] = child
        with self.assertRaises(ExternalTwinLiteError):
            load_external_twinlite(root=str(self.root))


if __name__ == "__main__":
    unittest.main()

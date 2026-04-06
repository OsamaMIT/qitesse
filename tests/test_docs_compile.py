import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class DocsCompileTests(unittest.TestCase):
    def test_sphinx_config_compiles(self):
        py_compile.compile(str(ROOT / "docs" / "conf.py"), doraise=True)

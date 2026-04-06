import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ExampleCompileTests(unittest.TestCase):
    def test_examples_compile(self):
        examples = [
            ROOT / "examples" / "simple.py",
            ROOT / "examples" / "h_example.py",
            ROOT / "examples" / "custom_unitary.py",
            ROOT / "examples" / "backend_adapter.py",
            ROOT / "examples" / "pennylane_poc.py",
            ROOT / "examples" / "vqe_loop.py",
        ]

        for path in examples:
            with self.subTest(path=path.name):
                py_compile.compile(str(path), doraise=True)

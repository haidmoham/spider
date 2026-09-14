"""Validate notebook source without executing user-owned learning work."""

from pathlib import Path
import unittest

import nbformat

NOTEBOOKS = Path(__file__).resolve().parents[1] / "lab" / "notebooks"


class NotebookSourceTests(unittest.TestCase):
    def test_notebooks_have_valid_structure_and_python_syntax(self):
        paths = sorted(NOTEBOOKS.glob("*.ipynb"))
        self.assertTrue(paths, "No notebooks found")
        for path in paths:
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        compile(cell.source, f"{path.name}:{cell.id}", "exec")


if __name__ == "__main__":
    unittest.main()

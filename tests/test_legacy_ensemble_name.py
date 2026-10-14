"""Naming-only regression without importing model or evaluation dependencies."""

import ast
import os
from pathlib import Path


def naming_function(dataset):
    path = (
        Path(__file__).resolve().parents[1]
        / "legacy"
        / "spatial"
        / dataset
        / "evaluate.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_ensemble_name"
    )
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["_build_ensemble_name"]


def test_cholect45_optional_ensemble_exports_use_existing_naming_rule():
    files = ["nested/2026_Student_B224.csv", "2026_Student_B384.csv", "plain.csv"]
    expected = "Student_B224__Student_B384__plain"
    assert naming_function("cholect45")(files) == expected
    assert naming_function("lls48")(files) == expected

import re
from os import path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _output_text(output):
    if hasattr(output, "text"):
        return output.text
    return getattr(output, "data", {}).get("text/plain", "")


def run_notebook(notebook):
    current_dir = path.dirname(path.realpath(__file__))
    file_path = path.join(current_dir, notebook)

    with open(file_path, encoding="utf-8") as inb_f:
        inb = nbformat.read(inb_f, as_version=4)

    exec_pre = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        exec_pre.preprocess(inb, {"metadata": {"path": current_dir}})
    except Exception as e:
        raise RuntimeError(f"Notebook execution failed: {e}") from e

    text = "\n".join(
        _output_text(output)
        for cell in inb.cells
        if cell.cell_type == "code" and "outputs" in cell
        for output in cell.outputs
    )
    text = _ANSI_ESCAPE.sub("", text)

    failed = re.search(r"^(FAILED|ERROR)\b", text, flags=re.MULTILINE)
    failed_count = re.search(r"\b[1-9]\d* (?:failed|error|errors)\b", text)
    if failed or failed_count:
        raise RuntimeError(f"Error in notebook cell:\n{text}")

    runs_tests = any(
        cell.cell_type == "code" and ("%%ipytest" in cell.source or "ipytest.run" in cell.source)
        for cell in inb.cells
    )
    if runs_tests and not re.search(r"\[100%\]|\b\d+ passed\b", text):
        raise RuntimeError(f"No tests were run in the notebook:\n{text}")

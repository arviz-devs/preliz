from os import path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook):
    current_dir = path.dirname(path.realpath(__file__))
    file_path = path.join(current_dir, notebook)
    with open(file_path, encoding="ascii") as inb_f:
        inb = nbformat.read(inb_f, as_version=4)
        exec_pre = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert exec_pre.preprocess(inb) is not None, f"Got empty notebook for {notebook}"
        except Exception:  # pylint: disable=broad-except
            assert False, f"Failed executing {notebook}"

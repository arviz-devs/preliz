from os import path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


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

    for cell in inb.cells:
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                try:
                    if "FAILED" in output.text:
                        raise RuntimeError(f"Error in notebook cell:\n{output.text}")
                except AttributeError:
                    pass

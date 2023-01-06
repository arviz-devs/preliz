from test_helper import run_notebook
from preliz.internal.plot_helper import check_inside_notebook


def test_check_inside_notebook_not(capsys):
    check_inside_notebook()
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.out


def test_check_inside_notebook_yes():
    run_notebook("check_inside_notebook.ipynb")

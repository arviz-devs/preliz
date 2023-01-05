from preliz.utils.utils import check_inside_notebook


def test_check_inside_notebook(capsys):
    check_inside_notebook()
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.out
    check_inside_notebook(need_widget=True)
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.out

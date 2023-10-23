from test_helper import run_notebook
from preliz.unidimensional.roulette import create_figure, create_grid, Rectangles, on_leave_fig


def test_roulette():
    run_notebook("roulette.ipynb")


def test_roulette_mock():
    x_min = 0
    x_max = 10
    ncols = 10
    nrows = 10

    fig, ax_grid, ax_fit = create_figure((10, 9))
    coll = create_grid(x_min, x_max, nrows, ncols, ax=ax_grid)
    grid = Rectangles(fig, coll, nrows, ncols, ax_grid)
    grid.weights = {0: 2, 1: 6, 2: 10, 3: 10, 4: 7, 5: 3, 6: 1, 7: 1, 8: 1, 9: 1}
    w_repr = "kde"
    distributions = ["Gamma", "LogNormal", "StudentT", "BetaScaled", "Normal"]

    for idx, dist in enumerate(distributions):
        w_distributions = distributions[idx:]

        fitted_dist = on_leave_fig(
            fig.canvas, grid, w_distributions, w_repr, x_min, x_max, ncols, "", ax_fit
        )
        assert fitted_dist.__class__.__name__ == dist

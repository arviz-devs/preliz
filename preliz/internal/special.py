# pylint: disable=invalid-name
import numba as nb


@nb.vectorize(nopython=True)
def half_erf(x):
    """
    Error function for values of x >= 0, return 0 otherwise
    Equations 7.1.27 from Abramowitz and Stegun
    Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
    """
    if x <= 0:
        return 0

    a1 = 0.0705230784
    a2 = 0.0422820123
    a3 = 0.0092705272
    a4 = 0.0001520143
    a5 = 0.0002765672
    a6 = 0.0000430638

    t = 1.0 / (1.0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5 + a6 * x**6)
    approx = 1 - t**16

    return approx

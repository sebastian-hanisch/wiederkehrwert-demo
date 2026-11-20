"""Zahlenwerkzeuge ohne scipy gegen scipy (nur im Test): Nelder-Mead, Brent, Gamma-Verteilung, Wilson-Band."""

import math

import pytest

import wkw_numeric as N

optimize = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")


def test_nelder_mead_findet_das_minimum_der_rosenbrock_funktion():
    f = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    best, fb, nfev = N.nelder_mead(f, [-1.2, 1.0], [0.3, 0.3], xatol=1e-9, fatol=1e-14, maxiter=4000, maxfev=8000)
    assert best == pytest.approx([1.0, 1.0], abs=1e-5) and fb < 1e-9 and nfev < 8000


def test_nelder_mead_dreidimensional_gegen_scipy():
    f = lambda v: (v[0] - 1) ** 2 + 2 * (v[1] + 2) ** 2 + 3 * (v[2] - 0.5) ** 2 + 0.3 * v[0] * v[1]
    best, fb, _ = N.nelder_mead(f, [0, 0, 0], [0.5, 0.5, 0.5], xatol=1e-9, fatol=1e-13)
    ref = optimize.minimize(f, [0, 0, 0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-13))
    assert fb == pytest.approx(ref.fun, abs=1e-9) and best == pytest.approx(list(ref.x), abs=1e-4)


def test_brent_gegen_scipy_und_vorzeichenfehler():
    for f, a, b in ((lambda x: x ** 3 - 2, 0, 3), (lambda x: math.cos(x) - x, 0, 1), (lambda x: math.exp(x) - 5, 0, 5), (lambda x: x - 1, 1, 2)):
        assert N.brent(f, a, b, xtol=1e-14) == pytest.approx(optimize.brentq(f, a, b, xtol=1e-14), abs=1e-12)
    with pytest.raises(ValueError):
        N.brent(lambda x: x * x + 1, -1, 1)


@pytest.mark.parametrize("m", [1, 2, 5, 20, 100, 500])
def test_gamma_verteilung_gegen_scipy(m):
    for x in (0.1 * m, 0.5 * m, m, 1.5 * m, m + 5 * math.sqrt(m)):
        assert N.gamma_cdf_int(m, x) == pytest.approx(stats.gamma.cdf(x, m), abs=1e-10)
    for p in (0.025, 0.5, 0.975):
        assert N.gamma_ppf_int(m, p) == pytest.approx(stats.gamma.ppf(p, m), rel=1e-9)
    assert N.gamma_cdf_int(m, 0.0) == 0.0


def test_wilson_band():
    lo, hi = N.wilson(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    lo, hi = N.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)                           # Wilson-Obergrenze bei 0 von 10 (Tafelwert)
    assert N.wilson(0, 0) == (0.0, 1.0)

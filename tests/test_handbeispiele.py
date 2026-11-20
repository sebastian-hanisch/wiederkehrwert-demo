"""Handbeispiele von Hand: exakte Brüche, Aufzählungen, Tafelwerte (Plan: H 1 bis H 5). Jede Erwartung steht als Rechnung im Test.

Zähleinheiten: Jahre, Reihen. Alle Modelle sind ausgedacht. H 4 ist der Ein-Parameter-Fall (σ bekannt): er prüft die Wartezeit-Arithmetik, die Intervall-Routinen im einfachsten
Modell und die Gamma-Überdeckung, NICHT die drei Intervalle der Dreiparameter-Anpassung (dafür tests/test_oracle_wiederkehrwert.py)."""

import itertools
import math
from fractions import Fraction

import pytest

import wkw_numeric as N
import wkw_returnlevel as RL

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")


# --- H 1: Wiederkehrperiode ist ein Mittel -----------------------------------------------------------------------------------------------------

def test_h1_wahrscheinlichkeit_in_T_jahren_exakt_als_bruch():
    assert RL.p_mindestens_einmal_bruch(2, 2) == Fraction(3, 4)
    assert RL.p_mindestens_einmal_bruch(10, 10) == Fraction(6513215599, 10 ** 10)
    assert float(RL.p_mindestens_einmal_bruch(100, 100)) == pytest.approx(0.633968, abs=1e-6)
    assert RL.p_mindestens_einmal(100, 100) == pytest.approx(float(RL.p_mindestens_einmal_bruch(100, 100)), rel=1e-13)


def test_h1_grenzwert_und_mittel_der_wartezeit():
    assert abs(RL.p_mindestens_einmal(10 ** 6, 10 ** 6) - (1 - math.exp(-1))) < 1e-6
    mittel = sum(k * 0.01 * 0.99 ** (k - 1) for k in range(1, 4001))
    assert mittel == pytest.approx(100.0, abs=1e-9)                                   # Mittel der geometrischen Wartezeit (p = 1/100) ist T


def test_h1_median_der_wartezeit_69_und_30_jahre():
    assert Fraction(99, 100) ** 69 < Fraction(1, 2) <= Fraction(99, 100) ** 68        # Handrechnung in Brüchen
    assert RL.median_wartezeit(100) == 69
    assert RL.median_wartezeit(1000) == 693
    assert RL.p_mindestens_einmal(100, 30) == pytest.approx(0.2603, abs=5e-5)


# --- H 2: Fünf Jahre, T = 10 ------------------------------------------------------------------------------------------------------------------

def test_h2_aufzaehlung_aller_32_ausgaenge():
    summe = Fraction(0)
    for muster in itertools.product((0, 1), repeat=5):
        k = sum(muster)
        if k >= 1:
            summe += Fraction(1, 10) ** k * Fraction(9, 10) ** (5 - k)
    assert summe == Fraction(40951, 100000)
    assert summe == 1 - Fraction(9, 10) ** 5
    assert RL.p_mindestens_einmal_bruch(10, 5) == summe


def test_h2_reihe_von_50_jahren_und_jahre_fuer_sicherheit():
    assert float(RL.p_mindestens_einmal_bruch(100, 50)) == pytest.approx(0.3950, abs=5e-5)
    assert float(RL.p_mindestens_einmal_bruch(1000, 50)) == pytest.approx(0.0488, abs=5e-5)
    for T, q, soll in ((100, Fraction(1, 2), 69), (100, Fraction(9, 10), 230), (100, Fraction(95, 100), 299), (100, Fraction(99, 100), 459), (1000, Fraction(9, 10), 2302), (10, Fraction(9, 10), 22)):
        assert RL.jahre_fuer_sicherheit(T, q) == soll
        assert RL.p_mindestens_einmal_bruch(T, soll) >= q > RL.p_mindestens_einmal_bruch(T, soll - 1)    # kleinste Länge mit der Sicherheit


def test_h2_sicherheit_ueber_nutzungsdauer_deich_beispiel():
    """EKM-Rechnung (Zahlen des Buchs nicht übernommen): Sicherheit 95 % in 50 Jahren verlangt T = 975,3; in 100 Jahren 1 950,1."""
    assert RL.T_fuer_sicherheit(0.95, 50) == pytest.approx(975.3, abs=0.05)
    assert RL.T_fuer_sicherheit(0.95, 100) == pytest.approx(1950.1, abs=0.05)
    assert 1.0 / RL.T_fuer_sicherheit(0.95, 50) == pytest.approx(1 - 0.95 ** (1 / 50), rel=1e-12)


# --- H 3: exakte Pegel aus F^n ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("T, v1", [(100, 10.50), (1000, 12.81)])
def test_h3_exponential_365_zwei_wege(T, v1):
    n = 365
    x = RL.pegel_exponential(T, n)
    bisek = RL.bisektion_pegel(lambda y: n * math.log1p(-math.exp(-y)), T, 0.0, 60.0)       # zweiter Weg: Bisektion auf F^365 selbst
    assert bisek == pytest.approx(x, abs=1e-9)
    assert abs(x - v1) < 0.005
    assert x == pytest.approx(stats.expon.isf(RL.tagesanteil(T, n)), abs=1e-9)                # dritter Weg: scipy
    gumbel = math.log(n) - math.log(-math.log(1 - 1 / T))
    assert abs(gumbel - x) < 0.01                                                               # Abstand 0,2707/n gilt hier


# --- H 4: fünf Jahresmaxima, Gumbel mit bekanntem σ = 1 ----------------------------------------------------------------------------------------

def test_h4_ml_und_pegel(maxima_h4):
    x = maxima_h4
    assert sum(Fraction(1, k) for k in (2, 3, 4, 5, 6)) / 5 == Fraction(29, 100)
    mu_hat = RL.gumbel_mu_dach(x, 1.0)
    assert mu_hat == pytest.approx(-math.log(0.29), abs=1e-12)
    assert mu_hat == pytest.approx(1.237874, abs=1e-6)
    nll = lambda mu: sum(v - mu for v in x) + sum(math.exp(-(v - mu)) for v in x)
    assert optimize.minimize_scalar(nll, bounds=(-2, 5), method="bounded", options={"xatol": 1e-12}).x == pytest.approx(mu_hat, abs=1e-6)
    assert stats.gumbel_r.fit(x, fscale=1.0)[0] == pytest.approx(mu_hat, abs=1e-6)
    assert sum(math.exp(-(v - mu_hat)) for v in x) == pytest.approx(5.0, abs=1e-12)           # beobachtete Information = m = 5, SE = 1/sqrt(5) = 0,4472
    y100 = -math.log(-math.log(0.99))
    assert y100 == pytest.approx(4.600149, abs=1e-6)
    assert round(y100, 4) == 4.6001                                                        # Verschiebung der drei Intervalle für μ auf Intervalle für x_100 (README)
    assert RL.gev_pegel(0.01, mu_hat, 1.0, 0.0) - mu_hat == pytest.approx(4.6001, abs=5e-5)
    assert RL.gev_pegel(0.01, mu_hat, 1.0, 0.0) == pytest.approx(5.838023, abs=1e-5)
    assert RL.gev_pegel(0.01, mu_hat, 1.0, 0.0) == pytest.approx(mu_hat + y100, abs=1e-12)


def test_h4_drei_intervalle_fuer_mu(maxima_h4):
    m = 5
    mu_hat = RL.gumbel_mu_dach(maxima_h4, 1.0)
    c = RL.gumbel_delta_faktor(m)
    assert (mu_hat - c, mu_hat + c) == pytest.approx((0.3614, 2.1144), abs=1e-4)                  # Delta
    r1, r2 = RL.gumbel_profil_faktoren(m)
    assert (mu_hat + math.log(r1), mu_hat + math.log(r2)) == pytest.approx((0.2123, 2.0030), abs=1e-4)   # Profil
    g_lo, g_hi = N.gamma_ppf_int(m, 0.025), N.gamma_ppf_int(m, 0.975)
    assert (g_lo, g_hi) == pytest.approx((1.6235, 10.2416), abs=1e-3)
    assert (mu_hat + math.log(g_lo / m), mu_hat + math.log(g_hi / m)) == pytest.approx((0.1130, 1.9549), abs=1e-4)     # exakt über den Pivot
    assert g_lo == pytest.approx(stats.gamma.ppf(0.025, m), rel=1e-9) and g_hi == pytest.approx(stats.gamma.ppf(0.975, m), rel=1e-9)
    crit = stats.chi2.ppf(0.95, 1)
    nll = lambda mu: sum(v - mu for v in maxima_h4) + sum(math.exp(-(v - mu)) for v in maxima_h4)
    for end in (mu_hat + math.log(r1), mu_hat + math.log(r2)):
        assert 2 * (nll(end) - nll(mu_hat)) == pytest.approx(crit, abs=1e-9)                       # Devianz an beiden Enden direkt aus der Likelihood


def test_h4_exakte_ueberdeckung_bei_m_5_und_streuband():
    delta, profil = RL.gumbel_ueberdeckung_exakt(5)
    assert delta == pytest.approx(0.9322, abs=5e-4) and profil == pytest.approx(0.9462, abs=5e-4)
    assert delta < profil < 0.95                                                                    # Delta schon im einfachsten Fall unter Profil, beide unter 95 %
    lo, hi = RL.gumbel_fehler_band(5)
    assert (lo, hi) == pytest.approx((-0.605, 0.931), abs=1e-3)
    for m in (5, 10, 20, 50, 500):                                                                   # Gamma-Verteilungsfunktion gegen scipy
        c = RL.gumbel_delta_faktor(m)
        ref = stats.gamma.cdf(m * math.exp(c), m) - stats.gamma.cdf(m * math.exp(-c), m)
        assert RL.gumbel_ueberdeckung_exakt(m)[0] == pytest.approx(ref, abs=1e-9)


# --- H 5: Verlängern ----------------------------------------------------------------------------------------------------------------------------

def test_h5_bei_bekannter_skala_wandert_das_intervall_nur(maxima_h4):
    mu_hat = RL.gumbel_mu_dach(maxima_h4, 1.0)
    y100, y1000 = -math.log(-math.log(0.99)), -math.log(-math.log(0.999))
    assert y1000 == pytest.approx(6.907255, abs=1e-5)
    assert RL.gev_pegel(0.001, mu_hat, 1.0, 0.0) - RL.gev_pegel(0.01, mu_hat, 1.0, 0.0) == pytest.approx(y1000 - y100, abs=1e-12)
    assert y1000 - y100 == pytest.approx(2.3071, abs=1e-4)

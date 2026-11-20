"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): Pegel, Anpassung und Intervalle gegen einen ANDEREN Rechenweg.

(a) Pegel-Formel gegen `scipy.stats.genextreme.isf` mit c = -ξ (scipy nur hier);
(b) wahre Pegel aus F^n: Umkehrformel gegen Bisektion auf F^n und gegen scipy;
(c) Reihe erreicht den Pegel: Simulation gegen 1 - (1 - 1/T)^m als Band (|z| < 4, 15 Zellen);
(d) Ein-Parameter-Fall: allgemeine Delta- und Profil-Routine (numerische Ableitung, Wurzelsuche auf der Likelihood) gegen die geschlossenen Endpunkte und die Gamma-Überdeckung;
(e) Drei-Parameter-Anpassung gegen `genextreme.fit` (Minimum mindestens so tief) und Profil-Endpunkte mit Devianz 3,8415 über eine unabhängige Mehrfachstart-Minimierung;
(f) verteilungsfreie Probe: P(Maximum von m Blöcken ≤ Pegel) = (1 - p)^m = Beta(m, 1);
dazu die Ziehungen gegen die exakten Verteilungen (Kolmogorov-Abstand). Zufallsanteile als Band, nie als letztes Bit."""

import math

import numpy as np
import pytest

import wkw_evaluation as EV
import wkw_gev as G
import wkw_numeric as N
import wkw_returnlevel as RL
import wkw_scenario as S
from conftest import stichprobe
from wkw_rng import SplitMix64

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")


# --- (a) Pegel-Formel -------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("xi", [-0.4, -0.1, 0.0, 1e-12, 0.2, 0.5])
@pytest.mark.parametrize("T", [2, 10, 100, 1000])
def test_a_pegel_formel_gleich_genextreme_mit_c_gleich_minus_xi(xi, T):
    mu, sigma = 3.0, 2.0
    ref = stats.genextreme.isf(1.0 / T, c=-xi, loc=mu, scale=sigma)
    assert RL.gev_pegel(1.0 / T, mu, sigma, xi) == pytest.approx(ref, rel=1e-10)
    assert RL.gev_cdf(RL.gev_pegel(1.0 / T, mu, sigma, xi), mu, sigma, xi) == pytest.approx(1 - 1.0 / T, abs=1e-12)           # Rundlauf, nicht der Beweis


def test_a_vorzeichen_der_linie_xi_positiv_ist_fréchet():
    """ξ > 0 hat unendlichen oberen Rand (Pegel wächst weiter), ξ < 0 einen endlichen Rand μ - σ/ξ."""
    assert RL.gev_pegel(1e-9, 0, 1, 0.3) > 30
    assert RL.gev_pegel(1e-12, 0, 1, -0.4) < 2.5 and RL.gev_pegel(1e-12, 0, 1, -0.4) == pytest.approx(2.5, abs=1e-3)
    assert stats.genextreme.ppf(1 - 1e-12, c=0.4) == pytest.approx(2.5, abs=1e-3)                                               # c = -ξ = 0,4 -> Rand bei 1/0,4


# --- (b) exakte Pegel aus F^n ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("T", [2, 10, 100, 1000, 10000])
def test_b_exponential_und_normal_365_zwei_wege(T):
    n = 365
    x = RL.pegel_exponential(T, n)
    assert RL.bisektion_pegel(lambda y: n * math.log1p(-math.exp(-y)), T, 0.0, 60.0) == pytest.approx(x, abs=1e-9)
    assert x == pytest.approx(stats.expon.isf(RL.tagesanteil(T, n)), abs=1e-9)
    xn = RL.pegel_normal(T, n)
    assert RL.bisektion_pegel(lambda y: n * math.log1p(-stats.norm.sf(y)), T, -5.0, 12.0) == pytest.approx(xn, abs=1e-9)
    assert xn == pytest.approx(stats.norm.isf(RL.tagesanteil(T, n)), abs=1e-9)


def test_b_handwerte_exakter_pegel():
    assert RL.pegel_exponential(100) == pytest.approx(10.500, abs=5e-4) and RL.pegel_exponential(1000) == pytest.approx(12.807, abs=5e-4)
    assert RL.pegel_normal(100) == pytest.approx(4.033, abs=5e-4) and RL.pegel_normal(1000) == pytest.approx(4.545, abs=5e-4)
    assert S.wahrer_pegel(S.Fall("vwz"), 100) == pytest.approx(155.128, abs=5e-4) and S.wahrer_pegel(S.Fall("vwz"), 1000) == pytest.approx(287.123, abs=5e-4)


# --- (c) Reihe erreicht den Pegel: Simulation gegen die verteilungsfreie Formel ----------------------------------------------------------------

def test_c_reihe_erreicht_pegel_simulation_im_band():
    zellen = [(T, m) for T in (10, 100, 1000) for m in (20, 50, 100, 200, 500)]          # 15 Zellen
    for T, m in zellen:
        R = 4000
        erste = EV.simuliere_reihen(T, m, R, 7000 + T + m)
        k = sum(1 for f in erste if f > 0)
        p = RL.p_mindestens_einmal(T, m)
        se = math.sqrt(p * (1 - p) / len(erste))
        assert abs(k / len(erste) - p) < 4 * max(se, 1e-4), (T, m, k / len(erste), p)


def test_c_wahrer_pegel_wird_in_diesem_anteil_der_reihen_erreicht():
    """Verteilungsfrei, gegen die Ziehung selbst: Jahresmaxima aus GEV, der exakte Pegel x_T: Anteil der Reihen mit Maximum >= x_T gegen 1 - (1 - 1/T)^m."""
    T, m, R = 100, 50, 2500
    fall = S.Fall("gev", 0.2, 100.0, 20.0)
    xt = S.wahrer_pegel(fall, T)
    rng = SplitMix64(S.stream_seed(5, 5, 5))
    k = sum(1 for _ in range(R) if max(S.ziehe_bloecke(fall, rng, m)) >= xt)
    p = RL.p_mindestens_einmal(T, m)
    assert abs(k / R - p) < 4 * math.sqrt(p * (1 - p) / R)


# --- (d) Ein-Parameter-Fall: allgemeine Routinen gegen geschlossene Endpunkte -------------------------------------------------------------------

def _nll_mu(x):
    return lambda mu: G.nll((mu, 0.0, 0.0), np.array(x))                       # σ = 1 (ln σ = 0), ξ = 0: Gumbel-Zweig der Likelihood


@pytest.mark.parametrize("m", [5, 20, 50])
def test_d_allgemeine_delta_und_profil_gegen_geschlossene_endpunkte(m):
    for k in range(40):
        x = np.array(S.ziehe_bloecke(S.Fall("gev", 0.0, 0.0, 1.0), SplitMix64(S.stream_seed(61, m, k)), m))
        f = _nll_mu(x)
        mu_hat = RL.gumbel_mu_dach(list(x), 1.0)
        # Delta: numerische zweite Ableitung der negativen Log-Likelihood (beobachtete Information) -> SE
        h = 1e-4
        info = (f(mu_hat + h) - 2 * f(mu_hat) + f(mu_hat - h)) / h ** 2
        assert info == pytest.approx(m, rel=1e-5)                                    # beobachtete Information ist exakt m
        assert 1.959963984540054 / math.sqrt(info) == pytest.approx(RL.gumbel_delta_faktor(m), rel=1e-5)
        # Profil: Devianz-Wurzeln per Brent auf der Likelihood selbst
        crit = 3.841458820694124
        g = lambda mu: 2 * (f(mu) - f(mu_hat)) - crit
        lo = N.brent(g, mu_hat - 8.0, mu_hat, xtol=1e-12)
        hi = N.brent(g, mu_hat, mu_hat + 8.0, xtol=1e-12)
        r1, r2 = RL.gumbel_profil_faktoren(m)
        assert lo == pytest.approx(mu_hat + math.log(r1), abs=1e-6) and hi == pytest.approx(mu_hat + math.log(r2), abs=1e-6)
        # ML selbst: zweiter Weg über die Minimierung
        assert optimize.minimize_scalar(f, bounds=(mu_hat - 3, mu_hat + 3), method="bounded", options={"xatol": 1e-12}).x == pytest.approx(mu_hat, abs=1e-6)
        assert G.fit_mu_bekannt(x, 1.0, 0.0) == pytest.approx(mu_hat, abs=1e-6)


@pytest.mark.parametrize("m", [5, 20, 50, 500])
def test_d_ueberdeckung_exakt_gegen_scipy_gamma_und_simulation(m):
    delta, profil = RL.gumbel_ueberdeckung_exakt(m)
    c = 1.959963984540054 / math.sqrt(m)
    ref_d = stats.gamma.cdf(m * math.exp(c), m) - stats.gamma.cdf(m * math.exp(-c), m)
    assert delta == pytest.approx(ref_d, abs=1e-9)
    r1, r2 = RL.gumbel_profil_faktoren(m)
    ref_p = stats.gamma.cdf(m * r2, m) - stats.gamma.cdf(m * r1, m)
    assert profil == pytest.approx(ref_p, abs=1e-9)
    assert delta < profil < 0.95                                                       # in allen vier m: Delta unter Profil, beide unter nominell
    if m <= 20:                                                                         # Simulation mit SplitMix64 als Band
        R = 3000
        k_d = k_p = 0
        for k in range(R):
            x = np.array(S.ziehe_bloecke(S.Fall("gev", 0.0, 0.0, 1.0), SplitMix64(S.stream_seed(62, m, k)), m))
            fehler = RL.gumbel_mu_dach(list(x), 1.0) - 0.0                             # mu_hat - mu (mu = 0)
            k_d += abs(fehler) <= c
            k_p += (math.log(r1) <= -fehler <= math.log(r2))
        for kk, p in ((k_d, delta), (k_p, profil)):
            assert abs(kk / R - p) < 4 * math.sqrt(p * (1 - p) / R)


def test_d_streuband_gegen_gamma_quantile():
    """Fehler μ̂ - μ in Einheiten σ: 5 % bis 95 % gleich -ln(G/m)-Quantilen (Simulation, m = 50)."""
    m, R = 50, 1500
    fehler = sorted(RL.gumbel_mu_dach(list(S.ziehe_bloecke(S.Fall("gev", 0.0, 0.0, 1.0), SplitMix64(S.stream_seed(63, k)), m)), 1.0) for k in range(R))
    lo, hi = RL.gumbel_fehler_band(m)
    q = lambda p: fehler[int(p * R)]
    assert q(0.05) == pytest.approx(lo, abs=0.04) and q(0.95) == pytest.approx(hi, abs=0.04)
    assert (lo, hi) == pytest.approx((-0.218, 0.249), abs=2e-3)
    ref = stats.gamma.ppf([0.95, 0.05], m) / m
    assert (-math.log(ref[0]), -math.log(ref[1])) == pytest.approx((lo, hi), abs=1e-9)


# --- (e) Drei-Parameter-Anpassung ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_e_ml_findet_mindestens_so_tiefes_minimum_wie_scipy(xi):
    for k in range(8):
        x = stichprobe("gev", xi, 100, 71, k)
        mu, s, xh, f, ok = G.fit_ml(x)
        c, loc, sc = stats.genextreme.fit(x)
        f_scipy = -np.sum(stats.genextreme.logpdf(x, c, loc=loc, scale=sc))
        assert f <= f_scipy + 1e-6, (xi, k, f, f_scipy)
        assert abs(xh - (-c)) < 0.02 or f < f_scipy - 1e-9                           # gleiches Optimum oder tieferes
    assert ok


def test_e_eigene_likelihood_gleich_scipy_logpdf_mit_c_minus_xi():
    for k in range(20):
        xi = (-0.4, -0.1, 0.0, 0.3)[k % 4]
        x = stichprobe("gev", xi, 40, 72, k)
        th = (0.1, math.log(1.1), xi + 0.03)
        assert G.nll(th, x) == pytest.approx(-np.sum(stats.genextreme.logpdf(x, -th[2], loc=th[0], scale=math.exp(th[1]))), abs=1e-9)


def _profil_unabhaengig(z, x, p, ls0, xi0):
    """Zweiter Weg: Mehrfachstart-Nelder-Mead aus scipy über (ln σ, ξ), μ aus z (eigene Formel hier neu geschrieben)."""
    y = -math.log1p(-p)

    def f(par):
        ls, xi = par
        if xi < -1.0:
            return 1e300
        s = math.exp(ls)
        mu = z + s * math.log(y) if abs(xi) < 1e-9 else z - s * (y ** (-xi) - 1) / xi
        return G.nll((mu, ls, xi), x)
    grid = [(ls0 + dl, xi) for dl in (-0.5, 0.0, 0.5) for xi in (-0.9, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3)] + [(ls0, xi0)]
    starts = sorted(grid, key=f)[:4]
    return min(optimize.minimize(f, st, method="Nelder-Mead", options=dict(initial_simplex=np.vstack([st, np.add(st, [0.2, 0]), np.add(st, [0, 0.1])]), xatol=1e-8, fatol=1e-11, maxiter=2000)).fun for st in starts)


@pytest.mark.parametrize("xi, m", [(0.2, 50), (-0.3, 20), (0.0, 20), (0.2, 20)])
def test_e_profil_endpunkte_haben_devianz_3_8415(xi, m):
    for k in range(3):
        x = stichprobe("gev", xi, m, 73, m, k)
        mu, s, xh, f, ok = G.fit_ml(x)
        p = 0.01
        d = G.delta_intervall(x, mu, s, xh, p)
        lo, hi = G.profil_intervall(x, mu, s, xh, f, p, se=None if d is None else d[2])
        for ende in (lo, hi):
            if math.isfinite(ende):
                dev = 2 * (_profil_unabhaengig(ende, x, p, math.log(s), xh) - f)
                assert dev == pytest.approx(3.8415, abs=3e-3), (xi, m, k, ende, dev)
        z_hat = RL.gev_pegel(p, mu, s, xh)
        assert lo < z_hat < hi
        if d is not None:
            assert d[0] < z_hat < d[1] and (d[0] + d[1]) / 2 == pytest.approx(z_hat, rel=1e-9)        # Delta symmetrisch um den Schätzwert


def test_e_delta_gegen_unabhaengige_hesse_mit_scipy_ableitungen():
    """Delta-SE ein zweites Mal: Hesse-Matrix aus 4-Punkt-Differenzen mit anderer Schrittweite, Gradient des Pegels per Complex-Step-freier Differenz mit anderer Schrittweite."""
    x = stichprobe("gev", 0.2, 100, 74, 0)
    mu, s, xh, f, ok = G.fit_ml(x)
    p = 0.01
    th = np.array([mu, math.log(s), xh])
    h = np.array([1e-3 * s, 1e-3, 1e-3])
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.zeros(3), np.zeros(3)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = (G.nll(th + ei + ej, x) - G.nll(th + ei - ej, x) - G.nll(th - ei + ej, x) + G.nll(th - ei - ej, x)) / (4 * h[i] * h[j])
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h[i]
        g[i] = (G._pegel_theta(th + e, p) - G._pegel_theta(th - e, p)) / (2 * h[i])
    se = math.sqrt(g @ np.linalg.inv(H) @ g)
    assert G.delta_intervall(x, mu, s, xh, p)[2] == pytest.approx(se, rel=2e-3)


# --- Ziehungen gegen die exakten Verteilungen -----------------------------------------------------------------------------------------------

def _ks(werte, cdf):
    v = np.sort(np.asarray(werte, float))
    n = v.size
    F = cdf(v)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(0, n) / n)))


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_ziehung_gev_gegen_cdf(xi):
    n = 4000
    x = stichprobe("gev", xi, n, 75, int(xi * 10) + 5, mu=2.0, sigma=3.0)
    assert _ks(x, lambda v: stats.genextreme.cdf(v, c=-xi, loc=2.0, scale=3.0)) < 1.63 / math.sqrt(n)


def test_ziehung_exponential_und_normal_365_gegen_f_hoch_n():
    n = 4000
    assert _ks(stichprobe("exp365", 0, n, 76), lambda v: stats.expon.cdf(v) ** 365) < 1.63 / math.sqrt(n)
    assert _ks(stichprobe("normal365", 0, n, 77), lambda v: stats.norm.cdf(v) ** 365) < 1.63 / math.sqrt(n)
    # Nebenbefund: die exakten Pegel stehen auf dieser Verteilung
    assert (stats.expon.cdf(RL.pegel_exponential(100)) ** 365) == pytest.approx(0.99, abs=1e-12)


# --- (f) verteilungsfreie Probe ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2, 20, 50])
@pytest.mark.parametrize("p", [0.1, 0.01, 0.001])
def test_f_maximum_von_m_bloecken_ist_beta_m_1(m, p):
    assert stats.beta.cdf(1 - p, m, 1) == pytest.approx((1 - p) ** m, rel=1e-12)
    assert 1 - stats.beta.cdf(1 - p, m, 1) == pytest.approx(RL.p_mindestens_einmal(1 / p, m), rel=1e-9)


def test_f_rand_der_daten_exakt_1_durch_m_plus_1_fuer_stetige_verteilungen():
    """Das Maximum von m + 1 stetigen iid-Werten steht mit Wahrscheinlichkeit 1/(m + 1) an der letzten Stelle (Aufzählung der Rangordnungen für m = 4)."""
    import itertools
    from fractions import Fraction
    m = 4
    ordn = list(itertools.permutations(range(m + 1)))
    letzte_ist_max = sum(1 for o in ordn if o[-1] == m)
    assert Fraction(letzte_ist_max, len(ordn)) == Fraction(1, m + 1)
    assert RL.p_rand_der_daten(m) == pytest.approx(float(Fraction(letzte_ist_max, len(ordn))), rel=1e-12)


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2, 0.4])
def test_d_ml_fuer_mu_bei_bekanntem_sigma_und_xi_gegen_scipy(xi):
    """Ein-Parameter-ML für alle vier Vorzeichen-Fälle (nicht nur Gumbel): goldener Schnitt gegen `minimize_scalar` auf derselben Likelihood."""
    for k in range(10):
        x = stichprobe("gev", xi, 30, 64, k, mu=100.0, sigma=20.0)
        mu = G.fit_mu_bekannt(x, 20.0, xi)
        f = lambda m_: G.nll((m_, math.log(20.0), xi), x)
        ref = optimize.minimize_scalar(f, bounds=(mu - 5, mu + 5), method="bounded", options={"xatol": 1e-10}).x
        assert mu == pytest.approx(ref, abs=1e-5)

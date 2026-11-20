"""Eigenschaften über je 15 bis 60 Zufallsinstanzen mit festem Zufalls-Seed: Intervalle, Monotonie, Äquivarianz, Schrumpfen des Streubands mit m."""

import math

import numpy as np
import pytest

import wkw_gev as G
import wkw_returnlevel as RL
import wkw_scenario as S
import wkw_study as ST
from conftest import stichprobe


@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.2])
def test_intervalle_enthalten_den_schaetzwert_und_profil_waechst_mit_T(xi):
    for k in range(15):
        x = stichprobe("gev", xi, 40, 91, k)
        mu, s, xh, f, ok = G.fit_ml(x)
        letzte_ober = None
        for T in (10, 100, 1000):
            p = 1.0 / T
            z = RL.gev_pegel(p, mu, s, xh)
            d = G.delta_intervall(x, mu, s, xh, p)
            if d is not None:
                assert d[0] < z < d[1]
            lo, hi = G.profil_intervall(x, mu, s, xh, f, p, se=None if d is None else d[2])
            assert lo < z < hi
            if letzte_ober is not None:
                assert hi >= letzte_ober - 1e-6                                           # die Obergrenze wächst mit T
            letzte_ober = hi


def test_pegel_waechst_in_T_und_wartezeit_wahrscheinlichkeit_faellt_gegen_1_minus_1_durch_e():
    for k in range(25):
        x = stichprobe("gev", (-0.3, 0.0, 0.2)[k % 3], 30, 92, k)
        mu, s, xh, f, ok = G.fit_ml(x)
        werte = [RL.gev_pegel(1.0 / T, mu, s, xh) for T in (2, 5, 10, 50, 100, 1000)]
        assert all(a < b for a, b in zip(werte, werte[1:]))
    ps = [RL.p_mindestens_einmal(T, T) for T in (2, 5, 10, 50, 100, 1000, 10 ** 5)]
    assert all(a > b for a, b in zip(ps, ps[1:])) and all(p > 1 - math.exp(-1) for p in ps)
    assert all(RL.median_wartezeit(T) <= T for T in (2, 3, 5, 10, 100, 1000))
    assert all(RL.median_wartezeit(T) < T for T in (5, 10, 100, 1000))                  # Median unter dem Mittel (geometrische Wartezeit)


def test_anpassung_ist_aequivariant_bei_ort_und_skala():
    """Ort und Skala der Daten ändern die Intervalle nur mit: darum hängt die Überdeckung nicht an (μ, σ), das Streuband von x̂/x_T schon."""
    for k in range(6):
        x = stichprobe("gev", 0.2, 30, 93, k)
        y = 100.0 + 20.0 * x
        a, b = G.fit_ml(x), G.fit_ml(y)
        assert b[0] == pytest.approx(100.0 + 20.0 * a[0], abs=2e-4 * 20) and b[1] == pytest.approx(20.0 * a[1], rel=2e-4) and b[2] == pytest.approx(a[2], abs=2e-4)
        p = 0.01
        da, db = G.delta_intervall(x, *a[:3], p), G.delta_intervall(y, *b[:3], p)
        assert db[0] == pytest.approx(100.0 + 20.0 * da[0], abs=1e-3 * 20) and db[1] == pytest.approx(100.0 + 20.0 * da[1], abs=1e-3 * 20)
        pa, pb = G.profil_intervall(x, *a[:3], a[3], p), G.profil_intervall(y, *b[:3], b[3], p)
        assert pb[0] == pytest.approx(100.0 + 20.0 * pa[0], abs=2e-3 * 20) and pb[1] == pytest.approx(100.0 + 20.0 * pa[1], abs=2e-3 * 20)


def test_streuband_schrumpft_mit_m_im_mittel_ueber_laeufe():
    breiten = {}
    fall = S.Fall("gev", 0.2, 100.0, 20.0)
    x_wahr = S.wahrer_pegel(fall, 100)
    for m in (20, 100, 500):
        laeufe = [ST.lauf_streuband(fall, m, (100,), (94, m, r)) for r in range(60)]
        v = [r["T"][100]["z"] / x_wahr for r in laeufe if r is not None]
        breiten[m] = np.quantile(v, 0.95) - np.quantile(v, 0.05)
    assert breiten[20] > breiten[100] > breiten[500]


def test_bei_bekanntem_sigma_ist_der_fehler_in_sigma_fuer_jedes_T_gleich():
    """σ und ξ bekannt: der Fehler in Einheiten σ ist für jedes T derselbe (z_T - x_T gleich für alle T), die Extrapolation von T = 100 auf 1 000 kostet nichts."""
    fall = S.Fall("gev", 0.0, 100.0, 20.0)
    for r in range(20):
        res = ST.lauf_streuband(fall, 20, (10, 100, 1000), (95, r), bekannt=True)
        fehler = [(res["T"][T]["z"] - S.wahrer_pegel(fall, T)) / 20.0 for T in (10, 100, 1000)]
        assert max(fehler) - min(fehler) < 1e-6

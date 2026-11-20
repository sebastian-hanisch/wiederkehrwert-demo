"""Randfälle (DEMO-PLAYBOOK §12): m = 1 und 2, T = 2, T kleiner als m, ξ am Rand -1, Hesse-Matrix nicht positiv definit, Profil nach oben offen, R = 1, nicht endliche Eingaben,
Vehikel B mit m = 1, alle Blöcke gleich. Jede Zeile prüft einen Zweig der eigenen Routinen."""

import math

import numpy as np
import pytest

import wkw_evaluation as EV
import wkw_gev as G
import wkw_returnlevel as RL
import wkw_scenario as S
import wkw_study as ST
from conftest import stichprobe
from wkw_rng import SplitMix64


@pytest.mark.parametrize("m", [1, 2])
def test_anpassung_mit_ein_und_zwei_bloecken_ist_unmoeglich(m):
    with pytest.raises(ValueError):
        G.fit_ml(np.arange(float(m)))
    r = EV.einzelreihe(EV.Settings(m=m))
    assert "fehler" in r and "keine Anpassung" in r["fehler"] and "mindestens drei" in r["fehler"]
    assert "fehler" in EV.streuband_live(EV.Settings(m=m))


@pytest.mark.parametrize("m", [3, 4])
def test_anpassung_mit_drei_und_vier_bloecken_rechnet_ohne_fehler(m):
    x = stichprobe("gev", 0.2, m, 81, m)
    mu, s, xi, f, ok = G.fit_ml(x)
    assert math.isfinite(mu) and s > 0 and math.isfinite(f)


def test_alle_bloecke_gleich_und_nicht_endliche_eingabe():
    with pytest.raises(ValueError):
        G.fit_ml(np.full(10, 3.0))
    with pytest.raises(ValueError):
        G.fit_ml(np.array([1.0, 2.0, float("nan"), 4.0]))
    assert G.nll((float("nan"), 0.0, 0.1), np.array([1.0, 2.0, 3.0])) == G.HUGE
    assert G.nll((0.0, 0.0, -1.5), np.array([1.0, 2.0, 3.0])) == G.HUGE                  # ξ unter -1 ist ausgeschlossen
    assert G.nll((0.0, 0.0, 0.5), np.array([-5.0, 2.0, 3.0])) == G.HUGE                  # Wert außerhalb des Trägers


def test_T_gleich_2_und_T_kleiner_als_m():
    assert RL.p_mindestens_einmal(2, 1) == 0.5 and RL.p_mindestens_einmal(2, 2) == 0.75
    assert RL.median_wartezeit(2) == 2                                                    # P(W <= 1) = 1/2 ist nicht größer als 1/2: erst k = 2 überschreitet die Hälfte
    assert RL.gev_pegel(0.5, 0.0, 1.0, 0.0) == pytest.approx(-math.log(math.log(2)), abs=1e-12)
    r = EV.einzelreihe(EV.Settings(T=10, m=500, seed=3))
    assert "fehler" not in r and r["T"][10]["z"] > 0                                      # T = 10 bei 500 Blöcken
    assert RL.p_mindestens_einmal(10, 500) == pytest.approx(1.0, abs=1e-20)


def test_T_gleich_1_ist_kein_pegel():
    assert RL.p_mindestens_einmal(1, 1) == 1.0
    with pytest.raises(ValueError):
        RL.gev_pegel(1.0, 0.0, 1.0, 0.0)                                                    # Überschreitungswahrscheinlichkeit 1 hat keinen Pegel
    with pytest.raises(ValueError):
        RL.p_mindestens_einmal(0.5, 3)


def test_xi_am_rand_minus_1_bricht_nicht_ab():
    x = stichprobe("gev", -1.0, 30, 82)
    mu, s, xi, f, ok = G.fit_ml(x)
    assert xi >= -1.0 - 1e-9 and math.isfinite(f)
    assert ok in (True, False)
    d = G.delta_intervall(x, mu, s, xi, 0.01)
    assert d is None or d[0] <= d[1]                                                          # bei klebendem ξ̂ darf das Intervall entarten (Breite 0), die App warnt dann mit „ξ̂ klebt an der Grenze“


def test_hesse_matrix_nicht_positiv_definit_gibt_kein_delta_intervall():
    x = stichprobe("gev", 0.2, 50, 83)
    assert G.delta_intervall(x, 5.0, 0.2, 0.9, 0.01) is None                                # weit weg vom Optimum: Hesse nicht endlich oder nicht positiv definit
    mu, s, xi, f, ok = G.fit_ml(x)
    th = np.array([mu, math.log(s), xi])
    assert np.all(np.linalg.eigvalsh(G.hesse_nll(th, x)) > 0)                                # am Optimum positiv definit


def test_profil_nach_oben_offen_wird_als_unendlich_ausgewiesen():
    x = stichprobe("gev", 0.2, 20, 84)
    mu, s, xi, f, ok = G.fit_ml(x)
    lo, hi = G.profil_intervall(x, mu, s, xi, f, 0.001, max_expand=1)                       # Suchbereich zu klein: Obergrenze offen
    assert hi == math.inf and lo <= RL.gev_pegel(0.001, mu, s, xi)
    text = EV.verdict_einzelreihe(1000, 5.0, 4.0, {"Profil": (lo, hi)}, 20)[1]
    assert "offen" in text


def test_R_gleich_1_probe():
    p = EV.probe_laeufe(EV.Settings(R4=1, m=20, delta=True, profil=True, boot=False))
    assert p["R"] == 1
    for art in ("delta", "profil"):
        e = p["arten"][art]
        assert e["R"] == 1 and e["cov"] in (0.0, 1.0) and e["wilson"][0] < e["wilson"][1]
    assert "1 von 1" in EV.verdict_probe(p)[1] or "0 von 1" in EV.verdict_probe(p)[1]


def test_vehikel_b_mit_einem_block():
    rng = SplitMix64(5)
    assert len(S.ziehe_bloecke(S.Fall("vwz"), rng, 1)) == 1
    assert "fehler" in EV.einzelreihe(EV.Settings(fall="vwz", m=1))


def test_keine_intervalle_gewaehlt():
    p = EV.probe_laeufe(EV.Settings(R4=2, m=20, delta=False, profil=False, boot=False))
    assert p["arten"] == {}
    assert EV.verdict_probe(p) == ("info", "Keine Intervallart gewählt.")


def test_bewerte_intervalle_mit_fehlenden_und_offenen_intervallen():
    laeufe = [{"T": {10: {"delta": iv}}} for iv in ((1.0, 3.0), (4.0, 5.0), (0.0, 2.0), None, (-math.inf, 10.0), (2.0, math.inf))]
    e = ST.bewerte_intervalle(laeufe, 10, "delta", 2.5)
    assert (e["R"], e["cov"] * 6, e["hoch"] * 6, e["niedrig"] * 6, e["ohne"] * 6, e["offen_oben"] * 6) == pytest.approx((6, 3, 1, 1, 1, 1))
    assert e["breite_median"] == pytest.approx(2.0)                                         # endliche Breiten 2, 1, 2 -> Median 2

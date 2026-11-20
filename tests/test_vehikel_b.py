"""Vehikel B: die kopierte Gewichtstabelle gegen eine eigene Aufzählung aus der Definition, die exakten Pegel, die Ziehung des Tagesmaximums.

Definition (ausgedacht, Verweilzeit-Mischung der Linie): Art (Import 0,35 / Export 0,30 / Umschlag 0,25 / Leer 0,10) x Kunde (12, gleich wahrscheinlich, Effekt UK[kunde] * U_SCALE[art])
x Zoll (nur Import, Anteil 0,25) x Wochenende (Anteil 2/7, wirkt nur auf Import und Leer) x Vorlauf (Export gleichverteilt (1, 10), Umschlag (0,5, 6): Gauß-Legendre mit 48 Knoten);
log(Verweilzeit) je Komponente normal mit μ = ln(Median) + UK * U_SCALE und s = S_BASE[art]. Hier wird die Aufzählung neu geschrieben, mit eigenen Schleifen und scipy für die Normalverteilung."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import wkw_vehikel_b as VB
from conftest import WURZEL
from wkw_rng import SplitMix64, exact_log

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")

ART_P = (0.35, 0.30, 0.25, 0.10)
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)


def eigene_aufzaehlung():
    """(Zahl der Schritte, {(μ gerundet, s): Gewicht}) aus der Definition."""
    knoten, gew = np.polynomial.legendre.leggauss(48)
    schritte = 0
    tab = {}
    for art in range(4):
        for kunde in range(12):
            for zoll, p_zoll in ((0, 0.75), (1, 0.25)) if art == 0 else ((0, 1.0), (1, 0.0)):
                for we, p_we in ((0, 5 / 7), (1, 2 / 7)):
                    if art in (1, 2):
                        lo, hi = (1.0, 10.0) if art == 1 else (0.5, 6.0)
                        vor = [lo + (hi - lo) * (k + 1) / 2 for k in knoten]
                        wv = [w / 2 for w in gew]
                    else:
                        vor, wv = [0.0], [1.0]
                    for v, w in zip(vor, wv):
                        schritte += 1
                        gewicht = ART_P[art] / 12 * p_zoll * p_we * w
                        if gewicht <= 0:
                            continue
                        if art == 0:
                            med = 3.5 * math.exp(0.7 * zoll + 0.3 * we)
                        elif art == 1:
                            med = 0.6 + 0.85 * v
                        elif art == 2:
                            med = 0.3 + 0.8 * v
                        else:
                            med = 7.0 * math.exp(0.3 * we)
                        schluessel = (round(math.log(med) + UK[kunde] * U_SCALE[art], 12), S_BASE[art])
                        tab[schluessel] = tab.get(schluessel, 0.0) + gewicht
    return schritte, tab


@pytest.fixture(scope="module")
def aufzaehlung():
    return eigene_aufzaehlung()


def test_tabelle_gleich_eigener_aufzaehlung(aufzaehlung):
    schritte, tab = aufzaehlung
    daten = json.loads(VB.TABELLE.read_text(encoding="utf-8"))
    assert schritte == 4704
    assert daten["n_komponenten"] == len(daten["komponenten"]) == len(tab) == 1224
    assert sum(1 for k in daten["komponenten"]) == 1224
    kopie = {(round(mu, 12), s): w for w, mu, s in daten["komponenten"]}
    assert set(kopie) == set(tab)                                                                         # dieselben (μ, s)
    assert max(abs(kopie[k] - tab[k]) for k in tab) < 1e-12                                              # dieselben Gewichte
    assert sum(w for w, _, _ in daten["komponenten"]) == pytest.approx(1.0, abs=1e-12)
    arten = {0.55: 48, 0.25: 576, 0.35: 576, 0.90: 24}
    for s, n in arten.items():
        assert sum(1 for _, _, ss in daten["komponenten"] if ss == s) == n                               # 48 Import, 576 Export, 576 Umschlag, 24 Leer


def test_zahl_der_komponenten_mit_gewicht_2376(aufzaehlung):
    """Von den 4 704 Schritten haben 2 376 ein Gewicht größer 0 (Zollzweig bei Nicht-Import fällt weg); vor dem Zusammenfassen gleicher (μ, s)."""
    knoten, gew = np.polynomial.legendre.leggauss(48)
    n = 0
    for art in range(4):
        for _ in range(12):
            for p_zoll in ((0.75, 0.25) if art == 0 else (1.0, 0.0)):
                for _we in (0, 1):
                    n += (48 if art in (1, 2) else 1) * (p_zoll > 0)
    assert n == 2376


def test_kopierte_tabelle_unveraendert():
    import hashlib
    assert hashlib.sha256(VB.TABELLE.read_bytes()).hexdigest() == "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f"


def test_pegel_gegen_scipy_mischung(aufzaehlung):
    """Wahre Pegel aus F^100 der Mischung: eigene Summe über die Komponenten mit scipy.stats.norm, Wurzelsuche mit scipy.brentq (anderer Weg als wkw_vehikel_b)."""
    _, tab = aufzaehlung
    w = np.array(list(tab.values()))
    mu = np.array([k[0] for k in tab])
    s = np.array([k[1] for k in tab])
    sf = lambda x: float(np.sum(w * stats.norm.sf((math.log(x) - mu) / s)))
    for T, soll in ((100, 155.128), (1000, 287.123)):
        f = lambda lx: 100 * math.log1p(-sf(math.exp(lx))) - math.log1p(-1 / T)
        x = math.exp(optimize.brentq(f, -1.0, 12.0, xtol=1e-13))
        assert x == pytest.approx(soll, abs=5e-4)
        assert VB.pegel_tagesmax(T) == pytest.approx(x, rel=1e-9)
    f = lambda lx: math.log(sf(math.exp(lx))) - math.log(1e-5)
    assert math.exp(optimize.brentq(f, -1.0, 12.0, xtol=1e-13)) == pytest.approx(287.158, abs=5e-4)       # Zielgröße (a): je Container p = 1e-5
    assert VB.pegel_container(1e-5) == pytest.approx(287.158, abs=5e-4)
    assert VB.mittel_container() == pytest.approx(5.646, abs=5e-4)


def test_ziehung_des_tagesmaximums_trifft_f_hoch_100():
    """5 000 Tagesmaxima per Umkehrung: Kolmogorov-Abstand zu F^100 (exakt aus der Mischung) unter der 1-%-Schranke 1,63/sqrt(n)."""
    rng = SplitMix64(4711)
    n = 5000
    xs = sorted(VB.tagesmax_aus_u(rng.random(), exact_log) for _ in range(n))
    d = 0.0
    for i, x in enumerate(xs):
        F = math.exp(VB.log_cdf_tagesmax(x))
        d = max(d, abs(F - i / n), abs(F - (i + 1) / n))
    assert d < 1.63 / math.sqrt(n)


def test_umkehrung_gegen_einzelzuege_von_containern():
    """Zweiter Weg: Tagesmaxima aus je 100 gezogenen Containern (Komponente über Summenverteilung, dann exp(μ + s·normal())) haben dieselbe Verteilung wie die Umkehrung (Zweistichproben-KS, 300 gegen 300)."""
    rng = SplitMix64(99)
    a = sorted(max(VB.zug_container(rng) for _ in range(100)) for _ in range(300))
    rng2 = SplitMix64(100)
    b = sorted(VB.tagesmax_aus_u(rng2.random(), exact_log) for _ in range(300))
    ks = stats.ks_2samp(a, b)
    assert ks.pvalue > 0.001


def test_kopier_test_gegen_verweilzeit_demo_wenn_vorhanden():
    """Konstanten der Definition gegen `verweilzeit-demo` (Nachbarordner; fehlt er, etwa in der CI, entfällt der Test)."""
    nachbar = WURZEL.parent / "verweilzeit-demo"
    if not (nachbar / "vwz_constants.py").exists():
        pytest.skip("verweilzeit-demo nicht gefunden")
    import importlib.util

    def lade(name):
        spec = importlib.util.spec_from_file_location(name, nachbar / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        import sys
        sys.path.insert(0, str(nachbar))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(nachbar))
        return mod
    C = lade("vwz_constants")
    G = lade("vwz_generator")
    assert tuple(C.ART_P) == ART_P and tuple(C.U_SCALE) == U_SCALE and tuple(C.S_BASE) == S_BASE and C.K_KUNDEN == 12 and C.P_ZOLL == 0.25 and C.U_SD == 0.30
    assert np.array_equal(np.array(UK), np.asarray(G.UK))

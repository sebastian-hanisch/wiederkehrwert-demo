"""Eingefrorene Daten: Aufbau, innere Widerspruchsfreiheit, wahre Pegel gleich der Neurechnung, Stichprobe der Neumessung als Band (tools/wkw_make_coverage.py --check)."""

import json
import math
import os
import subprocess
import sys

import pytest

import wkw_reference as REF
import wkw_returnlevel as RL
import wkw_scenario as S
from conftest import WURZEL


@pytest.fixture(scope="module")
def cov():
    return REF.lade_coverage()


def test_aufbau_und_zellenzahl(cov):
    assert len(cov["ueberdeckung"]["zellen"]) == 13 and len(cov["streuband"]["zellen"]) == 45 and len(cov["modellfehler"]["zellen"]) == 6
    assert {z["R"] for z in cov["ueberdeckung"]["zellen"]} == {1000} and {z["R"] for z in cov["streuband"]["zellen"]} == {2000} and {z["R"] for z in cov["modellfehler"]["zellen"]} == {1000}
    assert cov["meta"]["bootstrap_B"] == 199 and cov["meta"]["basis_seed"] == 20261121                        # nicht der Zufalls-Seed der Vormessung (20261120)
    for teil in ("ueberdeckung", "streuband", "modellfehler"):
        k = cov[teil]["kopf"]
        assert k["quelle"] and k["stand"] == "2026-11-20" and k["basis_seed"] == 20261121
    boot = [z for z in cov["ueberdeckung"]["zellen"] if "bootstrap" in z]
    assert len(boot) == 6 and all(z["T"] == 100 and z["xi"] in (0.0, 0.2) and z["bootstrap"]["B"] == 199 for z in boot)
    assert all("bootstrap" in z for z in cov["modellfehler"]["zellen"])
    for t in ("ueberdeckung", "streuband", "modellfehler"):
        ids = [z["id"] for z in cov[t]["zellen"]]
        assert len(ids) == len(set(ids)), t                                                                    # jede Zelle je Teil genau einmal


def test_anteile_summieren_sich_zu_eins(cov):
    for t in ("ueberdeckung", "modellfehler"):
        for z in cov[t]["zellen"]:
            for art in ("delta", "profil", "bootstrap"):
                if art in z:
                    e = z[art]
                    assert e["cov"] + e["hoch"] + e["niedrig"] + e["ohne"] == pytest.approx(1.0, abs=1e-9), (z["id"], art)
                    assert e["se"] == pytest.approx(math.sqrt(e["cov"] * (1 - e["cov"]) / z["R"]), rel=0.02, abs=1e-3), (z["id"], art)


def test_wahre_pegel_gleich_der_neurechnung(cov):
    for z in cov["ueberdeckung"]["zellen"]:
        assert z["x_wahr"] == pytest.approx(RL.gev_pegel(1.0 / z["T"], 0.0, 1.0, z["xi"]), rel=1e-12)
    for z in cov["streuband"]["zellen"]:
        assert z["x_wahr"] == pytest.approx(RL.gev_pegel(1.0 / z["T"], 100.0, 20.0, z["xi"]), rel=1e-12)
    for z in cov["modellfehler"]["zellen"]:
        assert z["x_wahr"] == pytest.approx(S.wahrer_pegel(S.Fall(z["fall"]), z["T"]), rel=1e-9)


def test_streuband_quantile_geordnet_und_anteile_in_0_1(cov):
    for z in cov["streuband"]["zellen"]:
        assert 0 < z["q05"] <= z["q50"] <= z["q95"] and all(0.0 <= z[k] <= 1.0 for k in ("unter_1", "ueber_2", "unter_08"))
        assert z["mu"] == 100.0 and z["sigma"] == 20.0


def test_sigma_bekannt_robuste_streuung_gleich_exakt(cov):
    block = cov["streuband"]["sigma_bekannt_robuste_streuung"]
    lo, hi = RL.gumbel_fehler_band(50, 0.25, 0.75)                                                             # Viertelspanne des Fehlers in Einheiten σ
    exakt = (hi - lo) / 1.349
    for T, wert in block["werte"].items():
        assert wert == pytest.approx(exakt, rel=0.08)                                                          # Simulation (2 000 Läufe) gegen Gamma-Rechnung
    assert max(block["werte"].values()) - min(block["werte"].values()) < 1e-9                                  # für jedes T derselbe Fehler (gleiche Läufe)


def test_referenz_json_gleich_neurechnung():
    sys.path.insert(0, str(WURZEL / "tools"))
    import wkw_make_reference as MR
    alt = json.loads((WURZEL / "wkw_reference.json").read_text(encoding="utf-8"))
    assert MR.vergleiche(MR.erzeuge(), alt) == []


def test_check_der_eingefrorenen_messung_als_band():
    """`tools/wkw_make_coverage.py --check`: 3 Stichproben mit R = 100 bzw. 200 neu gerechnet (dieselben Ströme wie die ersten Läufe der Zellen), als Band verglichen."""
    env = dict(os.environ, WKW_NPROC="2", OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8", str(WURZEL / "tools" / "wkw_make_coverage.py"), "--check"], capture_output=True, text=True, timeout=600, env=env, cwd=str(WURZEL))
    assert r.returncode == 0, r.stdout + r.stderr

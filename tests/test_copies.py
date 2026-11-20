"""Kopier-Tests: die kopierten Dateien sind unverändert (Vorlage SplitMix64, Gewichtstabelle Vehikel B) und der Zufallsgenerator liefert feste Zahlen."""

import hashlib
import math
from pathlib import Path

import pytest

from conftest import WURZEL
from wkw_rng import SplitMix64, exact_log

VORLAGE = WURZEL.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def test_rng_ist_eine_unveraenderte_kopie_der_vorlage():
    if not VORLAGE.exists():
        pytest.skip("Vorlage (Website-Repo) nicht gefunden")
    assert (WURZEL / "wkw_rng.py").read_bytes().replace(b"\r\n", b"\n") == VORLAGE.read_bytes().replace(b"\r\n", b"\n")


def test_splitmix64_feste_zahlen_auf_jeder_plattform():
    """Bekannte Ausgaben des Standardverfahrens (Zustand 0): 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F; `random()` nimmt die oberen 53 Bit."""
    g = SplitMix64(0)
    assert [g.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    h = SplitMix64(0)
    assert h.random() == (0xE220A8397B1DCDAF >> 11) / float(1 << 53)


def test_exact_log_ist_auf_dem_naechsten_double():
    for x in (0.37, 1e-9, 0.999999, 2.5, 123456.789):
        assert exact_log(x) == pytest.approx(math.log(x), rel=1e-15, abs=1e-300)


def test_tabelle_vehikel_b_ist_byteweise_die_kopie():
    assert hashlib.sha256((WURZEL / "wkw_vehikel_b_tabelle.json").read_bytes()).hexdigest() == "669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f"
    quelle = WURZEL.parent / "extremwert-planung" / "vormessung" / "ev_v13_vehikel_b_tabelle.json"
    if quelle.exists():
        assert quelle.read_bytes() == (WURZEL / "wkw_vehikel_b_tabelle.json").read_bytes()

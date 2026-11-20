"""Eingefrorene Daten laden: `wkw_coverage.json` (Messungen, Zähleinheit Läufe) und `wkw_reference.json` (exakte Tabellen und Quellen mit Stufe).

Erzeugung: `tools/wkw_make_coverage.py` und `tools/wkw_make_reference.py` (beide mit `--check`). Die App rechnet diese Zellen nicht neu."""

import json
from functools import lru_cache
from pathlib import Path

WURZEL = Path(__file__).resolve().parent
COVERAGE = WURZEL / "wkw_coverage.json"
REFERENCE = WURZEL / "wkw_reference.json"


@lru_cache(maxsize=1)
def lade_coverage() -> dict:
    return json.loads(COVERAGE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def lade_reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def finde_ueberdeckung(fall: str, xi: float, m: int, T: int):
    """Eingefrorene Überdeckungszelle (GEV: ξ, m, T; sonst die Modellfehler-Zelle mit m = 50) oder None."""
    c = lade_coverage()
    if fall == "gev":
        for z in c["ueberdeckung"]["zellen"]:
            if abs(z["xi"] - xi) < 1e-9 and z["m"] == m and z["T"] == T:
                return z
        return None
    for z in c["modellfehler"]["zellen"]:
        if z["fall"] == fall and z["m"] == m and z["T"] == T:
            return z
    return None


def finde_streuband(xi: float, m: int, T: int):
    for z in lade_coverage()["streuband"]["zellen"]:
        if abs(z["xi"] - xi) < 1e-9 and z["m"] == m and z["T"] == T:
            return z
    return None


def eingefrorene_ueberdeckung_zellen():
    c = lade_coverage()
    return [(z["fall"], z.get("xi"), z["m"], z["T"]) for z in c["ueberdeckung"]["zellen"]] + [(z["fall"], None, z["m"], z["T"]) for z in c["modellfehler"]["zellen"]]

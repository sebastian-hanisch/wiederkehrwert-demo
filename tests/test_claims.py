"""Claims: jede Zahl der README ist belegt.

Drei Wege, auf denen eine README-Zahl belegt sein kann:
1. Sie steht in einer der Tabellen von `tests/readme_tabellen.py`; diese Tabellen erzeugt der Code aus der Rechnung und aus `wkw_coverage.json`,
   und der Test unten verlangt, dass sie WÖRTLICH in der README stehen.
2. Sie ist ein Wert aus `wkw_coverage.json` oder `wkw_reference.json` (in einer der Schreibweisen der README: Dezimalzahl oder Prozent mit 0 bis 4 Stellen).
3. Sie steht in `BELEGT` (unten) mit dem Test, der genau diese Zahl prüft; `BELEGT` ist zugleich die Tabelle „Zahl → Test“ der README.
Jede andere Zahl in der README (außer den Strukturzahlen in `STRUKTUR`) lässt `test_jede_zahl_der_readme_ist_belegt` scheitern.

Die Testzahl der README prüft zusätzlich `tools/readme_test_counts.py`."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import readme_absaetze as AB
import readme_tabellen as TAB
from conftest import WURZEL

README = (WURZEL / "README.md").read_text(encoding="utf-8")
TESTS = {p.name: p.read_text(encoding="utf-8") for p in (WURZEL / "tests").glob("test_*.py")}

# Zahl (wie in der README geschrieben) -> (Test, der sie prüft)
BELEGT = {
    "63,4": "test_handbeispiele.py::test_h1_wahrscheinlichkeit_in_T_jahren_exakt_als_bruch",
    "63,2": "test_handbeispiele.py::test_h1_grenzwert_und_mittel_der_wartezeit",
    "69": "test_handbeispiele.py::test_h1_median_der_wartezeit_69_und_30_jahre",
    "39,5": "test_handbeispiele.py::test_h2_reihe_von_50_jahren_und_jahre_fuer_sicherheit",
    "4,9": "test_handbeispiele.py::test_h2_reihe_von_50_jahren_und_jahre_fuer_sicherheit",
    "975,3": "test_handbeispiele.py::test_h2_sicherheit_ueber_nutzungsdauer_deich_beispiel",
    "1 950,1": "test_handbeispiele.py::test_h2_sicherheit_ueber_nutzungsdauer_deich_beispiel",
    "475,06": "test_claims.py::test_ekm_475_jahre_beispiel",
    "1,96": "test_oracle_wiederkehrwert.py::test_f_rand_der_daten_exakt_1_durch_m_plus_1_fuer_stetige_verteilungen",
    "10,500": "test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel",
    "12,807": "test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel",
    "4,033": "test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel",
    "4,545": "test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel",
    "155,128": "test_vehikel_b.py::test_pegel_gegen_scipy_mischung",
    "287,123": "test_vehikel_b.py::test_pegel_gegen_scipy_mischung",
    "287,158": "test_vehikel_b.py::test_pegel_gegen_scipy_mischung",
    "1 224": "test_vehikel_b.py::test_tabelle_gleich_eigener_aufzaehlung",
    "4 704": "test_vehikel_b.py::test_tabelle_gleich_eigener_aufzaehlung",
    "2 376": "test_vehikel_b.py::test_zahl_der_komponenten_mit_gewicht_2376",
    "0,29": "test_handbeispiele.py::test_h4_ml_und_pegel",
    "1,2379": "test_handbeispiele.py::test_h4_ml_und_pegel",
    "5,8380": "test_handbeispiele.py::test_h4_ml_und_pegel",
    "0,3614": "test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu",
    "2,1144": "test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu",
    "0,2123": "test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu",
    "2,0030": "test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu",
    "0,1130": "test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu",
    "1,9549": "test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu",
    "3,8415": "test_oracle_wiederkehrwert.py::test_e_profil_endpunkte_haben_devianz_3_8415",
    "475": "test_claims.py::test_ekm_475_jahre_beispiel",
    "63,21": "test_handbeispiele.py::test_h1_grenzwert_und_mittel_der_wartezeit",
    "43": "test_claims.py::test_streuband_aussagen",
    "11": "test_claims.py::test_streuband_aussagen",
    "2,04": "test_claims.py::test_streuband_aussagen",
    "1,45": "test_claims.py::test_streuband_aussagen",
    "73": "test_claims.py::test_pegel_liegt_erst_ab_etwa_1_4_m_ueber_dem_reihenmaximum",
    "1,4": "test_claims.py::test_pegel_liegt_erst_ab_etwa_1_4_m_ueber_dem_reihenmaximum",
    "4,6001": "test_handbeispiele.py::test_h4_ml_und_pegel",
}

# Strukturzahlen: Einstellungen, Zählungen von Dateien und Zellen, Namen von Zellen, Jahreszahlen, Versionen. Sie sind keine Messungen.
STRUKTUR = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "13", "15", "20", "45", "50", "95", "100", "199", "200", "365", "500", "1 000", "2 000", "1000", "2000", "4 000", "10", "90", "99", "150",
    "2026", "11", "256", "10085", "14", "25", "60", "30", "40", "300", "20", "12", "16", "70", "400", "5", "999999", "000",
}

TOKEN = re.compile(r"(?<![\w.,/\-])(\d{1,3}(?: \d{3})+|\d+)(?:,(\d+))?(?![\w])")


def _bereinigt(text: str) -> str:
    """README ohne Codeteile, Links, Quellenkennungen und Kommentare: dort stehen Namen und Versionen, keine Messzahlen."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\n## Tests\n.*?(?=\n## )", "\n", text, flags=re.S)       # Testzahlen und Laufzeit prüft tools/readme_test_counts.py
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "] ", text)
    text = re.sub(r"arXiv \d+\.\d+", " ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
    text = re.sub(r"\b(Gl\.|Bsp\.|Tabelle|Python|numpy|Windows|Abschnitt)\s*[\d.,]+", " ", text)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)          # Jahreszahlen der Quellen
    return text


def _alle_zahlen(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _alle_zahlen(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _alle_zahlen(v, out)


def _schreibweisen(x: float):
    """Wie eine Zahl in der README stehen kann: Dezimalzahl mit 0 bis 4 Stellen, Prozent mit 0 bis 4 Stellen (mit Komma, Tausenderleerzeichen)."""
    out = set()
    for faktor in (1.0, 100.0):
        for nd in range(5):
            s = f"{abs(x) * faktor:.{nd}f}"
            ganz, _, rest = s.partition(".")
            ganz = f"{int(ganz):,}".replace(",", " ")
            out.add(ganz + ("," + rest if rest else ""))
    return out


def _belegte_tokens() -> set[str]:
    zahlen = set()
    for datei in ("wkw_coverage.json", "wkw_reference.json"):
        _alle_zahlen(json.loads((WURZEL / datei).read_text(encoding="utf-8")), zahlen)
    belegt = set()
    for x in zahlen:
        belegt |= _schreibweisen(x)
    for tab in list(TAB.alle().values()) + list(AB.alle().values()):
        belegt |= {_norm(m) for m in _tokens(tab)}
    return belegt | set(BELEGT) | STRUKTUR


def _norm(t: str) -> str:
    return t


def _tokens(text: str):
    for m in TOKEN.finditer(text):
        yield m.group(1) + ("," + m.group(2) if m.group(2) else "")


def test_tabellen_stehen_woertlich_in_der_readme():
    for name, tab in {**TAB.alle(), **AB.alle()}.items():
        assert tab in README, f"Tabelle {name} fehlt oder weicht ab (python tests/readme_tabellen.py)"


def test_zahl_zu_test_tabelle_nennt_vorhandene_tests():
    """Jede Zeile von BELEGT verweist auf eine Testdatei und eine Testfunktion, die es gibt."""
    for zahl, ref in BELEGT.items():
        datei, _, name = ref.partition("::")
        assert datei in TESTS, (zahl, ref)
        assert re.search(rf"def {re.escape(name)}\b", TESTS[datei]), (zahl, ref)


def test_zahl_zu_test_tabelle_steht_in_der_readme():
    assert "Zahl → Test" in README
    for zahl, ref in BELEGT.items():
        datei, _, name = ref.partition("::")
        assert f"| {zahl} | `{datei}::{name}` |" in README, zahl


def test_jede_zahl_der_readme_ist_belegt():
    belegt = _belegte_tokens()
    text = _bereinigt(README)
    unbelegt = sorted({t for t in _tokens(text) if t not in belegt})
    assert not unbelegt, "Zahlen der README ohne Beleg (Tabelle, Messdatei oder BELEGT): " + ", ".join(unbelegt)


def test_ekm_475_jahre_beispiel():
    """EKM Bsp. 6.2.19: P(L <= 50) = 0,1 gilt für T = 475,06 (T aus 1 - (1 - 1/T)^50 = 0,1)."""
    import wkw_returnlevel as RL
    T = 1.0 / (-((0.9) ** (1 / 50) - 1))
    assert T == pytest.approx(475.06, abs=0.01)
    assert RL.p_mindestens_einmal(T, 50) == pytest.approx(0.1, abs=1e-9)


def test_streuband_aussagen():
    """Zählungen über alle 45 Zellen des Streubands (Zähleinheit Läufe, R = 2 000): Unterschätzung, Median unter 1, Rechtsschiefe, Band bei gleichem T/m."""
    cov = json.loads((WURZEL / "wkw_coverage.json").read_text(encoding="utf-8"))["streuband"]["zellen"]
    assert len(cov) == 45
    assert min(c["unter_1"] for c in cov) == pytest.approx(0.478, abs=5e-4) and max(c["unter_1"] for c in cov) == pytest.approx(0.6175, abs=1e-3)
    assert sum(c["q50"] < 1 for c in cov) == 43 and {c["id"] for c in cov if c["q50"] >= 1} == {"xi+0.2_m100_T100", "xi+0.2_m100_T1000"}
    schief = {xi: sum(c["q95"] - 1 > 1 - c["q05"] for c in cov if c["xi"] == xi) for xi in (-0.2, 0.0, 0.2)}
    assert schief == {-0.2: 3, 0.0: 11, 0.2: 15}
    z = {c["id"]: c for c in cov}
    assert z["xi+0.2_m50_T100"]["q95"] / z["xi+0.2_m50_T100"]["q05"] == pytest.approx(2.04, abs=0.005)       # T/m = 2
    assert z["xi+0.2_m500_T1000"]["q95"] / z["xi+0.2_m500_T1000"]["q05"] == pytest.approx(1.45, abs=0.005)    # T/m = 2, größeres m: schmaler


def test_pegel_liegt_erst_ab_etwa_1_4_m_ueber_dem_reihenmaximum():
    """Der T-Jahres-Wert liegt in mehr als der Hälfte der Reihen von m Jahren über dem größten Wert der Reihe, wenn (1 - 1/T)^m > 1/2 (stetige Verteilung, unabhängige Jahre).

    Unabhängiger Rechenweg: lineare Suche mit exakten Brüchen statt der Ganzzahl-Ungleichung der Funktion; m = 50 gibt T = 73, lange Reihen etwa das 1,4-Fache von m
    (Grenzwert 1/ln 2 = 1,44)."""
    import math
    from fractions import Fraction

    import wkw_returnlevel as RL

    def suche(m):
        T = 2
        while Fraction(T - 1, T) ** m <= Fraction(1, 2):
            T += 1
        return T

    for m in list(range(1, 31)) + [50, 100]:
        assert RL.T_pegel_ueber_maximum(m) == suche(m), m
    assert RL.T_pegel_ueber_maximum(50) == 73
    assert Fraction(72, 73) ** 50 > Fraction(1, 2) >= Fraction(71, 72) ** 50          # 73 ist das kleinste T, bei dem die Reihe den Pegel nicht erreicht
    for m in (200, 500, 5000):
        assert 1.4 <= RL.T_pegel_ueber_maximum(m) / m <= 1.5
    assert RL.T_pegel_ueber_maximum(5000) / 5000 == pytest.approx(1 / math.log(2), abs=2e-3)
    assert RL.T_pegel_ueber_maximum(1) == 3


def test_preset_fazits_gegen_die_messung():
    """Die Zahlen in den Hilfetexten der Presets stammen aus den eingefrorenen Zellen (Kurze Reihe: Delta-Überdeckung und Breitenverhältnis; Vehikel B: Median von x̂/x_T)."""
    import wkw_constants as C
    delta, breite, median = AB.preset_fazit_kurz()
    assert f"nur {delta} % der Läufe" in C.PRESET_HELP["Kurze Reihe m = 20"] and f"{breite}-mal so breit" in C.PRESET_HELP["Kurze Reihe m = 20"]
    assert f"beim {median}-Fachen" in C.PRESET_HELP["Vehikel B, 1000er-Wert"]

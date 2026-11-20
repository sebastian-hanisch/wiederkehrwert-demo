"""Erzeugt `wkw_reference.json`: exakte Tabellen (Wartezeit, Reihenlänge, Gumbel mit bekanntem σ) und die Quellen mit Stufe. Alles ohne Zufall.

Aufruf (aus dem Demo-Ordner): python tools/wkw_make_reference.py          schreibt die Datei
                              python tools/wkw_make_reference.py --check   rechnet neu und vergleicht (exakte Zahlen gleich, Gleitkomma auf 1e-12 relativ)
Stufen der Quellen: W = Wortlaut gelesen, A = nur Abstract, S = Suchtreffer / Zusammenfassung, R = eigene Rechnung."""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))
sys.dont_write_bytecode = True

import wkw_returnlevel as RL  # noqa: E402

DATEI = WURZEL / "wkw_reference.json"
STAND = "2026-11-20"


def erzeuge() -> dict:
    exakt = {
        "p_100_100": RL.p_mindestens_einmal(100, 100),
        "p_in_T_jahren": {str(T): RL.p_mindestens_einmal(T, T) for T in (2, 10, 50, 100, 1000)},
        "grenzwert_1_minus_1_durch_e": 1 - 2.718281828459045 ** -1,
        "median_wartezeit": {str(T): RL.median_wartezeit(T) for T in (10, 100, 1000)},
        "nutzungsdauer_T100": {str(N): RL.p_mindestens_einmal(100, N) for N in (30, 50, 100)},
        "reihe_erreicht": {str(m): {str(T): RL.p_mindestens_einmal(T, m) for T in (10, 100, 1000)} for m in (20, 50, 100)},
        "jahre_fuer_q": {str(T): {str(q): RL.jahre_fuer_sicherheit(T, Fraction(q).limit_denominator(1000)) for q in (0.5, 0.9, 0.95, 0.99)} for T in (10, 100, 1000)},
        "T_fuer_sicherheit_095": {"50": RL.T_fuer_sicherheit(0.95, 50), "100": RL.T_fuer_sicherheit(0.95, 100)},
        "rand_der_daten": {"50": RL.p_rand_der_daten(50)},
        "gumbel_sigma_bekannt": {str(m): dict(zip(("delta", "profil"), RL.gumbel_ueberdeckung_exakt(m))) for m in (5, 10, 20, 50, 500)},
        "gumbel_fehlerband_sigma": {str(m): list(RL.gumbel_fehler_band(m)) for m in (5, 10, 20, 50, 500)},
    }
    quellen = [
        {"kurz": "Coles 2001", "stufe": "W", "inhalt": "Pegel-Formel (Gl. 3.4), Delta-Methode (Gl. 3.11), Profil-Likelihood (Gl. 3.12): Gleichungen und Befunde nur in eigenen Worten"},
        {"kurz": "Embrechts/Klüppelberg/Mikosch 1997, 6.2.4, Bsp. 6.2.19, 8.1.5", "stufe": "W", "inhalt": "Wiederkehrperiode als Mittel der Wartezeit, 475-Jahres-Beispiel, Deich-Beispiel (Zahlen des Buchs nicht übernommen)"},
        {"kurz": "Huang/Nychka/Zhang, arXiv 1802.09387, Tabelle 1", "stufe": "W", "inhalt": "GPD an Überschreitungen, nominell 90 %, anderes Verfahren und Niveau, hier nicht nachgerechnet: Delta 0,63 / 0,74 / 0,77 und Profil 0,80 / 0,84 / 0,82 für den 25-, 50-, 100-Jahres-Wert"},
        {"kurz": "Huang u. a., arXiv 1512.08775", "stufe": "A", "inhalt": "Warnzeichen: Reihe so lang wie die Wiederkehrperiode oder kürzer"},
        {"kurz": "genEVA (Buriticá u. a.), arXiv 2401.14910", "stufe": "S", "inhalt": "nominell 50-%-Bootstrap-Intervalle deckten etwa 30 % (Quantilregression, anderes Problem); nach Zusammenfassung, ungeprüft"},
        {"kurz": "Rootzén/Katz", "stufe": "W", "inhalt": "Wiederkehrperiode gegen Risiko über die Nutzungsdauer"},
        {"kurz": "Smith 1985; Hosking/Wallis; Fisher/Tippett", "stufe": "nicht gelesen", "inhalt": "in dieser Demo nicht zitiert (Smith 1985 nur als Suchtreffer für die Bedingung ξ ≥ -1)"},
    ]
    return {"meta": {"beschreibung": "Exakte Tabellen (ohne Zufall) und Quellen mit Stufe", "quelle": "tools/wkw_make_reference.py", "stand": STAND}, "exakt": exakt, "quellen": quellen}


def vergleiche(a, b, pfad="", tol=1e-12):
    fehler = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            fehler.append(f"{pfad}: Schlüssel verschieden")
        for k in a:
            if k in b:
                fehler += vergleiche(a[k], b[k], f"{pfad}/{k}", tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            fehler.append(f"{pfad}: Länge verschieden")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                fehler += vergleiche(x, y, f"{pfad}[{i}]", tol)
    elif isinstance(a, float) or isinstance(b, float):
        if abs(a - b) > tol * max(1.0, abs(b)):
            fehler.append(f"{pfad}: {a} gegen {b}")
    elif a != b:
        fehler.append(f"{pfad}: {a!r} gegen {b!r}")
    return fehler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    neu = erzeuge()
    if a.check:
        alt = json.loads(DATEI.read_text(encoding="utf-8"))
        fehler = vergleiche(neu, alt)
        for f in fehler:
            print("FEHLER:", f)
        print("wkw_reference.json: gleich der Neurechnung" if not fehler else f"{len(fehler)} Abweichungen")
        sys.exit(1 if fehler else 0)
    DATEI.write_text(json.dumps(neu, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben:", DATEI)


if __name__ == "__main__":
    main()

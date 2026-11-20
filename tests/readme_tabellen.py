"""Die Zahlentabellen der README, aus den Rechenwegen und den eingefrorenen Daten erzeugt.

`tests/test_claims.py` prüft, dass jede dieser Tabellen wörtlich in der README steht: so kann keine README-Zahl von der Rechnung oder der Messung abweichen.
Ausgabe zum Einfügen in die README: `python tests/readme_tabellen.py`."""

import json
import sys
from fractions import Fraction
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))

import wkw_constants as C  # noqa: E402
import wkw_evaluation as EV  # noqa: E402
import wkw_returnlevel as RL  # noqa: E402
from wkw_evaluation import de, pct, tausend  # noqa: E402


def _cov():
    return json.loads((WURZEL / "wkw_coverage.json").read_text(encoding="utf-8"))


def md(kopf, zeilen):
    out = ["| " + " | ".join(kopf) + " |", "|" + "|".join("---" for _ in kopf) + "|"]
    out += ["| " + " | ".join(z) + " |" for z in zeilen]
    return "\n".join(out)


def tabelle_wartezeit():
    return md(["T", "P(mindestens eine Überschreitung in T Jahren)", "Median der Wartezeit (Jahre)"],
              [[str(T), pct(RL.p_mindestens_einmal(T, T), 2), str(RL.median_wartezeit(T))] for T in (2, 10, 50, 100, 1000)])


def tabelle_reihe_erreicht():
    zeilen = [[str(m)] + [pct(RL.p_mindestens_einmal(T, m), 2) for T in (10, 100, 1000)] for m in (20, 50, 100)]
    return md(["Reihenlänge m (Jahre)", "T = 10", "T = 100", "T = 1000"], zeilen)


def tabelle_jahre_fuer_q():
    return md(["Sicherheit q", "T = 10", "T = 100", "T = 1000"], [[f"{int(q * 100)} %"] + [tausend(j) for j in jahre] for q, jahre in EV.jahre_tabelle()])


def tabelle_sigma_bekannt():
    zeilen = []
    for m in (5, 10, 20, 50, 500):
        d, p = RL.gumbel_ueberdeckung_exakt(m)
        lo, hi = RL.gumbel_fehler_band(m)
        zeilen.append([str(m), pct(d, 3), pct(p, 3), f"{de(lo, 3)} bis {de(hi, 3)}"])
    return md(["m (Reihen)", "Delta deckt (exakt)", "Profil deckt (exakt)", "Fehler μ̂ − μ, 5 bis 95 % (Einheit σ)"], zeilen)


def tabelle_streuband(xi=0.2):
    cov = _cov()["streuband"]["zellen"]
    zeilen = []
    for m in (20, 50, 100, 200, 500):
        zelle = [str(m)]
        for T in (10, 100, 1000):
            z = next(c for c in cov if abs(c["xi"] - xi) < 1e-9 and c["m"] == m and c["T"] == T)
            zelle.append(f"{de(z['q05'], 2)} bis {de(z['q95'], 2)} (Median {de(z['q50'], 2)})")
        zeilen.append(zelle)
    return md(["m (Blöcke)", "T = 10", "T = 100", "T = 1000"], zeilen)


def tabelle_streuband_xi():
    cov = _cov()["streuband"]["zellen"]
    zeilen = []
    for xi in (-0.2, 0.0):
        z = next(c for c in cov if abs(c["xi"] - xi) < 1e-9 and c["m"] == 50 and c["T"] == 100)
        z3 = next(c for c in cov if abs(c["xi"] - xi) < 1e-9 and c["m"] == 50 and c["T"] == 1000)
        zeilen.append([de(xi, 1), f"{de(z['q05'], 2)} bis {de(z['q95'], 2)}", f"{de(z3['q05'], 2)} bis {de(z3['q95'], 2)}"])
    return md(["ξ (m = 50)", "T = 100", "T = 1000"], zeilen)


def tabelle_ueberdeckung():
    zellen = _cov()["ueberdeckung"]["zellen"]
    zeilen = []
    for z in zellen:
        b = z.get("bootstrap")
        zeilen.append([de(z["xi"], 1), str(z["m"]), str(z["T"]), pct(z["delta"]["cov"]), pct(z["profil"]["cov"]), "-" if b is None else pct(b["cov"]),
                       f"{pct(z['delta']['niedrig'])} / {pct(z['delta']['hoch'])}"])
    return md(["ξ", "m", "T", "Delta", "Profil", "Perzentil-Bootstrap (B = 199)", "Delta: zu niedrig / zu hoch"], zeilen)


def tabelle_modellfehler():
    zellen = _cov()["modellfehler"]["zellen"]
    zeilen = []
    for z in zellen:
        zeilen.append([EV.FALL_NAMEN[z["fall"]], str(z["T"]), de(z["x_wahr"], 3), pct(z["delta"]["cov"]), pct(z["profil"]["cov"]), pct(z["bootstrap"]["cov"])])
    return md(["Fall (m = 50)", "T", "wahrer Pegel", "Delta", "Profil", "Perzentil-Bootstrap (B = 199)"], zeilen)


def tabelle_wahre_ueberschreitung():
    zellen = _cov()["modellfehler"]["zellen"]
    zeilen = []
    for z in zellen:
        w = z["wahre_ueberschreitung"]
        zeilen.append([EV.FALL_NAMEN[z["fall"]], str(z["T"]), de(w["p_wahr_mal_T_median"], 2), pct(w["anteil_ueber_2"]), pct(w["anteil_unter_halb"]), de(w["v_median"], 2), pct(w["unter_1"])])
    return md(["Fall (m = 50)", "T", "Median p_wahr · T", "p_wahr ≥ 2/T", "p_wahr ≤ 0,5/T", "Median x̂/x_T", "Anteil x̂ < x_T"], zeilen)


def alle():
    return {name[len("tabelle_"):]: f() for name, f in sorted(globals().items()) if name.startswith("tabelle_") and callable(f)}


if __name__ == "__main__":
    for name, text in alle().items():
        print(f"### {name}\n{text}\n")

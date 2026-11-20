"""Erzeugt `wkw_coverage.json` (eingefrorene Messungen: Überdeckung der Intervalle, Streuband des Pegels, Modellfehler) und prüft sie.

Aufruf (aus dem Demo-Ordner, mit dem Python des Test-venv):
  python tools/wkw_make_coverage.py --check                   rechnet 3 Zellen mit R = 100 neu und vergleicht sie als Band mit der eingefrorenen Datei (einige Zehn Sekunden)
  python tools/wkw_make_coverage.py --lang --teil ueberdeckung --out TEIL.json   rechnet einen Teil in voller Länge (ueberdeckung | streuband | modellfehler)
  python tools/wkw_make_coverage.py --zusammenfuegen T1.json T2.json T3.json     setzt die Teile zu wkw_coverage.json zusammen
Umgebungsvariable WKW_NPROC = Zahl der Prozesse (Standard 6). Zufall: SplitMix64 (wkw_rng.py), je Lauf ein eigener Strom aus (Basis, Zelle, Lauf); numpy.random wird nicht benutzt.

Zähleinheit überall: Läufe (unabhängige Reihen mit m Blöcken). Ort und Skala der GEV: Überdeckung und Modellfehler GEV(0, 1, ξ); Streuband GEV(100, 20, ξ)."""

import argparse
import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))
sys.dont_write_bytecode = True

import wkw_returnlevel as RL  # noqa: E402
import wkw_study as ST  # noqa: E402
from wkw_scenario import Fall, wahrer_pegel  # noqa: E402

DATEI = WURZEL / "wkw_coverage.json"
BASIS_SEED = 20261121                      # nicht der Zufalls-Seed der Vormessung (20261120)
STAND = "2026-11-20"
QUELLE = "tools/wkw_make_coverage.py (SplitMix64, wkw_rng.py)"
R_UEBERDECKUNG = 1000
R_STREUBAND = 2000
B_BOOT = 199
ZELLEN_UEBERDECKUNG = [(xi, m, 100) for xi in (-0.3, 0.0, 0.2) for m in (20, 50, 100)] + [(0.2, 50, 10), (0.2, 50, 1000), (0.0, 50, 1000), (0.2, 20, 1000)]
BOOT_ZELLEN = {(xi, m) for xi in (0.0, 0.2) for m in (20, 50, 100)}      # Bootstrap nur bei T = 100
STREU_XI, STREU_M, STREU_T = (-0.2, 0.0, 0.2), (20, 50, 100, 200, 500), (10, 100, 1000)
MODELL_FAELLE = ("exp365", "normal365", "vwz")
MODELL_M = 50


def nproc() -> int:
    return int(os.environ.get("WKW_NPROC", "6"))


def chunks(n, k):
    return [list(range(i, min(i + k, n))) for i in range(0, n, k)]


# --- Aufgaben für den Prozesspool (müssen auf oberster Ebene stehen) ---------------------------------------------------------------------------

def aufgabe_ueberdeckung(args):
    art, xi, m, Ts, tag, runs, mit_boot = args
    fall = Fall(art, xi, 0.0, 1.0)
    return [ST.lauf(fall, m, Ts, (BASIS_SEED, tag, r), profil=True, boot_B=B_BOOT if mit_boot else 0) for r in runs]


def aufgabe_streuband(args):
    xi, m, Ts, tag, runs, bekannt, gumbel = args
    fall = Fall("gev", xi, 100.0, 20.0)
    return [ST.lauf_streuband(fall, m, Ts, (BASIS_SEED, tag, r), bekannt=bekannt, gumbel_zusatz=gumbel) for r in runs]


def _kopf(teil):
    return {"quelle": QUELLE, "stand": STAND, "python": platform.python_version(), "numpy": np.__version__, "basis_seed": BASIS_SEED, "teil": teil}


def _breite(e):
    return None if math.isnan(e["breite_median"]) else e["breite_median"]


def _intervall_feld(e):
    return {"cov": e["cov"], "se": e["se"], "hoch": e["hoch"], "niedrig": e["niedrig"], "ohne": e["ohne"], "offen_oben": e["offen_oben"], "breite_median": _breite(e)}


# --- Teil Überdeckung --------------------------------------------------------------------------------------------------------------------------

def zelle_ueberdeckung(pool, xi, m, T, R, tag_zusatz=0, art="gev"):
    tag = 100000 + int(round((xi + 1) * 100)) * 1000 + m + 7 * int(math.log10(T)) * 10 ** 6 + tag_zusatz
    mit_boot = (xi, m) in BOOT_ZELLEN and T == 100 and art == "gev"
    t0 = time.perf_counter()
    teile = pool.map(aufgabe_ueberdeckung, [(art, xi, m, (T,), tag, ch, mit_boot) for ch in chunks(R, 10)])
    laeufe = [r for t in teile for r in t]
    fall = Fall(art, xi, 0.0, 1.0)
    x_wahr = wahrer_pegel(fall, T)
    z = {"id": f"xi{xi:+.1f}_m{m}_T{T}", "fall": art, "xi": xi, "m": m, "T": T, "R": R, "seed": [BASIS_SEED, tag], "x_wahr": x_wahr,
         "anpassung_am_rand": float(np.mean([not r["ok"] for r in laeufe if r is not None])), "keine_anpassung": sum(r is None for r in laeufe) / R,
         "delta": _intervall_feld(ST.bewerte_intervalle(laeufe, T, "delta", x_wahr)), "profil": _intervall_feld(ST.bewerte_intervalle(laeufe, T, "profil", x_wahr))}
    if mit_boot:
        z["bootstrap"] = dict(_intervall_feld(ST.bewerte_intervalle(laeufe, T, "boot", x_wahr)), B=B_BOOT)
    z["laufzeit_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Überdeckung {z['id']}: Delta {z['delta']['cov']:.3f}, Profil {z['profil']['cov']:.3f}" + (f", Bootstrap {z['bootstrap']['cov']:.3f}" if mit_boot else "") + f" ({z['laufzeit_s']} s)", flush=True)
    return z


def teil_ueberdeckung(R=R_UEBERDECKUNG, nur=None):
    out = {"kopf": _kopf("ueberdeckung"), "zellen": []}
    zellen = ZELLEN_UEBERDECKUNG if nur is None else nur
    with Pool(nproc()) as pool:
        for xi, m, T in zellen:
            out["zellen"].append(zelle_ueberdeckung(pool, xi, m, T, R))
        if nur is None:
            # Nachmessung der schwächsten Profil-Zelle mit frischen Strömen (Zähleinheit Läufe, R = 3 000)
            z = zelle_ueberdeckung(pool, -0.3, 20, 100, 3000, tag_zusatz=5_000_000)
            z["id"] = "nachmessung_xi-0.3_m20_T100"
            out["nachmessung"] = z
    return out


# --- Teil Streuband ----------------------------------------------------------------------------------------------------------------------------

def zelle_streuband(pool, xi, m, R):
    tag = 200000 + int(round((xi + 1) * 100)) * 1000 + m
    gum = xi == 0.0
    t0 = time.perf_counter()
    teile = pool.map(aufgabe_streuband, [(xi, m, STREU_T, tag, ch, False, gum) for ch in chunks(R, 25)])
    laeufe = [r for t in teile for r in t]
    fall = Fall("gev", xi, 100.0, 20.0)
    zellen = []
    for T in STREU_T:
        x_wahr = wahrer_pegel(fall, T)
        e = ST.bewerte_streuband([r for r in laeufe if r is not None], T, x_wahr)
        e.update({"id": f"xi{xi:+.1f}_m{m}_T{T}", "xi": xi, "m": m, "T": T, "seed": [BASIS_SEED, tag], "x_wahr": x_wahr, "mu": 100.0, "sigma": 20.0})
        if gum:
            def rsd(werte):
                q25, q75 = np.quantile(werte, [0.25, 0.75])
                return float((q75 - q25) / 1.349)
            fehler_gev = [(r["T"][T]["z"] - x_wahr) / 20.0 for r in laeufe if r is not None]
            fehler_gum = [(r["T"][T]["z_gumbel"] - x_wahr) / 20.0 for r in laeufe if r is not None]
            e["robuste_streuung_sigma"] = {"gev_frei": rsd(fehler_gev), "gumbel_mu_sigma_frei": rsd(fehler_gum)}
        zellen.append(e)
    print(f"  Streuband xi={xi:+.1f} m={m}: " + ", ".join(f"T={z['T']}: {z['q05']:.3f} bis {z['q95']:.3f}" for z in zellen) + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    return zellen


def zelle_sigma_bekannt(pool, m, R):
    """Gumbel (ξ = 0), σ bekannt: robuste Streuung des Fehlers in σ (aus Läufen; exakt steht sie in wkw_returnlevel)."""
    tag = 290000 + m
    teile = pool.map(aufgabe_streuband, [(0.0, m, STREU_T, tag, ch, True, False) for ch in chunks(R, 50)])
    laeufe = [r for t in teile for r in t if r is not None]
    fall = Fall("gev", 0.0, 100.0, 20.0)
    out = {}
    for T in STREU_T:
        xw = wahrer_pegel(fall, T)
        f = [(r["T"][T]["z"] - xw) / 20.0 for r in laeufe]
        q25, q75 = np.quantile(f, [0.25, 0.75])
        out[str(T)] = float((q75 - q25) / 1.349)
    return out


def teil_streuband(R=R_STREUBAND, nur=None):
    out = {"kopf": _kopf("streuband"), "zellen": []}
    zellen = [(xi, m) for xi in STREU_XI for m in STREU_M] if nur is None else nur
    with Pool(nproc()) as pool:
        for xi, m in zellen:
            out["zellen"].extend(zelle_streuband(pool, xi, m, R))
        if nur is None:
            out["sigma_bekannt_robuste_streuung"] = {"m": 50, "R": R, "werte": zelle_sigma_bekannt(pool, 50, R), "seed": [BASIS_SEED, 290050]}
    return out


# --- Teil Modellfehler -------------------------------------------------------------------------------------------------------------------------

def teil_modellfehler(R=R_UEBERDECKUNG, faelle=MODELL_FAELLE):
    out = {"kopf": _kopf("modellfehler"), "zellen": []}
    with Pool(nproc()) as pool:
        for art in faelle:
            tag = 300000 + MODELL_FAELLE.index(art) * 1000
            t0 = time.perf_counter()
            teile = pool.map(aufgabe_ueberdeckung, [(art, 0.0, MODELL_M, (100, 1000), tag, ch, True) for ch in chunks(R, 10)])
            laeufe = [r for t in teile for r in t]
            fall = Fall(art)
            for T in (100, 1000):
                x_wahr = wahrer_pegel(fall, T)
                z = {"id": f"{art}_m{MODELL_M}_T{T}", "fall": art, "m": MODELL_M, "T": T, "R": R, "seed": [BASIS_SEED, tag], "x_wahr": x_wahr,
                     "delta": _intervall_feld(ST.bewerte_intervalle(laeufe, T, "delta", x_wahr)), "profil": _intervall_feld(ST.bewerte_intervalle(laeufe, T, "profil", x_wahr)),
                     "bootstrap": dict(_intervall_feld(ST.bewerte_intervalle(laeufe, T, "boot", x_wahr)), B=B_BOOT),
                     "wahre_ueberschreitung": ST.bewerte_modellfehler(fall, [r for r in laeufe if r is not None], T)}
                out["zellen"].append(z)
                print(f"  Modellfehler {z['id']}: x_T = {x_wahr:.3f}, Delta {z['delta']['cov']:.3f}, Profil {z['profil']['cov']:.3f}, Bootstrap {z['bootstrap']['cov']:.3f}", flush=True)
            print(f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


# --- Zusammenfügen, Schreiben, Prüfen ----------------------------------------------------------------------------------------------------------

def schreibe(daten, pfad):
    Path(pfad).write_text(json.dumps(daten, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def zusammenfuegen(teile):
    gesamt = {"meta": {"beschreibung": "Eingefrorene Messungen der Demo Wiederkehrwert (Zähleinheit überall Läufe; Zufall SplitMix64). Je Zelle stehen R, Zufalls-Seed, x_wahr und die Standardfehler.",
                       "quelle": QUELLE, "stand": STAND, "basis_seed": BASIS_SEED, "ueberdeckung_R": R_UEBERDECKUNG, "streuband_R": R_STREUBAND, "bootstrap_B": B_BOOT}}
    for p in teile:
        t = json.loads(Path(p).read_text(encoding="utf-8"))
        name = t["kopf"]["teil"]
        gesamt[name] = {k: v for k, v in t.items()}
    return gesamt


def lade():
    return json.loads(DATEI.read_text(encoding="utf-8"))


def pruefe(verbose=True):
    """Rechnet 3 Zellen mit R = 100 (dieselben Ströme wie die ersten 100 Läufe der eingefrorenen Zellen) neu und vergleicht als Band; prüft den Aufbau der Datei."""
    f = lade()
    fehler = []
    for abschnitt, n_erwartet in (("ueberdeckung", 13), ("streuband", 45), ("modellfehler", 6)):
        n = len(f[abschnitt]["zellen"])
        if n != n_erwartet:
            fehler.append(f"{abschnitt}: {n} Zellen statt {n_erwartet}")
    zellen = {z["id"]: z for z in f["ueberdeckung"]["zellen"]}
    ref = zellen["xi+0.2_m50_T100"]
    with Pool(min(nproc(), 4)) as pool:
        neu = zelle_ueberdeckung(pool, 0.2, 50, 100, 100)
        for art in ("delta", "profil"):
            a, b = neu[art]["cov"], ref[art]["cov"]
            band = 4.0 * math.sqrt(max(b * (1 - b), 0.02) / 100)
            if abs(a - b) > band:
                fehler.append(f"Überdeckung {art}: neu {a:.3f}, eingefroren {b:.3f}, Band {band:.3f}")
        sb = {z["id"]: z for z in f["streuband"]["zellen"]}["xi+0.2_m50_T100"]
        z = zelle_streuband(pool, 0.2, 50, 200)
        z = {e["T"]: e for e in z}[100]
        for k in ("q05", "q50", "q95"):
            if abs(z[k] - sb[k]) > 0.12 * sb[k]:
                fehler.append(f"Streuband {k}: neu {z[k]:.3f}, eingefroren {sb[k]:.3f}")
    # exakte Zahlen müssen gleich sein
    ref_ex = json.loads((WURZEL / "wkw_reference.json").read_text(encoding="utf-8"))
    if abs(ref_ex["exakt"]["p_100_100"] - RL.p_mindestens_einmal(100, 100)) > 1e-12:
        fehler.append("wkw_reference.json: p(100 in 100) weicht ab")
    for x in fehler:
        print("FEHLER:", x)
    if verbose and not fehler:
        print("wkw_coverage.json: 3 Stichproben im Band, Aufbau in Ordnung")
    return fehler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lang", action="store_true")
    ap.add_argument("--teil", choices=("ueberdeckung", "streuband", "modellfehler"))
    ap.add_argument("--out")
    ap.add_argument("--zusammenfuegen", nargs="+")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if pruefe() else 0)
    if a.zusammenfuegen:
        schreibe(zusammenfuegen(a.zusammenfuegen), DATEI)
        print("geschrieben:", DATEI)
        return
    if a.lang and a.teil and a.out:
        t0 = time.perf_counter()
        daten = {"ueberdeckung": teil_ueberdeckung, "streuband": teil_streuband, "modellfehler": teil_modellfehler}[a.teil]()
        daten["kopf"]["gesamtlaufzeit_s"] = round(time.perf_counter() - t0)
        schreibe(daten, a.out)
        print(f"Teil {a.teil} fertig in {time.perf_counter() - t0:.0f} s: {a.out}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()

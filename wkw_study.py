"""Ein Lauf = eine Reihe von m Blockmaxima, Anpassung, Pegel, drei Intervalle; und die Kennzahlen über viele Läufe.

Dieselben Funktionen rechnen die Live-Experimente der App und die eingefrorenen Tabellen (`tools/wkw_make_coverage.py`). Zähleinheit: Läufe (unabhängige Reihen mit m Blöcken).
Zufall: je Lauf ein eigener SplitMix64-Strom aus `stream_seed(Basis, Zelle, Lauf)`; der Bootstrap eines Laufs zieht aus demselben Strom weiter."""

import math

import numpy as np

import wkw_gev as G
import wkw_numeric as N
from wkw_rng import SplitMix64
from wkw_scenario import Fall, stream_seed, wahre_ueberschreitung, wahrer_pegel, ziehe_bloecke, zug_gev


def lauf(fall: Fall, m: int, Ts, seed_teile, profil: bool = True, boot_B: int = 0, delta: bool = True, mit_daten: bool = False):
    """Ein Lauf. Rückgabe: None, wenn keine Anpassung möglich ist (m < 3, alle Blöcke gleich), sonst ein Wörterbuch mit den Schätzwerten und je T den Intervallen
    (`delta`: (unten, oben, Standardfehler) oder None, `profil`: (unten, oben), `boot`: (unten, oben))."""
    rng = SplitMix64(stream_seed(*seed_teile))
    x = np.array(ziehe_bloecke(fall, rng, m), float)
    try:
        mu, s, xi, f, ok = G.fit_ml(x)
    except ValueError:
        return None
    res = {"ok": ok, "mu": mu, "sigma": s, "xi": xi, "nll": f, "T": {}}
    if mit_daten:
        res["x"] = [float(v) for v in x]
    fits = None
    if boot_B:
        fits = G.bootstrap_anpassungen(lambda: np.array([zug_gev(rng, mu, s, xi) for _ in range(m)], float), mu, s, xi, boot_B)
    for T in Ts:
        p = 1.0 / T
        eintrag = {"z": G.gev_pegel(p, mu, s, xi), "delta": None, "profil": None, "boot": None}
        if delta or profil:
            d = G.delta_intervall(x, mu, s, xi, p)
            if delta:
                eintrag["delta"] = None if d is None else tuple(float(v) for v in d)
        if profil:
            try:
                eintrag["profil"] = G.profil_intervall(x, mu, s, xi, f, p, se=None if d is None else d[2])
            except (ValueError, OverflowError, ZeroDivisionError):
                eintrag["profil"] = None
        if fits:
            eintrag["boot"] = G.bootstrap_intervall(fits, p)
        res["T"][T] = eintrag
    return res


def bewerte_intervalle(laeufe, T: int, art: str, x_wahr: float):
    """Überdeckung einer Intervallart ('delta', 'profil', 'boot') über Läufe (Zähleinheit Läufe).

    hoch   = der wahre Pegel liegt UNTER dem Intervall (das Intervall liegt zu hoch);
    niedrig = der wahre Pegel liegt ÜBER dem Intervall (das Intervall liegt zu niedrig);
    ohne   = kein Intervall (Delta: Hesse-Matrix nicht positiv definit; zählt als verfehlt); offen_oben = Obergrenze unendlich (zählt als überdeckend, wenn der wahre Pegel über der Untergrenze liegt; sonst als zu hoch)."""
    n = len(laeufe)
    cover = hoch = niedrig = ohne = offen = 0
    breiten = []
    for r in laeufe:
        iv = r["T"][T][art] if r is not None else None
        if iv is None or math.isnan(iv[0]) or math.isnan(iv[1]):
            ohne += 1
            continue
        lo, hi = iv[0], iv[1]
        if lo <= x_wahr <= hi:
            cover += 1
        elif x_wahr < lo:
            hoch += 1
        else:
            niedrig += 1
        if hi == math.inf:
            offen += 1
        if math.isfinite(lo) and math.isfinite(hi):
            breiten.append(hi - lo)
    k = cover
    wlo, whi = N.wilson(k, n)
    return {
        "R": n, "cov": cover / n, "hoch": hoch / n, "niedrig": niedrig / n, "ohne": ohne / n, "offen_oben": offen / n,
        "breite_median": float(np.median(breiten)) if breiten else math.nan,
        "se": math.sqrt(max(cover / n * (1 - cover / n), 1e-12) / n), "wilson": (wlo, whi),
    }


def bewerte_streuband(laeufe, T: int, x_wahr: float):
    """Streuband von V = Pegel-Dach / wahrer Pegel über Läufe: q05, Median, q95, Anteil V < 1, V > 2, V < 0,8."""
    v = np.array([r["T"][T]["z"] / x_wahr for r in laeufe if r is not None], float)
    v = v[np.isfinite(v)]
    q05, q50, q95 = (float(q) for q in np.quantile(v, [0.05, 0.5, 0.95]))
    return {"R": int(v.size), "q05": q05, "q50": q50, "q95": q95, "unter_1": float(np.mean(v < 1)), "ueber_2": float(np.mean(v > 2)), "unter_08": float(np.mean(v < 0.8))}


def bewerte_modellfehler(fall: Fall, laeufe, T: int):
    """Wahre Überschreitungswahrscheinlichkeit des geschätzten Pegels in Vielfachen des Ziels 1/T (aus F^n), Median von x̂/x_T, Anteil x̂ < x_T."""
    x_wahr = wahrer_pegel(fall, T)
    zs = [r["T"][T]["z"] for r in laeufe if r is not None]
    pe = np.array([wahre_ueberschreitung(fall, z) for z in zs], float) * T
    v = np.array(zs, float) / x_wahr
    return {"R": len(zs), "p_wahr_mal_T_median": float(np.median(pe)), "anteil_ueber_2": float(np.mean(pe > 2.0)), "anteil_unter_halb": float(np.mean(pe < 0.5)),
            "v_median": float(np.median(v)), "unter_1": float(np.mean(v < 1))}


def lauf_streuband(fall: Fall, m: int, Ts, seed_teile, bekannt: bool = False, gumbel_zusatz: bool = False):
    """Nur Anpassung und Pegel (kein Intervall). `bekannt`: σ und ξ des Falls sind bekannt, nur μ wird geschätzt (Ein-Parameter-Fall).
    `gumbel_zusatz`: zusätzlich die Gumbel-Anpassung mit freiem (μ, σ) und ξ = 0 fest. Rückgabe: Wörterbuch mit z je T (GEV bzw. bekannt) und ggf. z_gumbel je T."""
    rng = SplitMix64(stream_seed(*seed_teile))
    x = np.array(ziehe_bloecke(fall, rng, m), float)
    try:
        if bekannt:
            mu = G.fit_mu_bekannt(x, fall.sigma, fall.xi)
            s, xi, ok = fall.sigma, fall.xi, True
        else:
            mu, s, xi, _, ok = G.fit_ml(x)
    except ValueError:
        return None
    res = {"ok": ok, "T": {T: {"z": G.gev_pegel(1.0 / T, mu, s, xi)} for T in Ts}}
    if gumbel_zusatz:
        gm, gs = G.fit_gumbel(x)
        for T in Ts:
            res["T"][T]["z_gumbel"] = G.gev_pegel(1.0 / T, gm, gs, 0.0)
    return res

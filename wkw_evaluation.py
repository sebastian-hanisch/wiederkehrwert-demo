"""Auswertung: Rechnung der vier Experimente und die Verdikte als reine Funktionen.

Jedes Verdikt ist `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Verdikte bekommen Zahlen, keine Zufalls-Seeds, damit
`tests/test_verdicts.py` Schwellen und jede genannte Zahl an von Hand gesetzten Werten prüft. Zahlen mit „eingefroren“ stammen aus `wkw_coverage.json`
(Zähleinheit Läufe, R und Standardfehler je Zelle), alle übrigen sind exakt gerechnet oder die Ergebnisse des Laufs in der App."""

import math
from dataclasses import dataclass
from fractions import Fraction

import wkw_constants as C
import wkw_reference as REF
import wkw_returnlevel as RL
import wkw_study as ST
from wkw_numeric import wilson
from wkw_rng import SplitMix64
from wkw_scenario import FALL_NAMEN, Fall, stream_seed, wahrer_pegel


@dataclass(frozen=True)
class Settings:
    fall: str = C.DEFAULT_FALL
    xi: float = C.DEFAULT_XI
    T: int = C.DEFAULT_T
    m: int = C.DEFAULT_M
    N: int = C.DEFAULT_N
    R1: int = C.DEFAULT_R1
    R3: int = C.DEFAULT_R3
    bekannt: str = C.BEKANNT_FREI
    R4: int = C.DEFAULT_R4
    delta: bool = True
    profil: bool = True
    boot: bool = False
    seed: int = C.DEFAULT_SEED


def fall_von(s: Settings) -> Fall:
    """Der Fall der App: GEV(100, 20, ξ) bei „GEV direkt“, sonst die exakte Blockverteilung."""
    if s.fall == "gev":
        return Fall("gev", s.xi, C.MU_GEV, C.SIGMA_GEV)
    return Fall(s.fall)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Formatierung (deutsche Zahlen)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def de(x: float, stellen: int = 1) -> str:
    """Deutsche Zahl: Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:.{stellen}f}".replace(".", ",").replace("-", "−")


def pct(x: float, stellen: int = 1) -> str:
    return de(100.0 * x, stellen) + " %"


def tausend(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: was heißt „T Jahre“?
# ---------------------------------------------------------------------------------------------------------------------------------------------

def simuliere_reihen(T: int, laenge: int, R: int, seed: int):
    """R Reihen von `laenge` unabhängigen Jahren; Rückgabe je Reihe das erste Jahr mit Überschreitung (1-basiert) oder 0, wenn keines.

    Jedes Jahr überschreitet den Pegel mit Wahrscheinlichkeit 1/T (gilt für jede stetige Verteilung der Jahre). Höchstens SIM_LIMIT gleichverteilte Zahlen: R wird gekürzt."""
    R_eff = max(1, min(R, C.SIM_LIMIT // max(1, laenge)))
    rng = SplitMix64(stream_seed(seed, 1))
    p = 1.0 / T
    erste = []
    for _ in range(R_eff):
        f = 0
        for j in range(1, laenge + 1):
            if rng.random() < p:
                f = j
                break
        erste.append(f)
    return erste


def wartezeit_analyse(s: Settings, mit_simulation: bool = True) -> dict:
    T, N, m = s.T, s.N, s.m
    a = {
        "T": T, "N": N, "m": m,
        "p_N": RL.p_mindestens_einmal(T, N), "p_T": RL.p_mindestens_einmal(T, T),
        "median": RL.median_wartezeit(T), "p_m": RL.p_mindestens_einmal(T, m),
        "p_m_andere": {t: RL.p_mindestens_einmal(t, m) for t in (100, 1000)},
        "jahre90": RL.jahre_fuer_sicherheit(T, Fraction(9, 10)),
        "kurve": [(n, RL.p_mindestens_einmal(T, n)) for n in range(1, C.N_MAX + 1)],
    }
    if mit_simulation:
        laenge = max(m, N)
        erste = simuliere_reihen(T, laenge, s.R1, s.seed)
        R = len(erste)
        k_m = sum(1 for f in erste if 0 < f <= m)
        k_N = sum(1 for f in erste if 0 < f <= N)
        a.update({"R_sim": R, "k_m": k_m, "k_N": k_N, "wilson_m": wilson(k_m, R), "wilson_N": wilson(k_N, R), "se_m": math.sqrt(a["p_m"] * (1 - a["p_m"]) / R),
                  "se_N": math.sqrt(a["p_N"] * (1 - a["p_N"]) / R)})
    return a


def verdict_wartezeit(a: dict):
    """Anker: T = 100: in 100 Jahren mit 63,4 % mindestens einmal überschritten, Median der Wartezeit 69 Jahre; 50-Jahres-Reihe enthält ihn in 39,5 % (1000-Jahres-Wert: 4,9 %); 90 % brauchen 230 Jahre."""
    T, N, m = a["T"], a["N"], a["m"]
    andere = 1000 if T != 1000 else 100
    in_T = "" if N == T else "in " + str(T) + " Jahren: " + pct(a["p_T"]) + ", "
    text = (
        f"**T = {T}:** in {N} Jahren wird der Pegel mit **{pct(a['p_N'])}** mindestens einmal überschritten "
        f"({in_T}Grenzwert für große T: 63,2 %); "
        f"der Median der Wartezeit ist **{a['median']} Jahre**, das Mittel {T}. Eine Reihe von {m} Jahren enthält ihn in **{pct(a['p_m'])}** der Reihen "
        f"({andere}-Jahres-Wert: {pct(a['p_m_andere'][andere])}); für 90 % bräuchte die Reihe **{tausend(a['jahre90'])} Jahre**."
    )
    return "info", text


def verdict_simulation(a: dict):
    """Gezogene Reihen gegen die exakte Zahl: im Band (|z| < 3) oder Abweichung nennen."""
    R = a["R_sim"]
    z = (a["k_m"] / R - a["p_m"]) / max(a["se_m"], 1e-12)
    text = (f"Gezogen ({tausend(R)} Reihen): {pct(a['k_m'] / R)} der Reihen von {a['m']} Jahren erreichen den Pegel, exakt {pct(a['p_m'])} "
            f"(Standardfehler der Simulation {de(100 * a['se_m'], 1)} Punkte; Abstand {de(abs(z), 1)} Standardfehler).")
    return ("success" if abs(z) < 3 else "warning"), text


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: eine Reihe, ein Pegel, drei Intervalle
# ---------------------------------------------------------------------------------------------------------------------------------------------

def einzelreihe(s: Settings) -> dict:
    """Eine Reihe mit Anpassung und den drei Intervallen; `fehler` enthält die Meldung, wenn keine Anpassung möglich ist."""
    fall = fall_von(s)
    if s.m < 3:
        return {"fehler": f"Mit {s.m} Block{'' if s.m == 1 else 'maxima'} ist keine Anpassung möglich: die GEV hat drei Parameter und braucht mindestens drei Blöcke (besser über 20)."}
    r = ST.lauf(fall, s.m, (s.T,), (s.seed, 2), profil=True, boot_B=C.BOOT_B, mit_daten=True)
    if r is None:
        return {"fehler": "Anpassung nicht möglich (alle Blöcke gleich)."}
    r["x_wahr"] = wahrer_pegel(fall, s.T)
    r["fall"] = fall
    return r


def lage_des_pegels(iv, x_wahr: float) -> str:
    """Wo liegt der wahre Pegel zum Intervall? Rückgabe: 'drin', 'ueber' (wahrer Pegel über dem Intervall: Intervall zu niedrig), 'unter' (Intervall zu hoch), 'ohne' (kein Intervall)."""
    if iv is None or math.isnan(iv[0]) or math.isnan(iv[1]):
        return "ohne"
    if x_wahr < iv[0]:
        return "unter"
    if x_wahr > iv[1]:
        return "ueber"
    return "drin"


LAGE_TEXT = {
    "drin": "enthält den wahren Pegel",
    "ueber": "liegt zu niedrig (der wahre Pegel liegt darüber)",
    "unter": "liegt zu hoch (der wahre Pegel liegt darunter)",
    "ohne": "ist nicht berechenbar",
}


def verdict_einzelreihe(T: int, x_wahr: float, z_hat: float, intervalle: dict, m: int):
    """intervalle: {'Delta': (unten, oben) oder None, 'Profil': ..., 'Bootstrap': ...}. Eine Reihe zeigt nur diesen Fall; ob die Intervalle halten, zeigt erst Experiment 4."""
    teile = []
    kinds = []
    for name, iv in intervalle.items():
        lage = lage_des_pegels(iv, x_wahr)
        kinds.append(lage)
        if iv is None:
            teile.append(f"{name}: {LAGE_TEXT[lage]}" + (" (Hesse-Matrix nicht positiv definit)" if name == "Delta" else ""))
        else:
            oben = "offen" if iv[1] == math.inf else de(iv[1], 1)
            unten = "offen" if iv[0] == -math.inf else de(iv[0], 1)
            teile.append(f"{name} [{unten}; {oben}] {LAGE_TEXT[lage]}")
    kopf = (f"Wahrer Pegel (exakt) {de(x_wahr, 1)}, Schätzwert {de(z_hat, 1)} (Verhältnis {de(z_hat / x_wahr, 2)}), Reihe aus {m} Blöcken, T = {T}. ")
    fuss = " Das ist eine einzige Reihe; ob ein Intervall sein Versprechen hält, zeigt erst die Überdeckung über viele Reihen (Experiment 4)."
    return ("success" if all(k == "drin" for k in kinds) else "warning"), kopf + "; ".join(teile) + "." + fuss


def verdict_modellfehler_anker(z: dict):
    """Eingefrorener Anker zu den Fällen aus Tageswerten (Zähleinheit Läufe): wie oft der geschätzte Pegel real mindestens doppelt so oft auftritt."""
    w = z["wahre_ueberschreitung"]
    return "info", (f"Eingefroren ({z['R']} Läufe, m = {z['m']}, T = {z['T']}, wahrer Pegel {de(z['x_wahr'], 3)}): in {pct(w['anteil_ueber_2'], 0)} der Läufe tritt der geschätzte Wert real "
                    f"mindestens doppelt so oft auf wie 1/T, in {pct(w['anteil_unter_halb'], 0)} höchstens halb so oft; Median von x̂/x_T {de(w['v_median'], 2)}.")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: viele Reihen, wie weit daneben (Streuband)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def streuband_live(s: Settings) -> dict:
    """R3 Läufe (Zähleinheit Läufe): V = Pegel-Dach / wahrer Pegel. `bekannt`: σ und ξ bekannt (nur GEV), nur μ geschätzt."""
    fall = fall_von(s)
    bekannt = s.bekannt == C.BEKANNT_SIGMA and s.fall == "gev"
    if s.m < 3 and not bekannt:
        return {"fehler": "Mit weniger als drei Blöcken ist keine Anpassung möglich."}
    x_wahr = wahrer_pegel(fall, s.T)
    laeufe = [ST.lauf_streuband(fall, s.m, (s.T,), (s.seed, 3, r), bekannt=bekannt) for r in range(s.R3)]
    laeufe = [r for r in laeufe if r is not None]
    if not laeufe:
        return {"fehler": "Keine Anpassung möglich."}
    band = ST.bewerte_streuband(laeufe, s.T, x_wahr)
    v = [r["T"][s.T]["z"] / x_wahr for r in laeufe]
    return {"x_wahr": x_wahr, "v": v, "band": band, "bekannt": bekannt}


def exaktes_band_bekannt(m: int, T: int, sigma: float = C.SIGMA_GEV, mu: float = C.MU_GEV):
    """Gumbel (ξ = 0), σ bekannt: V = 1 + Fehler·σ / x_T mit Fehler = -ln(G/m) (exakt aus der Gamma-Verteilung); Rückgabe (q05, q95) von V."""
    lo, hi = RL.gumbel_fehler_band(m)
    x_wahr = RL.gev_pegel(1.0 / T, mu, sigma, 0.0)
    return 1.0 + lo * sigma / x_wahr, 1.0 + hi * sigma / x_wahr


def verdict_streuband(band: dict, T: int, m: int, xi_text: str, eingefroren=None, bekannt: bool = False):
    """Anker: ξ = 0,2, m = 50: der 100er-Wert liegt zu 90 % zwischen dem 0,74- und 1,50-Fachen, der 1000er-Wert zwischen 0,57 und 2,22, in 7 % der Läufe mehr als doppelt so hoch."""
    text = (f"In {band['R']} Läufen ({xi_text}, m = {m}, T = {T}) lag der Schätzwert zu 90 % zwischen dem **{de(band['q05'], 2)}-** und **{de(band['q95'], 2)}-Fachen** des wahren Pegels "
            f"(Median {de(band['q50'], 2)}); in {pct(band['unter_1'], 0)} der Läufe darunter, in {pct(band['ueber_2'], 0)} mehr als doppelt so hoch.")
    if eingefroren is not None:
        e = eingefroren
        text += (f" Eingefroren ({tausend(e['R'])} Läufe, anderer Zufalls-Seed): {de(e['q05'], 2)} bis {de(e['q95'], 2)}, in {pct(e['ueber_2'], 0)} der Läufe mehr als doppelt so hoch, "
                 f"in {pct(e['unter_1'], 0)} darunter.")
    if bekannt:
        text += " Bei bekanntem σ und ξ ist der absolute Fehler (in Einheiten σ) für jedes T derselbe; das Verhältnis V hängt über x_T von T ab."
    return "info", text


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: hält das Intervall? (eingefroren + Probe)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def zelle_text(z: dict) -> str:
    return f"ξ = {de(z['xi'], 1)}" if z.get("fall") == "gev" else FALL_NAMEN[z["fall"]]


def verdict_ueberdeckung(z: dict):
    """Anker: ξ = 0,2, m = 50, T = 100: Delta 86,9 %, Bootstrap (B = 199) 90,8 %, Profil 95,2 % (je R = 1 000); Delta-Fehlgriffe zu niedrig, Profil auf beiden Seiten."""
    d, p = z["delta"], z["profil"]
    teile = [f"Delta {pct(d['cov'])}"]
    if "bootstrap" in z:
        teile.append(f"Perzentil-Bootstrap (B = {z['bootstrap']['B']}) {pct(z['bootstrap']['cov'])}")
    teile.append(f"Profil {pct(p['cov'])}")
    text = (f"{zelle_text(z)}, m = {z['m']}, T = {z['T']}: nominell 95 %, gemessen " + ", ".join(teile) + f" (je R = {tausend(z['R'])} Läufe; Standardfehler {de(100 * d['se'], 1)} bis {de(100 * p['se'], 1)} Punkte). "
            f"Verfehlte Delta-Intervalle: {pct(d['niedrig'])} zu niedrig, {pct(d['hoch'])} zu hoch; beim Profil {pct(p['niedrig'])} zu niedrig und {pct(p['hoch'])} zu hoch"
            + (f"; Profil-Intervalle ohne Obergrenze: {pct(p['offen_oben'])}" if p["offen_oben"] > 0 else "") + ". "
            f"Mediane Breite: Delta {de(d['breite_median'], 1)}, Profil {de(p['breite_median'], 1)} (Einheit σ des Falls).")
    kind = "warning" if min(d["cov"], p["cov"]) < C.UEBERDECKUNG_WARNUNG else "info"
    return kind, text


def probe_laeufe(s: Settings):
    """Probe: R4 Läufe live, Delta und/oder Profil und/oder Bootstrap (Zähleinheit Läufe)."""
    fall = fall_von(s)
    laeufe = []
    for r in range(s.R4):
        res = ST.lauf(fall, s.m, (s.T,), (s.seed, 4, r), profil=s.profil, delta=s.delta, boot_B=C.BOOT_B if s.boot else 0)
        laeufe.append(res)
    x_wahr = wahrer_pegel(fall, s.T)
    out = {"x_wahr": x_wahr, "R": s.R4, "arten": {}}
    for art, an in (("delta", s.delta), ("profil", s.profil), ("boot", s.boot)):
        if an:
            e = ST.bewerte_intervalle(laeufe, s.T, art, x_wahr)
            out["arten"][art] = e
    return out


def verdict_probe(probe: dict):
    teile = []
    for art, name in (("delta", "Delta"), ("profil", "Profil"), ("boot", "Bootstrap")):
        e = probe["arten"].get(art)
        if e is None:
            continue
        k = round(e["cov"] * e["R"])
        lo, hi = e["wilson"]
        teile.append(f"{name} {k} von {e['R']} ({pct(e['cov'], 0)}; 95-%-Band {pct(lo, 0)} bis {pct(hi, 0)})")
    if not teile:
        return "info", "Keine Intervallart gewählt."
    return "info", (f"Probe mit {probe['R']} Läufen: " + ", ".join(teile) + ". Bei so wenigen Läufen ist das Band breit: die Probe zeigt den Ablauf, belegt die Überdeckung aber nicht (dafür die eingefrorene Tabelle).")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Tabelle „Wie lang muss die Reihe sein?“
# ---------------------------------------------------------------------------------------------------------------------------------------------

def jahre_tabelle():
    """Zeilen (q, [Jahre für T = 10, 100, 1000]) exakt."""
    zeilen = []
    for q in C.Q_STUFEN:
        qf = Fraction(q).limit_denominator(1000)
        zeilen.append((q, [RL.jahre_fuer_sicherheit(T, qf) for T in C.T_STUFEN]))
    return zeilen


def verdict_reihenlaenge(T: int, jahre90: int, band100):
    """Anker: für T = 100 braucht die Reihe 230 Jahre, damit ihr Maximum den Wert mit 90 % erreicht; mit 100 Jahren liegt die Schätzung (ξ = 0,2) zwischen 0,80 und 1,28."""
    text = f"Für T = {T} braucht die Reihe **{tausend(jahre90)} Jahre**, damit ihr Maximum den Wert mit 90 % erreicht."
    if band100 is not None:
        text += f" Mit {band100['m']} Jahren liegt die Schätzung (ξ = {de(band100['xi'], 1)}, T = {band100['T']}, eingefroren, {tausend(band100['R'])} Läufe) zu 90 % zwischen {de(band100['q05'], 2)} und {de(band100['q95'], 2)} des wahren Pegels."
    return "info", text


def rand_der_daten_text(m: int) -> str:
    t_min = RL.T_pegel_ueber_maximum(m)
    return (f"Das Maximum aus {m} Jahren wird vom nächsten Jahr mit der Wahrscheinlichkeit genau 1/(m + 1) = 1/{m + 1} = {pct(RL.p_rand_der_daten(m), 2)} übertroffen, "
            f"für jede stetige Verteilung. Eine Wiederkehrperiode über {m + 1} Jahren liegt darum jenseits dessen, was die Rangordnung der Daten auflöst. "
            f"Erst ab T = {t_min} (das {de(t_min / m, 2)}-Fache von m) liegt der Pegel in mehr als der Hälfte der Reihen über dem größten Wert der Reihe.")

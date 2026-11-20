"""Vehikel B: ausgedachte Verweilzeit-Mischung (Tag = 100 Container), exakt aus der eingefrorenen Gewichtstabelle.

Die Tabelle `wkw_vehikel_b_tabelle.json` ist eine wörtliche Kopie der gemeinsamen Definition der Linie „Extremwerte erklärt“ (Eigentümer: die Demo „Überschreitungen und die GPD“,
Erzeugung als exakte Aufzählung, Stand 2026-11-20): 1224 verschiedene Komponenten (Gewicht, μ, s) der logarithmierten Verweilzeit (je Komponente normalverteilt),
4704 Aufzählungsschritte, 2376 davon mit Gewicht größer 0. Die Kopie wird nicht verändert; `tests/test_vehikel_b.py` prüft sie gegen eine eigene Aufzählung der Definition.

Verweilzeit in Tagen. Zielgrößen: (a) Pegel je Container zur Überschreitungswahrscheinlichkeit 1e-5: 287,158 Tage; (b) Tagesmaximum aus 100 Containern, einmal in 1000 Tagen:
287,123 Tage (T = 1000), 155,128 Tage (T = 100). Hier gebraucht wird (b). Alles ausgedacht, kein Echtdatenbezug."""

import json
import math
from pathlib import Path

import numpy as np

import wkw_numeric as N

TABELLE = Path(__file__).resolve().parent / "wkw_vehikel_b_tabelle.json"
CONTAINER_JE_TAG = 100
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_ERFC = np.frompyfunc(math.erfc, 1, 1)
_CACHE = {}


def lade_komponenten():
    """(Gewichte, μ, s) als numpy-Felder, in der Reihenfolge der Tabelle."""
    if "komp" not in _CACHE:
        daten = json.loads(TABELLE.read_text(encoding="utf-8"))
        k = np.array(daten["komponenten"], float)
        _CACHE["komp"] = (k[:, 0], k[:, 1], k[:, 2])
    return _CACHE["komp"]


def sf_container(x: float) -> float:
    """Überschreitungswahrscheinlichkeit eines Containers: Summe_i w_i (1 - Φ((ln x - μ_i)/s_i))."""
    w, mu, s = lade_komponenten()
    z = (math.log(x) - mu) / s
    return float(np.sum(w * 0.5 * _ERFC(z * _INV_SQRT2).astype(float)))


def log_cdf_tagesmax(x: float, n: int = CONTAINER_JE_TAG) -> float:
    """ln F^n(x) = n ln(1 - sf(x)) für das Maximum aus n Containern."""
    return n * math.log1p(-sf_container(x))


def pegel_tagesmax(T: float, n: int = CONTAINER_JE_TAG) -> float:
    """Wert x mit F^n(x) = 1 - 1/T (Tagesmaximum aus n Containern, exakt aus der Mischung) per Brent in ln x."""
    ziel = math.log1p(-1.0 / T)
    return math.exp(N.brent(lambda lx: n * math.log1p(-sf_container(math.exp(lx))) - ziel, -3.0, 14.0, xtol=1e-13))


def pegel_container(p: float) -> float:
    """Wert x mit sf(x) = p je Container (Zielgröße (a): p = 1e-5 gibt 287,158 Tage)."""
    return math.exp(N.brent(lambda lx: math.log(sf_container(math.exp(lx))) - math.log(p), -3.0, 14.0, xtol=1e-13))


def mittel_container() -> float:
    w, mu, s = lade_komponenten()
    return float(np.sum(w * np.exp(mu + s * s / 2.0)))


def umkehrtabelle():
    """Tabelle (ln sf aufsteigend, ln x) zur Ziehung des Tagesmaximums durch Umkehrung: ein Zug je Tag statt 100 Containerzüge.

    Gitter in ln x von 0 bis 9,5 (x von 1 bis rund 13 000 Tage), 2 401 Punkte; lineare Interpolation in ln sf gegen ln x. Einmal je Prozess gerechnet (rund 1 s)."""
    if "tab" not in _CACHE:
        w, mu, s = lade_komponenten()
        lx = np.linspace(0.0, 9.5, 2401)
        lnsf = np.empty(lx.size)
        for i, v in enumerate(lx):
            z = (v - mu) / s
            lnsf[i] = math.log(max(float(np.sum(w * 0.5 * _ERFC(z * _INV_SQRT2).astype(float))), 1e-300))
        _CACHE["tab"] = (lnsf[::-1].copy(), lx[::-1].copy())
    return _CACHE["tab"]


def tagesmax_aus_u(u: float, exact_log, n: int = CONTAINER_JE_TAG) -> float:
    """Tagesmaximum (Tage) zur gleichverteilten Zahl u in (0, 1): F^n(x) = u, also sf(x) = 1 - u^(1/n) (Umkehrung per Tabelle)."""
    if u <= 0.0:
        u = 2.0 ** -54
    ln_sf = math.log(-math.expm1(exact_log(u) / n))
    tab_sf, tab_lx = umkehrtabelle()
    return math.exp(float(np.interp(ln_sf, tab_sf, tab_lx)))


def zug_container(rng):
    """Ein Container nach der Logik der Tabelle: Komponente über die Summenverteilung der Gewichte, dann exp(μ + s · normal()) (zur Gegenprobe der Umkehrung)."""
    w, mu, s = lade_komponenten()
    if "cw" not in _CACHE:
        cw = np.cumsum(w)
        _CACHE["cw"] = cw / cw[-1]
    i = min(int(np.searchsorted(_CACHE["cw"], rng.random())), len(w) - 1)
    return math.exp(float(mu[i]) + float(s[i]) * rng.normal())

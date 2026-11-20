"""Szenarien: wie die Blockmaxima („Jahre“) entstehen und wie groß der wahre Pegel ist.

Alle Blöcke sind ausgedacht (kein Echtdatenbezug). Zufall nur aus `wkw_rng.SplitMix64` (reine Vorlage, auf jeder Plattform dieselben Zahlen); Logarithmen über `exact_log`.
Fälle:
* `gev`:        Blöcke direkt aus GEV(μ, σ, ξ) per Inversion; wahrer Pegel = GEV-Quantil (exakt).
* `exp365`:     Jahresmaximum aus n = 365 Exponential(1)-Tageswerten; wahrer Pegel exakt aus F^365 (die GEV ist hier nur Grenzgesetz).
* `normal365`:  Jahresmaximum aus 365 Normal(0, 1)-Tageswerten; wahrer Pegel exakt aus Φ^365.
* `vwz`:        Vehikel B: Tagesmaximum aus 100 Containern der Verweilzeit-Mischung (Tabelle `wkw_vehikel_b_tabelle.json`); wahrer Pegel exakt aus F^100.
Bei den drei letzten Fällen wird je Block ein Zug aus F^n durch Umkehrung gezogen (ein Zug je Jahr statt 365 Tageswerten)."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import wkw_returnlevel as RL
import wkw_vehikel_b as VB
from wkw_rng import MASK, SplitMix64, exact_log

FAELLE = ("gev", "exp365", "normal365", "vwz")
FALL_NAMEN = {
    "gev": "GEV direkt (Lehrbuch)",
    "exp365": "Exponential, Jahr aus 365 Tagen",
    "normal365": "Normal, Jahr aus 365 Tagen",
    "vwz": "Vehikel B, Tag aus 100 Containern",
}
MU_STANDARD, SIGMA_STANDARD = 0.0, 1.0       # Ort und Skala der GEV-Zellen der Überdeckungstabelle
_NORMAL = NormalDist()


def stream_seed(*teile: int) -> int:
    """Ein Zufalls-Seed aus mehreren Ganzzahlen (Basis, Zelle, Lauf): jeder Lauf bekommt seinen eigenen, von den anderen unabhängigen Strom."""
    s = 0x243F6A8885A308D3
    for t in teile:
        s = SplitMix64((s ^ (int(t) & MASK)) & MASK).next_u64()
    return s


@dataclass(frozen=True)
class Fall:
    art: str = "gev"
    xi: float = 0.2
    mu: float = MU_STANDARD
    sigma: float = SIGMA_STANDARD

    def beschreibung(self) -> str:
        if self.art == "gev":
            return f"GEV(μ = {self.mu:g}, σ = {self.sigma:g}, ξ = {self.xi:g})"
        return FALL_NAMEN[self.art]


def _u(rng) -> float:
    u = rng.random()
    return u if u > 0.0 else 2.0 ** -54


def zug_gev(rng, mu: float, sigma: float, xi: float) -> float:
    """Ein Block aus GEV(μ, σ, ξ) per Inversion: x = μ + σ (w^(-ξ) - 1)/ξ mit w = -ln u."""
    w = -exact_log(_u(rng))
    if abs(xi) < RL.GUMBEL_GRENZE:
        return mu - sigma * exact_log(w)
    return mu + sigma * (math.pow(w, -xi) - 1.0) / xi


def zug_exponential(rng, n: int = 365) -> float:
    """Maximum aus n Exponential(1)-Werten, F^n(x) = u: x = -ln(1 - u^(1/n))."""
    return -exact_log(-math.expm1(exact_log(_u(rng)) / n))


def zug_normal(rng, n: int = 365) -> float:
    """Maximum aus n Normal(0, 1)-Werten, Φ^n(x) = u: x = -Φ^-1(1 - u^(1/n))."""
    return -_NORMAL.inv_cdf(-math.expm1(exact_log(_u(rng)) / n))


def zug_vehikel_b(rng) -> float:
    return VB.tagesmax_aus_u(_u(rng), exact_log)


def ziehe_bloecke(fall: Fall, rng, m: int):
    """m Blockmaxima als Liste."""
    if fall.art == "gev":
        return [zug_gev(rng, fall.mu, fall.sigma, fall.xi) for _ in range(m)]
    if fall.art == "exp365":
        return [zug_exponential(rng) for _ in range(m)]
    if fall.art == "normal365":
        return [zug_normal(rng) for _ in range(m)]
    if fall.art == "vwz":
        return [zug_vehikel_b(rng) for _ in range(m)]
    raise ValueError(f"unbekannter Fall {fall.art!r}")


def wahrer_pegel(fall: Fall, T: float) -> float:
    """Exakter Pegel zur Überschreitungswahrscheinlichkeit 1/T je Block."""
    if fall.art == "gev":
        return RL.gev_pegel(1.0 / T, fall.mu, fall.sigma, fall.xi)
    if fall.art == "exp365":
        return RL.pegel_exponential(T, 365)
    if fall.art == "normal365":
        return RL.pegel_normal(T, 365)
    if fall.art == "vwz":
        return VB.pegel_tagesmax(T)
    raise ValueError(f"unbekannter Fall {fall.art!r}")


def wahre_ueberschreitung(fall: Fall, x: float) -> float:
    """Wahre Überschreitungswahrscheinlichkeit je Block eines Wertes x: 1 - F_Block(x) (für den geschätzten Pegel; F^n aus der exakten Verteilung)."""
    if fall.art == "gev":
        return 1.0 - RL.gev_cdf(x, fall.mu, fall.sigma, fall.xi)
    if x <= 0:
        return 1.0
    if fall.art == "exp365":
        lf = 365 * math.log1p(-math.exp(-x))
    elif fall.art == "normal365":
        lf = 365 * math.log1p(-0.5 * math.erfc(x / math.sqrt(2.0)))
    elif fall.art == "vwz":
        lf = VB.log_cdf_tagesmax(x)
    else:
        raise ValueError(f"unbekannter Fall {fall.art!r}")
    return -math.expm1(lf)

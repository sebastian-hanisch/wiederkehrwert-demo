"""Wiederkehrwert (Pegel) und Wiederkehrperiode: exakte Rechnung ohne Zufall.

Vorzeichen der Linie: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ mit endlichem Rand. `scipy.stats.genextreme` hat c = -ξ (nur im Test benutzt).
Ein Block ist ein Jahr; p = 1/T ist die Überschreitungswahrscheinlichkeit je Block. Die Wiederkehrperiode T ist ein Mittel der Wartezeit, keine Frist."""

import math
from fractions import Fraction
from statistics import NormalDist

import wkw_numeric as N

GUMBEL_GRENZE = 1e-9


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Wartezeit und Reihenlänge (unabhängige Jahre, stetige Verteilung, exakt)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def p_mindestens_einmal(T: float, n_jahre: int) -> float:
    """P(mindestens eine Überschreitung des T-Jahres-Wertes innerhalb von n_jahre Jahren) = 1 - (1 - 1/T)^n_jahre (unabhängige Jahre)."""
    if T < 1:
        raise ValueError("T muss mindestens 1 sein")
    if T == 1:
        return 1.0 if n_jahre >= 1 else 0.0
    return -math.expm1(n_jahre * math.log1p(-1.0 / T))


def p_mindestens_einmal_bruch(T: int, n_jahre: int) -> Fraction:
    """Dasselbe als exakter Bruch (T ganzzahlig)."""
    return 1 - Fraction(T - 1, T) ** n_jahre


def median_wartezeit(T: int) -> int:
    """Kleinstes k mit (1 - 1/T)^k < 1/2, in ganzen Zahlen gerechnet: 2 (T-1)^k < T^k (T = 100: 69, T = 1000: 693)."""
    if T < 2:
        return 1
    k = max(1, int(math.log(0.5) / math.log1p(-1.0 / T)) - 2)
    while 2 * (T - 1) ** k >= T ** k:
        k += 1
    while k > 1 and 2 * (T - 1) ** (k - 1) < T ** (k - 1):
        k -= 1
    return k


def jahre_fuer_sicherheit(T: int, q: Fraction) -> int:
    """Kleinste Reihenlänge m, bei der das größte von m Jahren den T-Jahres-Wert mit Wahrscheinlichkeit mindestens q erreicht: (1 - q) T^m >= (T-1)^m ... exakt in ganzen Zahlen."""
    if T < 2:
        return 1
    q = Fraction(q)
    rest = 1 - q
    if rest <= 0:
        raise ValueError("q muss kleiner als 1 sein")
    m = max(1, int(math.log(float(rest)) / math.log1p(-1.0 / T)) - 2)
    while Fraction(T - 1, T) ** m > rest:
        m += 1
    while m > 1 and Fraction(T - 1, T) ** (m - 1) <= rest:
        m -= 1
    return m


def T_fuer_sicherheit(q: float, n_jahre: int) -> float:
    """Wiederkehrperiode T, bei der der Pegel in n_jahre Jahren mit Wahrscheinlichkeit höchstens 1 - q überschritten wird: T = 1 / (1 - q^(1/n_jahre))."""
    return 1.0 / (-math.expm1(math.log(q) / n_jahre))


def p_rand_der_daten(m: int) -> float:
    """Das Maximum aus m Jahren wird vom nächsten Jahr mit Wahrscheinlichkeit genau 1/(m+1) übertroffen, für jede stetige Verteilung."""
    return 1.0 / (m + 1)


def T_pegel_ueber_maximum(m: int) -> int:
    """Kleinstes ganzzahliges T, bei dem der T-Jahres-Wert in mehr als der Hälfte der Reihen von m Jahren über dem größten Wert der Reihe liegt:
    (1 - 1/T)^m > 1/2, in ganzen Zahlen gerechnet: 2 (T-1)^m > T^m (stetige Verteilung, unabhängige gleich verteilte Jahre; m = 50: 73)."""
    if m < 1:
        raise ValueError("m muss mindestens 1 sein")
    T = max(2, int(m / math.log(2)) - 2)
    while 2 * (T - 1) ** m <= T ** m:
        T += 1
    while T > 2 and 2 * (T - 2) ** m > (T - 1) ** m:
        T -= 1
    return T


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Pegel der GEV und der exakten Blockverteilungen F^n
# ---------------------------------------------------------------------------------------------------------------------------------------------

def gev_pegel(p: float, mu: float, sigma: float, xi: float) -> float:
    """Wiederkehrwert zur Überschreitungswahrscheinlichkeit p = 1/T je Block: x = mu + sigma ((-ln(1-p))^(-xi) - 1)/xi, bei xi = 0: mu - sigma ln(-ln(1-p))."""
    y = -math.log1p(-p)
    if abs(xi) < GUMBEL_GRENZE:
        return mu - sigma * math.log(y)
    return mu + sigma * (math.pow(y, -xi) - 1.0) / xi


def gev_cdf(x: float, mu: float, sigma: float, xi: float) -> float:
    z = (x - mu) / sigma
    if abs(xi) < GUMBEL_GRENZE:
        return math.exp(-math.exp(-z))
    t = 1.0 + xi * z
    if t <= 0:
        return 0.0 if xi > 0 else 1.0
    return math.exp(-math.pow(t, -1.0 / xi))


def tagesanteil(T: float, n: int) -> float:
    """Überschreitungswahrscheinlichkeit je Tageswert, damit das Maximum aus n Tageswerten (ein Jahr) den Wert mit Wahrscheinlichkeit 1/T übertrifft: 1 - (1 - 1/T)^(1/n)."""
    return -math.expm1(math.log1p(-1.0 / T) / n)


def pegel_exponential(T: float, n: int = 365) -> float:
    """Jahresmaximum aus n Exponential(1)-Tageswerten: x_T = -ln(1 - (1 - 1/T)^(1/n)) (F^n geschlossen)."""
    return -math.log(tagesanteil(T, n))


def pegel_normal(T: float, n: int = 365) -> float:
    """Jahresmaximum aus n Normal(0, 1)-Tageswerten: x_T = Phi^-1((1 - 1/T)^(1/n)); die Umkehrfunktion der Standardbibliothek (reines Python)."""
    return -NormalDist().inv_cdf(tagesanteil(T, n))


def bisektion_pegel(logF, T: float, lo: float, hi: float, schritte: int = 200) -> float:
    """Zweiter Rechenweg zur Umkehrformel: Bisektion auf ln F^n(x) = ln(1 - 1/T) (logF liefert ln F^n)."""
    ziel = math.log1p(-1.0 / T)
    for _ in range(schritte):
        mid = 0.5 * (lo + hi)
        if logF(mid) < ziel:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Ein-Parameter-Fall: Gumbel, sigma bekannt (alles exakt über die Gamma-Verteilung)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def gumbel_mu_dach(x, sigma: float) -> float:
    """ML für mu bei bekanntem sigma: mu_hat = -sigma ln(Mittel von exp(-x_i/sigma)) (stabil über das Minimum)."""
    z = [-xi / sigma for xi in x]
    zmax = max(z)
    return -sigma * (zmax + math.log(sum(math.exp(v - zmax) for v in z) / len(z)))


def gumbel_fehler_band(m: int, p_lo: float = 0.05, p_hi: float = 0.95):
    """Quantile des Fehlers mu_hat - mu in Einheiten sigma (gleich für JEDEN Wiederkehrwert): mu_hat - mu = -sigma ln(G/m), G ~ Gamma(m, 1)."""
    return -math.log(N.gamma_ppf_int(m, p_hi) / m), -math.log(N.gamma_ppf_int(m, p_lo) / m)


def gumbel_delta_faktor(m: int, z: float = 1.959963984540054) -> float:
    return z / math.sqrt(m)


def gumbel_profil_faktoren(m: int, crit: float = 3.841458820694124):
    """Profil-Likelihood-Endpunkte für mu bei bekanntem sigma: mu_hat + ln r mit r - 1 - ln r = crit/(2m), r1 < 1 < r2."""
    ziel = crit / (2.0 * m)
    f = lambda r: r - 1.0 - math.log(r) - ziel
    r1 = N.brent(f, 1e-12, 1.0, xtol=1e-15)
    r2 = N.brent(f, 1.0, 1.0 + 4.0 * math.sqrt(ziel) + 4.0 * ziel + 10.0, xtol=1e-15)
    return r1, r2


def gumbel_ueberdeckung_exakt(m: int):
    """Exakte Überdeckung des nominell 95-%-Intervalls für mu (und jeden Wiederkehrwert) bei bekanntem sigma: (Delta, Profil). Aus der Gamma-Verteilungsfunktion von G."""
    c = gumbel_delta_faktor(m)
    r1, r2 = gumbel_profil_faktoren(m)
    # Delta deckt mu genau dann, wenn |ln(G/m)| <= c, also G/m in [exp(-c), exp(c)]; das Profil, wenn G/m in [r1, r2] (Pivot G ~ Gamma(m, 1))
    delta = N.gamma_cdf_int(m, m * math.exp(c)) - N.gamma_cdf_int(m, m * math.exp(-c))
    profil = N.gamma_cdf_int(m, m * r2) - N.gamma_cdf_int(m, m * r1)
    return delta, profil


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Quantile von Stichproben
# ---------------------------------------------------------------------------------------------------------------------------------------------

def quantil(werte, q: float) -> float:
    """Empirisches Quantil mit linearer Interpolation (wie numpy.quantile, Standard)."""
    v = sorted(werte)
    n = len(v)
    if n == 0:
        raise ValueError("leere Liste")
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return v[lo] + (pos - lo) * (v[hi] - v[lo])

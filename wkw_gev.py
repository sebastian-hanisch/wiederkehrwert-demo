"""GEV-Anpassung (Maximum Likelihood) und die drei Intervalle für den Wiederkehrwert: Delta, Profil-Likelihood, parametrischer Perzentil-Bootstrap.

Nur numpy und Standardbibliothek: eigenes Nelder-Mead (`wkw_numeric.nelder_mead`), eigene Wurzelsuche. `scipy.stats.genextreme` (c = -ξ) steht nur in den Tests.
Vorzeichen der Linie: ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ. ML mit der Nebenbedingung ξ >= -1 (für ξ < -1 ist die Likelihood unbeschränkt).
Parametrisierung der Optimierung: θ = (μ, ln σ, ξ), gerechnet an standardisierten Daten (Mittel 0, Standardabweichung 1), zurückgerechnet auf die Einheit der Daten.

Die drei Intervalle (alle nominell 95 %):
* Delta: beobachtete Information (numerische Hesse-Matrix von -ln L), symmetrisch um den Schätzwert (Coles 2001, Gl. 3.11 in eigenen Worten).
* Profil-Likelihood: alle Pegel z, deren Profil-Devianz höchstens χ²(1; 0,95) = 3,8415 ist (Umparametrisierung μ(z, σ, ξ), Coles Gl. 3.12 in eigenen Worten).
* Bootstrap: B Anpassungen an Stichproben aus der angepassten GEV (parametrisch), Perzentilintervall; keine BCa- oder Studentisierungs-Korrektur."""

import math

import numpy as np

import wkw_numeric as N
from wkw_returnlevel import GUMBEL_GRENZE, gev_pegel

XI_MIN = -1.0
HUGE = 1e300
EULER = 0.5772156649015329
Z95 = 1.959963984540054
CHI2_95 = 3.841458820694124           # χ²(1; 0,95)
GUMBEL_NLL = 1e-7                      # |ξ| darunter: Gumbel-Likelihood


def nll(theta, x) -> float:
    """Negative Log-Likelihood der GEV, theta = (μ, ln σ, ξ), x ein numpy-Feld."""
    mu, ls, xi = theta
    if xi < XI_MIN or not math.isfinite(mu) or not math.isfinite(ls):
        return HUGE
    with np.errstate(all="ignore"):
        z = (x - mu) * math.exp(-ls)
        if abs(xi) < GUMBEL_NLL:
            v = x.size * ls + float(z.sum()) + float(np.exp(-z).sum())
        else:
            t = 1.0 + xi * z
            if float(t.min()) <= 0.0:
                return HUGE
            lt = np.log(t)
            v = x.size * ls + (1.0 + 1.0 / xi) * float(lt.sum()) + float(np.exp(-lt / xi).sum())
    return v if math.isfinite(v) else HUGE


def l_momente_start(x):
    """Startwerte (μ, σ, ξ) aus den L-Momenten der Stichprobe (Näherung von Hosking, Wallis und Wood nach k = 7,8590 c + 2,9554 c²; ξ = -k)."""
    xs = np.sort(np.asarray(x, float))
    n = xs.size
    j = np.arange(1, n + 1)
    b0 = float(xs.mean())
    b1 = float(np.sum((j - 1) / (n - 1) * xs) / n)
    b2 = float(np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * xs) / n)
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    if not l2 > 0:
        raise ValueError("L-Momente entartet")
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - math.log(2.0) / math.log(3.0)
    k = 7.8590 * c + 2.9554 * c * c
    if abs(k) < 1e-9:
        sigma = l2 / math.log(2.0)
        mu = b0 - sigma * EULER
    else:
        g = math.gamma(1.0 + k)
        sigma = l2 * k / ((1.0 - 2.0 ** (-k)) * g)
        mu = b0 - sigma * (1.0 - g) / k
    return mu, sigma, -k


def _start(xs):
    """Startwert für die Optimierung an standardisierten Daten (L-Momente, sonst Gumbel nach Momenten)."""
    try:
        mu, s, xi = l_momente_start(xs)
        xi = float(min(max(xi, -0.6), 0.6))
        if s > 0 and math.isfinite(mu) and math.isfinite(s) and nll((mu, math.log(s), xi), xs) < HUGE / 10:
            return [mu, math.log(s), xi]
    except (ValueError, OverflowError, ZeroDivisionError):
        pass
    s = math.sqrt(6.0) / math.pi
    return [-EULER * s, math.log(s), 0.0]


def fit_ml(x, start=None, restart=True, loose=False):
    """ML-GEV an die Blockmaxima x. Rückgabe (μ, σ, ξ, nll, ok); ok ist falsch, wenn ξ an der Grenze -1 klebt oder die Likelihood nicht endlich ist.

    `start` (μ, σ, ξ) in Einheiten der Daten als Warmstart (Bootstrap). Gerechnet wird an standardisierten Daten; nll gilt für die Daten in ihrer Einheit."""
    x = np.asarray(x, float)
    if x.size < 3:
        raise ValueError("Anpassung braucht mindestens 3 Blöcke")
    loc = float(x.mean())
    scale = float(x.std(ddof=1))
    if not scale > 0:
        raise ValueError("alle Blöcke gleich: keine Anpassung")
    xs = (x - loc) / scale
    if start is None:
        p0 = _start(xs)
    else:
        p0 = [(start[0] - loc) / scale, math.log(start[1] / scale), start[2]]
        if nll(p0, xs) >= HUGE / 10:
            p0 = _start(xs)
    f = lambda th: nll(th, xs)
    sc = [0.5, 0.05, 0.05]
    best, fbest, _ = N.nelder_mead(f, p0, sc, xatol=1e-5 if loose else 1e-7, fatol=1e-8 if loose else 1e-10)
    if restart:
        b2, f2, _ = N.nelder_mead(f, best, [0.05 * s for s in sc], xatol=1e-8, fatol=1e-11)
        if f2 <= fbest:
            best, fbest = b2, f2
    mu = loc + scale * best[0]
    sigma = scale * math.exp(best[1])
    xi = best[2]
    nll_orig = fbest + x.size * math.log(scale)
    ok = bool(fbest < HUGE / 10 and xi > XI_MIN + 1e-3)
    return mu, sigma, xi, nll_orig, ok


def fit_mu_bekannt(x, sigma: float, xi: float):
    """ML nur für μ, wenn σ und ξ bekannt sind (Ein-Parameter-Fall); goldener Schnitt auf -ln L(μ) im zulässigen Bereich."""
    x = np.asarray(x, float)
    ls = math.log(sigma)
    xmin, xmax = float(x.min()), float(x.max())
    if abs(xi) < GUMBEL_NLL:
        lo, hi = xmin - 40.0 * sigma, xmax + 40.0 * sigma
    elif xi > 0:
        lo, hi = xmax - 60.0 * sigma, xmin + sigma / xi - 1e-9 * sigma
    else:
        lo, hi = xmax + sigma / xi + 1e-9 * sigma, xmin + 60.0 * sigma
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll((c, ls, xi), x), nll((d, ls, xi), x)
    for _ in range(200):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll((c, ls, xi), x)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll((d, ls, xi), x)
        if b - a < 1e-10 * sigma:
            break
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Delta-Intervall
# ---------------------------------------------------------------------------------------------------------------------------------------------

def hesse_nll(theta, x):
    """Numerische Hesse-Matrix von -ln L an theta = (μ, ln σ, ξ) (zentrale Differenzen)."""
    with np.errstate(all="ignore"):
        return _hesse_nll(theta, x)


def _hesse_nll(theta, x):
    th = np.asarray(theta, float)
    h = np.array([1e-4 * math.exp(th[1]), 1e-4, 1e-4])
    f0 = nll(th, x)
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3)
            ei[i] = h[i]
            ej = np.zeros(3)
            ej[j] = h[j]
            if i == j:
                H[i, i] = (nll(th + ei, x) - 2 * f0 + nll(th - ei, x)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (nll(th + ei + ej, x) - nll(th + ei - ej, x) - nll(th - ei + ej, x) + nll(th - ei - ej, x)) / (4 * h[i] * h[j])
    return H


def _pegel_theta(theta, p):
    return gev_pegel(p, theta[0], math.exp(theta[1]), theta[2])


def delta_intervall(x, mu, sigma, xi, p, level_z: float = Z95):
    """Delta-Intervall für den Pegel zur Überschreitungswahrscheinlichkeit p. Rückgabe (unten, oben, Standardfehler) oder None, wenn die Hesse-Matrix nicht positiv definit ist."""
    x = np.asarray(x, float)
    theta = np.array([mu, math.log(sigma), xi])
    H = hesse_nll(theta, x)
    if not np.all(np.isfinite(H)):
        return None
    try:
        if np.any(np.linalg.eigvalsh(H) <= 0):
            return None
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov)):
        return None
    g = np.zeros(3)
    h = np.array([1e-4 * sigma, 1e-4, 1e-4])
    for i in range(3):
        e = np.zeros(3)
        e[i] = h[i]
        g[i] = (_pegel_theta(theta + e, p) - _pegel_theta(theta - e, p)) / (2 * h[i])
    var = float(g @ cov @ g)
    if not var > 0:
        return None
    se = math.sqrt(var)
    z = _pegel_theta(theta, p)
    return z - level_z * se, z + level_z * se, se


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Profil-Likelihood-Intervall
# ---------------------------------------------------------------------------------------------------------------------------------------------

def profil_nll(z, x, p, start_ls_xi, extra_starts=()):
    """Profil von -ln L bei festem Pegel z (Überschreitungswahrscheinlichkeit p): Minimum über (ln σ, ξ), μ folgt aus z. Rückgabe (Wert, (ln σ, ξ)).

    Die Profilfläche hat lokale Minima (am Rand ξ → -1 und bei großem ξ): darum Mehrfachstart. Kandidaten sind die Warmstarts und ein Raster um ln σ des ML-Punkts;
    die drei besten Kandidaten starten je ein Nelder-Mead. (Ein einzelner Start machte das Intervall nach oben zu eng: PLAN-03 Prüfung U3-8.)"""
    y = -math.log1p(-p)
    ly = math.log(y)

    def f(par):
        ls, xi = par
        if xi < XI_MIN:
            return HUGE
        s = math.exp(ls)
        if abs(xi) < GUMBEL_GRENZE:
            mu = z + s * ly
        else:
            mu = z - s * (math.exp(-xi * ly) - 1.0) / xi
        return nll((mu, ls, xi), x)

    ls0 = extra_starts[0][0] if extra_starts else start_ls_xi[0]
    kandidaten = [tuple(start_ls_xi)] + [tuple(e) for e in extra_starts] + [(ls0 + dl, xi) for dl in (-0.5, 0.0, 0.5) for xi in _PROFIL_XI_RASTER]
    bewertet = sorted((f(c), i) for i, c in enumerate(kandidaten))
    starts, gesehen = [], set()
    for v, i in bewertet:
        if v >= HUGE / 10 or len(starts) >= 3:
            continue
        schluessel = (round(kandidaten[i][0], 6), round(kandidaten[i][1], 6))
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        starts.append(kandidaten[i])
    if not starts:
        starts = [tuple(start_ls_xi)]
    best = None
    for st in starts:
        r, fr, _ = N.nelder_mead(f, list(st), [0.2, 0.1], xatol=1e-7, fatol=1e-10, maxiter=1500)
        if best is None or fr < best[1]:
            best = (r, fr)
    return best[1], tuple(best[0])


_PROFIL_XI_RASTER = (-0.9, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3)


def profil_intervall(x, mu, sigma, xi, nll_hat, p, se=None, max_expand: int = 14, xtol_sigma: float = 1e-4):
    """Profil-Likelihood-Intervall (95 %) für den Pegel. Eine Seite ist ±inf, wenn die Profilkurve die Schwelle im Suchbereich nicht erreicht (Intervall dort offen).

    Rückgabe (unten, oben). Je Pegel zwei Optimierungsläufe (Warmstart vom Nachbarn und vom ML-Punkt), damit das Profil nicht in einem lokalen Minimum hängt."""
    x = np.asarray(x, float)
    crit = 0.5 * CHI2_95
    z_hat = gev_pegel(p, mu, sigma, xi)
    ls_hat = math.log(sigma)
    warm = {"s": (ls_hat, xi)}

    def g(z):
        v, par = profil_nll(z, x, p, warm["s"], extra_starts=((ls_hat, xi),))
        warm["s"] = par
        return v - nll_hat - crit

    out = []
    for sgn in (-1.0, 1.0):
        warm["s"] = (ls_hat, xi)
        step = 0.5 * sigma if se is None else max(1.0 * se, 0.05 * sigma)
        val = None
        for _ in range(max_expand):
            zz = z_hat + sgn * step
            if g(zz) > 0:
                a, b = (zz, z_hat) if sgn < 0 else (z_hat, zz)
                warm["s"] = (ls_hat, xi)
                val = N.brent(g, a, b, xtol=xtol_sigma * sigma)
                break
            step *= 2.0
        out.append(val)
    return (-math.inf if out[0] is None else out[0]), (math.inf if out[1] is None else out[1])


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Parametrischer Bootstrap
# ---------------------------------------------------------------------------------------------------------------------------------------------

def bootstrap_anpassungen(zieher, mu, sigma, xi, B: int = 199):
    """B Anpassungen an Stichproben aus der angepassten GEV (parametrischer Bootstrap): `zieher()` liefert eine neue Stichprobe gleicher Größe.

    Warmstart bei den Schätzwerten, lockere Toleranz, kein Neustart. Rückgabe: Liste (μ*, σ*, ξ*); misslungene Anpassungen fehlen."""
    fits = []
    for _ in range(B):
        xb = zieher()
        try:
            m2, s2, x2, _, _ = fit_ml(xb, start=(mu, sigma, xi), restart=False, loose=True)
        except ValueError:
            continue
        fits.append((m2, s2, x2))
    return fits


def bootstrap_intervall(fits, p, level: float = 0.95):
    """Perzentilintervall des Pegels zur Überschreitungswahrscheinlichkeit p aus den Bootstrap-Anpassungen (lineare Interpolation der Quantile wie numpy)."""
    zs = [gev_pegel(p, m2, s2, x2) for m2, s2, x2 in fits]
    zs = [v for v in zs if math.isfinite(v)]
    a = (1.0 - level) / 2.0
    arr = np.array(zs, float)
    return float(np.quantile(arr, a)), float(np.quantile(arr, 1.0 - a))


def fit_gumbel(x):
    """Gumbel-ML mit ξ = 0 fest, (μ, σ) frei. Rückgabe (μ, σ)."""
    x = np.asarray(x, float)
    loc = float(x.mean())
    scale = float(x.std(ddof=1))
    xs = (x - loc) / scale
    s0 = math.sqrt(6.0) / math.pi
    f = lambda th: nll((th[0], th[1], 0.0), xs)
    best, fb, _ = N.nelder_mead(f, [-EULER * s0, math.log(s0)], [0.3, 0.1], xatol=1e-9, fatol=1e-12)
    return loc + scale * best[0], scale * math.exp(best[1])

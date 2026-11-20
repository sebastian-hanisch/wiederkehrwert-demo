"""Kleine Zahlenwerkzeuge ohne scipy: Nelder-Mead, Brent-Wurzelsuche, Gamma-Verteilung mit ganzzahliger Form.

Die App rechnet nur mit numpy und der Standardbibliothek; scipy steht in `requirements-dev.txt` und wird nur in Tests als zweiter Rechenweg benutzt."""

import math


def nelder_mead(f, x0, steps, xatol=1e-8, fatol=1e-11, maxiter=2000, maxfev=4000):
    """Nelder-Mead (Standardkoeffizienten 1, 2, 1/2, 1/2) für eine Funktion f(Liste) -> float.

    Abbruch, wenn die Ecken des Simplex in jeder Koordinate höchstens `xatol` und die Funktionswerte höchstens `fatol` vom besten Punkt abweichen
    (wie scipy). Rückgabe: (bester Punkt als Liste, bester Wert, Zahl der Funktionsaufrufe)."""
    n = len(x0)
    sim = [list(map(float, x0))]
    for i in range(n):
        v = list(map(float, x0))
        v[i] += steps[i]
        sim.append(v)
    fs = [f(v) for v in sim]
    nfev = n + 1
    order = sorted(range(n + 1), key=fs.__getitem__)
    sim = [sim[i] for i in order]
    fs = [fs[i] for i in order]
    for _ in range(maxiter):
        if nfev >= maxfev:
            break
        best = sim[0]
        if max(abs(fs[i] - fs[0]) for i in range(1, n + 1)) <= fatol and max(abs(sim[i][j] - best[j]) for i in range(1, n + 1) for j in range(n)) <= xatol:
            break
        xbar = [sum(sim[i][j] for i in range(n)) / n for j in range(n)]
        worst = sim[n]
        xr = [xbar[j] + (xbar[j] - worst[j]) for j in range(n)]
        fr = f(xr)
        nfev += 1
        shrink = False
        if fr < fs[0]:
            xe = [xbar[j] + 2.0 * (xbar[j] - worst[j]) for j in range(n)]
            fe = f(xe)
            nfev += 1
            if fe < fr:
                sim[n], fs[n] = xe, fe
            else:
                sim[n], fs[n] = xr, fr
        elif fr < fs[n - 1]:
            sim[n], fs[n] = xr, fr
        else:
            if fr < fs[n]:
                xc = [xbar[j] + 0.5 * (xr[j] - xbar[j]) for j in range(n)]
                fc = f(xc)
                nfev += 1
                if fc <= fr:
                    sim[n], fs[n] = xc, fc
                else:
                    shrink = True
            else:
                xc = [xbar[j] - 0.5 * (xbar[j] - worst[j]) for j in range(n)]
                fc = f(xc)
                nfev += 1
                if fc < fs[n]:
                    sim[n], fs[n] = xc, fc
                else:
                    shrink = True
        if shrink:
            for i in range(1, n + 1):
                sim[i] = [sim[0][j] + 0.5 * (sim[i][j] - sim[0][j]) for j in range(n)]
                fs[i] = f(sim[i])
            nfev += n
        order = sorted(range(n + 1), key=fs.__getitem__)
        sim = [sim[i] for i in order]
        fs = [fs[i] for i in order]
    return sim[0], fs[0], nfev


def brent(f, a, b, xtol=1e-12, maxiter=200):
    """Nullstelle von f in [a, b] (Vorzeichenwechsel nötig), Brent-Verfahren."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("kein Vorzeichenwechsel")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * 2.220446049250313e-16 * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    return b


def gamma_cdf_int(m: int, x: float) -> float:
    """Verteilungsfunktion der Gamma(m, 1)-Verteilung für ganzzahliges m: 1 - exp(-x) * Summe_{j<m} x^j / j! (Poisson-Summe, in Logarithmen)."""
    if x <= 0:
        return 0.0
    lx = math.log(x)
    tail = 0.0
    for j in range(m):
        tail += math.exp(-x + j * lx - math.lgamma(j + 1))
    return max(0.0, min(1.0, 1.0 - tail))


def gamma_ppf_int(m: int, p: float) -> float:
    """Quantil der Gamma(m, 1)-Verteilung (ganzzahliges m) per Bisektion auf gamma_cdf_int."""
    lo, hi = 0.0, float(m) + 40.0 * math.sqrt(m) + 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gamma_cdf_int(m, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson-Intervall (zweiseitig 95 %) für einen Anteil k/n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return c - h, c + h

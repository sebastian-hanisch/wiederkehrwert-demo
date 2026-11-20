"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen nicht am Diagramm hängen bleibt)."""

import math

import plotly.graph_objects as go

import wkw_constants as C


def _fix(fig, titel_x, titel_y, hoehe=360, log_x=False, legende=True):
    fig.update_layout(
        height=hoehe, margin=dict(l=10, r=10, t=30, b=10), showlegend=legende, legend=dict(orientation="h", y=-0.28),
        xaxis=dict(title=titel_x, fixedrange=True, type="log" if log_x else "linear"), yaxis=dict(title=titel_y, fixedrange=True),
    )
    return fig


def build_wartezeit(kurve, T, N):
    """P(mindestens eine Überschreitung in n Jahren) gegen n, mit Marken bei n = N und n = T."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[k[0] for k in kurve], y=[100 * k[1] for k in kurve], mode="lines", name=f"T = {T}", line=dict(color=C.COLOR_WAHR, width=3)))
    marken = [(N, "Nutzungsdauer N")]
    if T <= C.N_MAX and T != N:
        marken.append((T, "n = T"))
    for n, name in marken:
        y = 100 * kurve[n - 1][1]
        fig.add_trace(go.Scatter(x=[n], y=[y], mode="markers+text", text=[f"{y:.1f} %".replace(".", ",")], textposition="top center", name=name, marker=dict(size=11, color=C.COLOR_SCHAETZ)))
    fig.add_hline(y=63.2, line_dash="dot", line_color="gray", annotation_text="1 − 1/e = 63,2 %", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 105])
    return _fix(fig, "Jahre n", "P(mindestens eine Überschreitung in n Jahren) in %")


def build_reihe_balken(p_m_exakt, k_m, R, wilson_m, m, T):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["exakt"], y=[100 * p_m_exakt], name="exakt", marker_color=C.COLOR_WAHR, text=[f"{100 * p_m_exakt:.1f} %".replace(".", ",")], textposition="outside"))
    p = k_m / R
    fig.add_trace(go.Bar(x=[f"gezogen ({R} Reihen)"], y=[100 * p], name="gezogen", marker_color=C.COLOR_HELL,
                         error_y=dict(type="data", symmetric=False, array=[100 * (wilson_m[1] - p)], arrayminus=[100 * (p - wilson_m[0])]),
                         text=[f"{100 * p:.1f} %".replace(".", ",")], textposition="outside"))
    fig.update_yaxes(range=[0, 115])
    return _fix(fig, f"Reihe von {m} Jahren erreicht den {T}-Jahres-Pegel", "Anteil der Reihen in %", legende=False)


def build_returnlevel(x, T_gitter, wahr, schaetz, T, x_wahr, z_hat, intervalle):
    """Pegel-Diagramm: Jahresmaxima nach Plotting-Position (Weibull), wahre Linie, angepasste Linie, Intervalle bei T."""
    m = len(x)
    xs = sorted(x)
    pos_T = [(m + 1) / (m + 1 - (i + 1)) for i in range(m)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos_T, y=xs, mode="markers", name=f"{m} Jahresmaxima", marker=dict(size=6, color=C.COLOR_HELL, line=dict(width=1, color="gray"))))
    fig.add_trace(go.Scatter(x=T_gitter, y=wahr, mode="lines", name="wahrer Pegel (exakt)", line=dict(color=C.COLOR_WAHR, width=3)))
    fig.add_trace(go.Scatter(x=T_gitter, y=schaetz, mode="lines", name="angepasste Linie", line=dict(color=C.COLOR_SCHAETZ, width=2, dash="dash")))
    farben = {"Delta": C.COLOR_DELTA, "Profil": C.COLOR_PROFIL, "Bootstrap": C.COLOR_BOOT}
    versatz = {"Delta": 0.85, "Profil": 1.0, "Bootstrap": 1.18}
    ymax = max(xs + [x_wahr, z_hat])
    for name, iv in intervalle.items():
        if iv is None:
            continue
        lo = iv[0] if math.isfinite(iv[0]) else min(xs)
        hi = iv[1] if math.isfinite(iv[1]) else ymax * 1.4
        fig.add_trace(go.Scatter(x=[T * versatz[name]] * 2, y=[lo, hi], mode="lines+markers", name=f"{name}-Intervall", line=dict(color=farben[name], width=4), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[T], y=[x_wahr], mode="markers", name="wahrer Pegel bei T", marker=dict(size=13, color=C.COLOR_WAHR, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[T], y=[z_hat], mode="markers", name="Schätzwert bei T", marker=dict(size=12, color=C.COLOR_SCHAETZ, symbol="x")))
    fig.update_xaxes(range=[math.log10(1.5), math.log10(max(T_gitter) * 1.3)])
    return _fix(fig, "Wiederkehrperiode T in Jahren (logarithmisch)", "Pegel", hoehe=440, log_x=True)


def build_streuband(v, q05, q50, q95, exakt=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[min(max(a, 0.0), 3.0) for a in v], xbins=dict(start=0, end=3, size=0.05), marker_color=C.COLOR_HELL, name="Läufe (Verhältnis > 3 bei 3 gezählt)"))
    for wert, name, farbe in ((1.0, "wahrer Pegel", C.COLOR_WAHR), (q05, "5 %", C.COLOR_SCHAETZ), (q50, "Median", C.COLOR_SCHAETZ), (q95, "95 %", C.COLOR_SCHAETZ)):
        fig.add_vline(x=min(wert, 3.0), line_color=farbe, line_dash="solid" if name == "wahrer Pegel" else "dash", annotation_text=name, annotation_position="top")
    if exakt is not None:
        for wert, name in ((exakt[0], "exakt 5 %"), (exakt[1], "exakt 95 %")):
            fig.add_vline(x=min(wert, 3.0), line_color=C.COLOR_PROFIL, line_dash="dot", annotation_text=name, annotation_position="bottom")
    fig.update_xaxes(range=[0, 3])
    return _fix(fig, "Schätzwert / wahrer Pegel", "Zahl der Läufe", legende=False)


def build_ueberdeckung(z, titel=None):
    """Gestapelte Balken je Intervallart: überdeckt, zu niedrig (wahrer Pegel darüber), zu hoch (wahrer Pegel darunter), kein Intervall. Nominell 95 % als Linie."""
    arten = [("delta", "Delta"), ("profil", "Profil"), ("bootstrap", "Perzentil-Bootstrap")]
    arten = [(k, n) for k, n in arten if k in z]
    fig = go.Figure()
    reihen = (("cov", "überdeckt", C.COLOR_WAHR), ("niedrig", "Intervall zu niedrig", C.COLOR_SCHAETZ), ("hoch", "Intervall zu hoch", "#7b3294"), ("ohne", "kein Intervall", "#999999"))
    for key, name, farbe in reihen:
        fig.add_trace(go.Bar(x=[n for _, n in arten], y=[100 * z[k][key] for k, _ in arten], name=name, marker_color=farbe,
                             text=[f"{100 * z[k][key]:.1f} %".replace(".", ",") if z[k][key] >= 0.0005 else "" for k, _ in arten], textposition="inside"))
    fig.add_hline(y=95, line_dash="dash", line_color="black", annotation_text="nominell 95 %", annotation_position="top left")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(range=[0, 100])
    return _fix(fig, titel or "Intervallart", "Anteil der Läufe in %")

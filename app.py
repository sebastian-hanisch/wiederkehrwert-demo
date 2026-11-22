"""Wiederkehrwert: was „100 Jahre“ heißt und wie sicher man ihn kennt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Demo der Linie „Extremwerte erklärt“: der Pegel (Wiederkehrwert) einer Reihe von Jahresmaxima, die Wiederkehrperiode als Mittel, das Streuband des geschätzten Pegels
und die Überdeckung seiner Intervalle. Alle Blöcke sind ausgedacht.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import wkw_constants as C
import wkw_evaluation as EV
import wkw_reference as REF
import wkw_returnlevel as RL
from wkw_evaluation import Settings, de, pct, tausend
from wkw_layout import dunkles_design, kopf_css
from wkw_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from wkw_scenario import FALL_NAMEN, wahrer_pegel
from wkw_visualization import build_returnlevel, build_reihe_balken, build_streuband, build_ueberdeckung, build_wartezeit

st.set_page_config(page_title="Wiederkehrwert – Sebastian Hanisch", layout="wide")


def _teil(s: Settings, *felder) -> Settings:
    """Nur die genannten Felder behalten, der Rest bleibt Standard: so hängt der Cache jedes Experiments nur an seinen eigenen Reglern."""
    return Settings(**{f: getattr(s, f) for f in felder})


@st.cache_data(show_spinner="Reihen ziehen (Experiment 1) …")
def _wartezeit(s):
    return EV.wartezeit_analyse(s)


@st.cache_data(show_spinner=False)
def _einzelreihe(s):
    return EV.einzelreihe(s)


@st.cache_data(show_spinner="Läufe rechnen (Experiment 3), je Lauf eine Anpassung …")
def _streuband(s):
    return EV.streuband_live(s)


@st.cache_data(show_spinner=False)
def _wahre_kurve(fall, xi, gitter):
    f = EV.fall_von(Settings(fall=fall, xi=xi))
    return [wahrer_pegel(f, T) for T in gitter]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wkw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌊 Wiederkehrwert: was „100 Jahre“ heißt und wie sicher man ihn kennt")
st.markdown(
    """
Die **Wiederkehrperiode** eines 100-Jahres-Werts ist ein **Mittel, keine Frist**: T sagt, wie oft ein Pegel im Mittel überschritten wird, nicht wann. Wird der Pegel aus
einer Reihe von Jahresmaxima **geschätzt**, ist er keine feste Zahl, sondern eine Zufallsgröße mit **Streuband**, und ein nominell 95-%-**Konfidenzintervall** hält sein Versprechen bei kurzen Reihen oft nicht. Diese Demo zeigt
das an ausgedachten Blöcken, mit **Zähleinheit** an jeder Zahl (Läufe, Reihen, Blöcke, Jahre; ein Block ist ein Jahr). Unten stehen vier Experimente, dann **🚧 Wo die Annahmen enden**
und der Expander **📐 Mathematische Formulierung**. Wer die Frage zuerst ohne Fachbegriffe verstehen will, öffnet „💬 In einfachen Worten“.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/wiederkehrwert-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Die Wiederkehrperiode eines Hochwassers sagt, wie oft ein Wasserstand im Mittel erreicht wird, nicht wann. Ein Wert, der im Mittel einmal in hundert Jahren kommt, bringt also keine hundert Jahre Ruhe, und auch eine Aufzeichnung von fünfzig Jahren erreicht ihn nur in einem Teil der Fälle. Die Demo rechnet das exakt vor. Danach schätzt sie den Wert aus kurzen, erfundenen Aufzeichnungen von Jahreshöchstständen, zählt, wie weit die Schätzung danebenliegt und nach welcher Seite, und prüft, ob ein Band, das „95 von 100 Fällen“ verspricht, sein Versprechen hält. Alles gilt für ausgedachte Jahre mit bekannten Eigenschaften, nicht für echte Messreihen.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/wiederkehrwert-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert der Pegel", expanded=True):
    st.markdown(
        """
1. **Pegel:** Der T-Jahres-Wert ist der Wert, den ein Jahresmaximum mit Wahrscheinlichkeit 1/T überschreitet. Er folgt aus den drei Parametern (μ, σ, ξ) der angepassten Extremwertverteilung (GEV).
2. **Wiederkehrperiode:** 1/T je Jahr heißt: im Mittel einmal in T Jahren, **nicht** „nach T Jahren“. In T Jahren wird der Pegel nur mit 1 − (1 − 1/T)^T überschritten (63,4 % bei T = 100).
3. **Unsicherheit:** Aus m Jahresmaxima geschätzt, streut der Pegel (Streuband); Delta, Profil-Likelihood und Bootstrap geben Intervalle, deren **Überdeckung** man nur über viele Wiederholungen messen kann.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Pegel und Reihe**")
    T = st.slider("Wiederkehrperiode T (Jahre)", *bounds("T_slider"), key="T_slider", help="Der Pegel wird mit Wahrscheinlichkeit 1/T je Jahr überschritten.")
    m = st.slider("Reihenlänge m (Jahre = Blöcke)", *bounds("m_slider"), key="m_slider", help="Wie viele Jahresmaxima die Reihe hat. Unter 3 ist keine Anpassung möglich, unter 5 sehr unsicher.")
    N = st.slider("Nutzungsdauer N (Jahre)", *bounds("N_slider"), key="N_slider", help="Experiment 1: in wie vielen Jahren man nach einer Überschreitung fragt.")
    st.markdown("**Woher die Jahresmaxima kommen**")
    fall = st.selectbox("Fall", C.FAELLE, key="fall_select", format_func=lambda k: FALL_NAMEN[k], help="Alle Fälle sind ausgedacht; bei den drei letzten kommt der wahre Pegel exakt aus F^n.")
    if fall == "gev":
        xi = st.select_slider("Formparameter ξ", options=C.XI_WERTE, key="xi_slider", help="ξ > 0: Fréchet-Typ (schwerer Tail), ξ = 0: Gumbel, ξ < 0: Weibull-Typ mit endlichem Rand.")
    else:
        xi = C.DEFAULT_XI
    st.markdown("**Experiment 1**")
    R1 = st.slider("Gezogene Reihen R", *bounds("R1_slider"), key="R1_slider", step=100, help=f"Läufe der Simulation; höchstens {tausend(C.SIM_LIMIT)} Zufallszahlen insgesamt (R wird bei langen Reihen gekürzt).")
    st.markdown("**Experiment 3**")
    R3 = st.slider("Läufe R (live)", *bounds("R3_slider"), key="R3_slider", step=10, help="Wie viele Reihen live gezogen und angepasst werden; die eingefrorene Tabelle hat 2 000 Läufe je Zelle.")
    if fall == "gev":
        bekannt = st.radio("Was ist bekannt?", C.BEKANNT_WERTE, key="bekannt_radio", help="Bei bekanntem σ und ξ wird nur μ geschätzt: dann ist der absolute Fehler (in Einheiten σ) für jedes T derselbe; das Verhältnis V hängt über x_T von T ab.")
    else:
        bekannt = C.BEKANNT_FREI
    st.markdown("**Experiment 4**")
    R4 = st.slider("Läufe der Probe", *bounds("R4_slider"), key="R4_slider", help="Wie viele Reihen die Probe live rechnet (Zähleinheit Läufe); die Überdeckung belegt erst die eingefrorene Tabelle.")
    delta_an = st.checkbox("Delta-Intervall", key="delta_check")
    profil_an = st.checkbox("Profil-Likelihood-Intervall", key="profil_check")
    boot_an = st.checkbox("Perzentil-Bootstrap (B = 199)", key="boot_check", help="Rechnet je Lauf 199 Anpassungen: ein Vielfaches der Zeit von Delta und Profil.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({k: st.session_state.get(k, SETTING_SPECS[k].default) for k in SETTING_SPECS})

s = Settings(fall, float(xi), int(T), int(m), int(N), int(R1), int(R3), bekannt, int(R4), bool(delta_an), bool(profil_an), bool(boot_an), int(seed))
fall_obj = EV.fall_von(s)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

wz = _wartezeit(_teil(s, "T", "N", "m", "R1", "seed"))
st.markdown("## 🎯 Ergebnis: was „T Jahre“ heißt")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Überschreitung in {s.N} Jahren", pct(wz["p_N"]), help="1 − (1 − 1/T)^N, exakt, unabhängige Jahre", delta=f"T = {s.T}", delta_color="off")
c2.metric("Median der Wartezeit", f"{wz['median']} Jahre", help="kleinstes k mit (1 − 1/T)^k < 1/2 (bei T = 2 sind 1 und 2 beide Medianwerte, hier 2); das Mittel der Wartezeit ist T", delta=f"Mittel {s.T} Jahre", delta_color="off")
c3.metric(f"Reihe von {s.m} Jahren erreicht ihn", pct(wz["p_m"]), help="1 − (1 − 1/T)^m, für jede stetige Verteilung unabhängiger, gleich verteilter Jahre (Anteil der Reihen)", delta=f"Reihen, T = {s.T}", delta_color="off")
c4.metric("Jahre Reihe für 90 %", tausend(wz["jahre90"]), help="kleinste Reihenlänge, bei der das Maximum den Pegel mit mindestens 90 % erreicht", delta="Jahre", delta_color="off")
st.plotly_chart(build_wartezeit(wz["kurve"], s.T, s.N), width="stretch", key="chart_wartezeit")
st.caption("Kurve: exakt, für unabhängige Jahre. Die Marke bei n = T liegt bei 63,4 % für T = 100, nicht bei 100 %.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Was heißt „100 Jahre“?**")
kind, text = EV.verdict_wartezeit(wz)
getattr(st, kind)(text)

st.markdown("**2. Wie weit liegt die Schätzung daneben, und nach welcher Seite?**")
sb_zelle = REF.finde_streuband(s.xi, s.m, s.T) if s.fall == "gev" else None
if sb_zelle is not None:
    kind, text = EV.verdict_streuband(sb_zelle, s.T, s.m, f"ξ = {de(s.xi, 1)}, GEV(100, 20, ξ)")
    getattr(st, kind)(text.replace(f"In {sb_zelle['R']} Läufen", f"Eingefroren: in {tausend(sb_zelle['R'])} Läufen", 1))
else:
    st.info("Für diese Einstellung steht kein eingefrorenes Streuband (eingefroren: GEV mit ξ = −0,2; 0; 0,2, m = 20; 50; 100; 200; 500, T = 10; 100; 1 000). Experiment 3 rechnet es live.")

st.markdown("**3. Hält das 95-%-Intervall?**")
ue_zelle = REF.finde_ueberdeckung(s.fall, s.xi, s.m, s.T)
if ue_zelle is not None:
    kind, text = EV.verdict_ueberdeckung(ue_zelle)
    getattr(st, kind)(text)
else:
    st.info("Für diese Einstellung steht keine eingefrorene Überdeckung (siehe Experiment 4 für die eingefrorenen Zellen). Die Probe in Experiment 4 rechnet wenige Läufe live.")

st.markdown("**4. Wie lang muss die Reihe sein?**")
sb100 = REF.finde_streuband(0.2, 100, 100)
kind, text = EV.verdict_reihenlaenge(s.T, wz["jahre90"], sb100 if s.T == 100 else None)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Was heißt „T Jahre“?")
st.caption(f"Zähleinheit: Reihen. Jede Reihe zieht Jahr für Jahr, ob der Pegel überschritten wird (Wahrscheinlichkeit 1/{s.T} je Jahr); bis zu {tausend(wz['R_sim'])} Reihen.")
c_a, c_b = st.columns(2)
with c_b:
    st.plotly_chart(build_reihe_balken(wz["p_m"], wz["k_m"], wz["R_sim"], wz["wilson_m"], s.m, s.T), width="stretch", key="chart_reihe")
with c_a:
    st.markdown(
        f"- **In {s.N} Jahren** (Nutzungsdauer): exakt {pct(wz['p_N'])}; gezogen {pct(wz['k_N'] / wz['R_sim'])} "
        f"(95-%-Band {pct(wz['wilson_N'][0])} bis {pct(wz['wilson_N'][1])}).\n"
        f"- **Reihe von {s.m} Jahren erreicht den Pegel:** exakt {pct(wz['p_m'])}, gezogen {pct(wz['k_m'] / wz['R_sim'])}.\n"
        f"- Die Zahl für N = {s.m} ist **dieselbe**: Reihenlänge und Nutzungsdauer sind dasselbe Zählen."
    )
kind, text = EV.verdict_simulation(wz)
getattr(st, kind)(text)
st.caption("Nebenbefund (Namensfalle): Wiederkehrperiode ist keine Frist. Mittel und Median der Wartezeit fallen auseinander (ab T = 3 liegt der Median darunter), weil die geometrische Verteilung rechtsschief ist.")

st.markdown("---")
st.subheader("🔬 Experiment 2: Eine Reihe, ein Pegel, drei Intervalle")
st.caption(f"Eine Reihe aus {s.m} Blöcken, Fall: {fall_obj.beschreibung()}. Maximum Likelihood (ξ ≥ −1), Delta, Profil-Likelihood und parametrischer Bootstrap (B = {C.BOOT_B}), alle nominell 95 %.")
if s.m < 5:
    st.warning(f"Mit m = {s.m} Blöcken ist die Anpassung sehr unsicher; unter m = 3 ist sie nicht möglich.")
with st.spinner(f"Anpassung, Profil und Bootstrap ({C.BOOT_B} Anpassungen) rechnen …"):
    er = _einzelreihe(_teil(s, "fall", "xi", "T", "m", "seed"))
if "fehler" in er:
    st.warning(er["fehler"])
else:
    iv = {"Delta": er["T"][s.T]["delta"] and er["T"][s.T]["delta"][:2], "Profil": er["T"][s.T]["profil"], "Bootstrap": er["T"][s.T]["boot"]}
    gitter = [2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    if s.T not in gitter:
        gitter = sorted(gitter + [s.T])
    wahr = _wahre_kurve(s.fall, s.xi, tuple(gitter))
    schaetz = [RL.gev_pegel(1.0 / t, er["mu"], er["sigma"], er["xi"]) for t in gitter]
    st.plotly_chart(build_returnlevel(er["x"], gitter, wahr, schaetz, s.T, er["x_wahr"], er["T"][s.T]["z"], iv), width="stretch", key="chart_returnlevel")
    kind, text = EV.verdict_einzelreihe(s.T, er["x_wahr"], er["T"][s.T]["z"], iv, s.m)
    getattr(st, kind)(text)
    st.caption(f"Angepasst: μ̂ = {de(er['mu'], 2)}, σ̂ = {de(er['sigma'], 2)}, ξ̂ = {de(er['xi'], 3)}" + ("" if er["ok"] else " (ξ̂ klebt an der Grenze −1: die Anpassung ist unbrauchbar)") + ". Die Zeitachse ist logarithmisch; Punkte: Jahresmaxima nach Weibull-Plotting-Position.")
    if s.fall != "gev":
        mf = REF.finde_ueberdeckung(s.fall, s.xi, 50, s.T)
        if mf is not None and "wahre_ueberschreitung" in mf:
            kind, text = EV.verdict_modellfehler_anker(mf)
            getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Viele Reihen - wie weit liegt die Schätzung daneben?")
st.caption(f"Zähleinheit: Läufe (je Lauf eine Reihe aus {s.m} Blöcken). V = Schätzwert / wahrer Pegel, live {s.R3} Läufe. Fall: {fall_obj.beschreibung()}.")
sbl = _streuband(_teil(s, "fall", "xi", "T", "m", "R3", "bekannt", "seed"))
if "fehler" in sbl:
    st.warning(sbl["fehler"])
else:
    exakt = None
    if sbl["bekannt"] and s.xi == 0.0:
        exakt = EV.exaktes_band_bekannt(s.m, s.T)
    st.plotly_chart(build_streuband(sbl["v"], sbl["band"]["q05"], sbl["band"]["q50"], sbl["band"]["q95"], exakt), width="stretch", key="chart_streuband")
    frozen = REF.finde_streuband(s.xi, s.m, s.T) if (s.fall == "gev" and not sbl["bekannt"]) else None
    kind, text = EV.verdict_streuband(sbl["band"], s.T, s.m, (f"ξ = {de(s.xi, 1)}" if s.fall == "gev" else fall_obj.beschreibung()) + (", σ und ξ bekannt" if sbl["bekannt"] else ""), frozen, sbl["bekannt"])
    getattr(st, kind)(text)
    if exakt is not None:
        st.caption(f"Exakt (σ bekannt, ξ = 0, Gamma-Verteilung): V liegt zu 90 % zwischen {de(exakt[0], 3)} und {de(exakt[1], 3)}, für jedes T derselbe Fehler in Einheiten σ.")
    if s.m < 50:
        st.caption("Ehrlicher Negativbefund: mehr Jahre helfen, aber das Band hängt nicht nur an T/m.")
    else:
        st.caption("Ehrlicher Negativbefund: mehr Jahre helfen, aber das Band hängt nicht nur an T/m (siehe die eingefrorenen Zellen in der README).")

st.markdown("---")
st.subheader("🔬 Experiment 4: Hält das Intervall? (Überdeckung)")
st.caption("Zähleinheit: Läufe. Überdeckung = Anteil der Läufe, in denen das Intervall den wahren Pegel enthält. „Zu niedrig“: der wahre Pegel liegt **über** dem Intervall, „zu hoch“: **darunter**.")
if ue_zelle is not None:
    st.plotly_chart(build_ueberdeckung(ue_zelle), width="stretch", key="chart_ueberdeckung")
    kind, text = EV.verdict_ueberdeckung(ue_zelle)
    getattr(st, kind)(text)
else:
    st.info("Diese Kombination ist nicht eingefroren. Eingefroren sind GEV mit ξ = −0,3; 0; 0,2 und m = 20; 50; 100 bei T = 100, dazu (ξ; m; T) = (0,2; 50; 10), (0,2; 50; 1 000), (0; 50; 1 000) und (0,2; 20; 1 000) "
            "sowie die drei Tageswert-Fälle bei m = 50 und T = 100 oder 1 000.")
st.markdown("**Probe (live)**")
sig = (s.fall, s.xi, s.T, s.m, s.R4, s.delta, s.profil, s.boot, s.seed)
if st.button("▶ Probe rechnen", key="probe_button", help="Rechnet R Läufe mit den gewählten Intervallarten; der Bootstrap kostet je Lauf ein Vielfaches."):
    if s.m < 3:
        st.session_state["probe"] = (sig, None)
    else:
        with st.spinner(f"{s.R4} Läufe rechnen (Delta, Profil{', Bootstrap' if s.boot else ''}) …"):
            st.session_state["probe"] = (sig, EV.probe_laeufe(s))
gespeichert = st.session_state.get("probe")
if gespeichert and gespeichert[0] == sig:
    if gespeichert[1] is None:
        st.warning("Mit weniger als drei Blöcken ist keine Anpassung möglich.")
    else:
        kind, text = EV.verdict_probe(gespeichert[1])
        getattr(st, kind)(text)
else:
    st.caption("Noch keine Probe für diese Einstellung. Der Knopf rechnet die Läufe; die Zufalls-Seeds der Läufe hängen am Zufalls-Seed der Seitenleiste.")
st.caption("Ehrlicher Negativbefund: auch das Profil-Intervall liegt bei kurzen Reihen unter 95 %, und es ist dort viel breiter (bei m = 20 das 1,9- bis 4,8-Fache von Delta). Nach oben offen war es in den eingefrorenen Zellen (m ≥ 20) in keinem Lauf, bei sehr kurzen Reihen kann es in einzelnen Läufen offen sein.")

st.markdown("---")
with st.expander("❓ Was hilft? Wie lang muss die Reihe sein?"):
    zeilen = EV.jahre_tabelle()
    df = pd.DataFrame([{"Sicherheit q": f"{int(q * 100)} %", **{f"T = {t}": tausend(j) for t, j in zip(C.T_STUFEN, jahre)}} for q, jahre in zeilen])
    st.markdown("Jahre Reihe, damit das größte der Jahre den T-Jahres-Wert mit Wahrscheinlichkeit q erreicht (exakt, für jede stetige Verteilung unabhängiger, gleich verteilter Jahre):")
    st.dataframe(df, hide_index=True, width="stretch")
    st.markdown(f"**Rand der Daten:** {EV.rand_der_daten_text(s.m)} Verwandt: die Demo `forecast-interval-demo` (konforme Prognoseintervalle).")
    bd = [z for z in (REF.finde_streuband(0.2, 50, T) for T in C.T_STUFEN) if z]
    if bd:
        st.markdown("90-%-Band von x̂/x_T für ξ = 0,2, m = 50 (eingefroren, 2 000 Läufe): " + "; ".join(f"T = {z['T']}: {de(z['q05'], 2)} bis {de(z['q95'], 2)}" for z in bd) + ".")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Blöcke sind ausgedacht** (GEV direkt, Exponential und Normal aus 365 Tageswerten, Vehikel B aus 100 Containern) | Es gibt keine reale Reihe, an der das Gezeigte geprüft ist; die Zahlen sagen nichts über einen echten Betrieb. | - |
| **Unabhängige, gleich verteilte Jahre** (iid, stationär) | Bei Abhängigkeit (θ < 1) ist die Wiederkehrperiode clusterweise zu lesen, bei Trend gilt „Risiko in n Jahren ≠ Wiederkehrwert“. Hier nicht gemessen. | - |
| **Maximum Likelihood mit ξ ≥ −1** | Für ξ < −1 ist die Likelihood unbeschränkt (darum die Nebenbedingung). Klebt ξ̂ an dieser Grenze, ist die Anpassung unbrauchbar; die App meldet es unter dem Diagramm von Experiment 2. | - |
| **Drei Intervallvarianten** (Delta mit beobachteter Information, Profil, parametrischer Perzentil-Bootstrap B = 199) | „Delta unterdeckt“ gilt für diese Delta-Variante, nicht für die Methodenklasse; BCa, Bootstrap-t und nichtparametrischer Bootstrap sind nicht gemessen. | - |
| **Blöcke gelten als bekannt** (Blockgröße ein Jahr) | Die Wahl der Blockgröße ist hier nicht Thema. | - |
| **Intervalle für Pegel aus Schwellenmodellen** | Nicht gemessen (nur Blockweg). | - |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/wiederkehrwert-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $m$ unabhängige Jahresmaxima $x_1,\dots,x_m$ (ein Block ist ein Jahr). Gesucht ist der Pegel $x_T$ zur Überschreitungswahrscheinlichkeit $p=1/T$ je Jahr und ein Intervall dafür.

| Symbol | Bedeutung |
|---|---|
| $\mu,\ \sigma,\ \xi$ | Ort, Skala und Form der Extremwertverteilung (GEV); $\xi>0$ schwerer Tail, $\xi=0$ Gumbel, $\xi<0$ endlicher Rand |
| $T,\ p=1/T$ | Wiederkehrperiode (Mittel der Wartezeit in Jahren) und Überschreitungswahrscheinlichkeit je Jahr |
| $x_T$ | Pegel (Wiederkehrwert): der Wert, den ein Jahresmaximum mit Wahrscheinlichkeit $p$ überschreitet |
| $y_p$ | Hilfsgröße $-\ln(1-p)$ |
| $m,\ N$ | Länge der Aufzeichnung in Jahren, Nutzungsdauer in Jahren |
| $W,\ n$ | Wartezeit in Jahren bis zur ersten Überschreitung (geometrisch verteilt) und eine Zahl von Jahren |
| $\ell(\theta)$ | Log-Likelihood, $\theta=(\mu,\ln\sigma,\xi)$ |
| $H,\ D(z)$ | beobachtete Information und Devianz des Profil-Intervalls |
| $G$ | Summe der Größen $e^{-(x_i-\mu)/\sigma}$, im Ein-Parameter-Fall Gamma$(m,1)$-verteilt |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/wiederkehrwert-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Gegeben:** $m$ unabhängige Jahresmaxima $x_1,\dots,x_m$ (ein Block = ein Jahr). **Gesucht:** der Pegel $x_T$ zur Überschreitungswahrscheinlichkeit $p=1/T$ je Jahr und ein Intervall dafür.

| Symbol | Bedeutung |
|---|---|
| $\mu,\ \sigma>0,\ \xi$ | Ort, Skala und Form der GEV (ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ mit endlichem Rand) |
| $T,\ p=1/T$ | Wiederkehrperiode (Mittel der Wartezeit) und Überschreitungswahrscheinlichkeit je Jahr |
| $x_T$ | Pegel (Wiederkehrwert, englisch Return Level): $G(x_T)=1-1/T$ |
| $m,\ N$ | Länge der Reihe in Jahren, Nutzungsdauer in Jahren |
| $\ell(\theta)$ | Log-Likelihood, $\theta=(\mu,\ln\sigma,\xi)$ |

**Pegel (Coles Gl. 3.4):** mit $y_p=-\ln(1-p)$ ist $x_T=\mu+\sigma\,\dfrac{y_p^{-\xi}-1}{\xi}$ und für $\xi=0$: $x_T=\mu-\sigma\ln y_p$. (`scipy.stats.genextreme` hat $c=-\xi$.)

**Wartezeit:** Jahre sind unabhängig mit Überschreitungswahrscheinlichkeit $p=1/T$. Dann ist die Wartezeit $W$ geometrisch, $P(W\le n)=1-(1-1/T)^n$, $E[W]=T$, und das größte von $m$ Jahren erreicht den Pegel mit $1-(1-1/T)^m$, für jede stetige Verteilung der Jahre (unabhängige, gleich verteilte Jahre). Für $T=100$, $n=100$ sind das $63{,}4\ \%$ (Grenzwert $1-1/e=63{,}2\ \%$).

**Delta-Intervall (Gl. 3.11):** $\hat x_T\pm 1{,}96\sqrt{\nabla x_T^\top\, H^{-1}\,\nabla x_T}$ mit der beobachteten Information $H=-\nabla^2\ell(\hat\theta)$ (numerisch).

**Profil-Likelihood-Intervall (Gl. 3.12):** alle $z$, für die $D(z)=2\bigl(\ell(\hat\theta)-\max_{\sigma,\xi}\ell(\theta\mid x_T=z)\bigr)\le\chi^2_{1;0{,}95}=3{,}8415$ gilt; dabei folgt $\mu$ aus $z$, $\sigma$ und $\xi$.

**Perzentil-Bootstrap:** $B=199$ Anpassungen an Stichproben aus der angepassten GEV; das Intervall sind das 2,5- und das 97,5-%-Quantil der $B$ Pegel (keine BCa-Korrektur).

**Ein Parameter bekannt (Gumbel, σ bekannt, ξ = 0):** $e^{-(x_i-\mu)/\sigma}\sim\mathrm{Exp}(1)$, also $\hat\mu-\mu=-\sigma\ln(G/m)$ mit $G\sim\mathrm{Gamma}(m,1)$, und der Fehler jedes Pegels ist dieser eine Fehler. Das Delta-Intervall $\hat\mu\pm1{,}96\,\sigma/\sqrt m$ deckt $\mu$ genau dann, wenn $G/m\in[e^{-c},e^{c}]$ mit $c=1{,}96/\sqrt m$.

**Das Modell ist vollständig für diese Demo:** die Blöcke sind unabhängig und gleich verteilt, die Anpassung ist Maximum Likelihood mit $\xi\ge-1$, die Intervalle sind die drei genannten Varianten. **Nicht enthalten:** abhängige Blöcke, Trend, Schwellenmodelle, andere Niveaus als 95 %.

**Literatur** (Stufen siehe README): Coles 2001; Embrechts, Klüppelberg, Mikosch 1997, 6.2.4 und 8.1.5.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)

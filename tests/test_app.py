"""End-to-end Smoke-Test über Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jedes Preset, fährt jeden Regler an seine Grenzen, wechselt Fälle und Schalter,
rechnet die Probe und prüft, dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId` bei mehreren `st.plotly_chart` und `StreamlitAPIException` bei Reglern mit
berechneten Grenzen). AppTest sieht keine Skriptfehler im Browser; die Browser-Prüfung steht in der README."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import wkw_constants as C
import wkw_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _frisch(**werte) -> AppTest:
    at = AppTest.from_file(APP_PATH)
    for k, v in werte.items():
        at.session_state[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _alle_text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [i.value for i in at.info] + [w.value for w in at.warning] + [s.value for s in at.success])


def test_app_laedt_ohne_fehler_und_zeigt_die_kernfragen():
    at = _frisch()
    text = _alle_text(at)
    for frage in ("1. Was heißt „100 Jahre“?", "2. Wie weit liegt die Schätzung daneben", "3. Hält das 95-%-Intervall?", "4. Wie lang muss die Reihe sein?"):
        assert frage in text
    assert "63,4 %" in text and "69 Jahre" in text and "230 Jahre" in text            # Anker der Kernfrage 1 und 4 bei den Standardwerten
    assert [s.value for s in at.subheader if s.value.startswith("🔬")] == [
        "🔬 Experiment 1: Was heißt „T Jahre“?", "🔬 Experiment 2: Eine Reihe, ein Pegel, drei Intervalle", "🔬 Experiment 3: Viele Reihen - wie weit liegt die Schätzung daneben?",
        "🔬 Experiment 4: Hält das Intervall? (Überdeckung)"]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_durch(name):
    at = _frisch()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, (name, [str(e) for e in at.exception])
    assert len(at.get("plotly_chart")) >= 4


@pytest.mark.parametrize("key", ["T_slider", "m_slider", "N_slider", "R1_slider", "R3_slider", "R4_slider"])
def test_jeder_slider_an_min_und_max(key):
    spec = P.SETTING_SPECS[key]
    for wert in (spec.lo, spec.hi):
        at = _frisch()
        at.sidebar.slider(key=key).set_value(int(wert)).run(timeout=TIMEOUT)
        assert not at.exception, (key, wert, [str(e) for e in at.exception])


@pytest.mark.parametrize("m", [1, 2, 3, 4, 500])
def test_reihenlaenge_an_den_raendern(m):
    at = _frisch(m_slider=m)
    assert not at.exception
    text = _alle_text(at)
    if m < 3:
        assert "keine Anpassung" in text and "mindestens drei" in text
    if m < 5:
        assert "sehr unsicher" in text


@pytest.mark.parametrize("fall", C.FAELLE)
def test_jeder_fall_laeuft(fall):
    at = _frisch(fall_select=fall)
    assert not at.exception
    xi_da = len(at.sidebar.select_slider) == 1                                       # ξ und „Was ist bekannt?“ gibt es nur bei GEV direkt
    assert xi_da == (fall == "gev") and (len(at.sidebar.radio) == 1) == (fall == "gev")


@pytest.mark.parametrize("xi", [C.XI_WERTE[0], 0.0, C.XI_WERTE[-1]])
def test_xi_raender(xi):
    at = _frisch(xi_slider=xi)
    assert not at.exception


def test_wechsel_von_gev_zu_normal_und_zurueck_behaelt_einen_gueltigen_zustand():
    at = _frisch()
    at.sidebar.selectbox(key="fall_select").set_value("normal365").run(timeout=TIMEOUT)
    assert not at.exception and not at.sidebar.radio
    at.sidebar.selectbox(key="fall_select").set_value("gev").run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.radio(key="bekannt_radio").value == C.BEKANNT_FREI and at.sidebar.select_slider(key="xi_slider").value == C.DEFAULT_XI


def test_sigma_bekannt_zeigt_die_exakte_gamma_band_zeile():
    at = _frisch(xi_slider=0.0, bekannt_radio=C.BEKANNT_SIGMA, m_slider=20, T_slider=1000)
    assert not at.exception
    assert "Exakt (σ bekannt, ξ = 0, Gamma-Verteilung)" in " ".join(c.value for c in at.caption)


def test_probe_knopf_rechnet_und_zeigt_ein_ergebnis():
    at = _frisch(R4_slider=5, m_slider=20)
    at.button(key="probe_button").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert "Probe mit 5 Läufen" in _alle_text(at)
    at.sidebar.slider(key="R4_slider").set_value(6).run(timeout=TIMEOUT)           # andere Einstellung: die alte Probe wird nicht mehr gezeigt
    assert "Probe mit 5 Läufen" not in _alle_text(at) and "Noch keine Probe" in " ".join(c.value for c in at.caption)


def test_probe_mit_bootstrap_und_ohne_intervallart():
    at = _frisch(R4_slider=5, m_slider=20, boot_check=True)
    at.button(key="probe_button").click().run(timeout=TIMEOUT)
    assert not at.exception and "Bootstrap" in _alle_text(at)
    at = _frisch(R4_slider=5, delta_check=False, profil_check=False)
    at.button(key="probe_button").click().run(timeout=TIMEOUT)
    assert not at.exception and "Keine Intervallart gewählt." in _alle_text(at)


def test_probe_mit_zu_kurzer_reihe_meldet_statt_zu_rechnen():
    at = _frisch(m_slider=2, R4_slider=5)
    at.button(key="probe_button").click().run(timeout=TIMEOUT)
    assert not at.exception and "weniger als drei Blöcken" in _alle_text(at)


def test_permalink_wird_gelesen_und_begrenzt():
    at = AppTest.from_file(APP_PATH)
    at.query_params["T"] = "5000"            # über der Grenze
    at.query_params["m"] = "20"
    at.query_params["fall"] = "ungueltig"    # ungültiger Auswahlwert wird ignoriert
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.session_state["T_slider"] == C.T_MAX and at.session_state["m_slider"] == 20 and at.session_state["fall_select"] == C.DEFAULT_FALL

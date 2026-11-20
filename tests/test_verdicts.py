"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import math

import wkw_evaluation as ev


def _analyse(T=100, N=100, m=50):
    import wkw_returnlevel as RL
    from fractions import Fraction
    return {"T": T, "N": N, "m": m, "p_N": RL.p_mindestens_einmal(T, N), "p_T": RL.p_mindestens_einmal(T, T), "median": RL.median_wartezeit(T), "p_m": RL.p_mindestens_einmal(T, m),
            "p_m_andere": {t: RL.p_mindestens_einmal(t, m) for t in (100, 1000)}, "jahre90": RL.jahre_fuer_sicherheit(T, Fraction(9, 10))}


def test_formatierung_deutsch():
    assert ev.de(-0.605, 3) == "−0,605" and ev.de(3.14159, 2) == "3,14" and ev.pct(0.6340, 1) == "63,4 %" and ev.pct(0.5, 0) == "50 %" and ev.tausend(2302) == "2 302"


def test_verdikt_wartezeit_nennt_die_anker_zahlen():
    kind, text = ev.verdict_wartezeit(_analyse())
    assert kind == "info"
    for teil in ("**T = 100:**", "in 100 Jahren", "**63,4 %**", "Grenzwert für große T: 63,2 %", "**69 Jahre**", "das Mittel 100", "Reihe von 50 Jahren", "**39,5 %**", "1000-Jahres-Wert: 4,9 %", "**230 Jahre**"):
        assert teil in text, teil


def test_verdikt_wartezeit_bei_T_1000_vergleicht_mit_dem_100er_wert():
    kind, text = ev.verdict_wartezeit(_analyse(T=1000, N=50, m=50))
    assert "**T = 1000:**" in text and "in 50 Jahren" in text and "(in 1000 Jahren: 63,2 %, Grenzwert" in text and "100-Jahres-Wert: 39,5 %" in text and "**693 Jahre**" in text and "**2 302 Jahre**" in text


def test_verdikt_simulation_schwelle_bei_drei_standardfehlern():
    a = {"R_sim": 10000, "k_m": 3950, "p_m": 0.395, "se_m": 0.004885, "m": 50}
    kind, text = ev.verdict_simulation(a)
    assert kind == "success" and "39,5 %" in text and "10 000 Reihen" in text
    kind, text = ev.verdict_simulation(dict(a, k_m=4000))                              # Abstand 0,005 / 0,004885 = 1,02 Standardfehler
    assert kind == "success"
    kind, text = ev.verdict_simulation(dict(a, k_m=4100))                              # 0,015 / 0,004885 = 3,07 Standardfehler
    assert kind == "warning" and "3,1 Standardfehler" in text


def test_verdikt_einzelreihe_alle_drin_oder_verfehlt():
    iv = {"Delta": (90.0, 110.0), "Profil": (88.0, 120.0), "Bootstrap": (91.0, 118.0)}
    kind, text = ev.verdict_einzelreihe(100, 100.0, 99.0, iv, 50)
    assert kind == "success" and text.count("enthält den wahren Pegel") == 3 and "Verhältnis 0,99" in text and "Wahrer Pegel (exakt) 100,0" in text and "eine einzige Reihe" in text
    iv["Delta"] = (50.0, 90.0)                                                          # wahrer Pegel liegt darüber: Intervall zu niedrig
    kind, text = ev.verdict_einzelreihe(100, 100.0, 70.0, iv, 50)
    assert kind == "warning" and "liegt zu niedrig (der wahre Pegel liegt darüber)" in text
    kind, text = ev.verdict_einzelreihe(100, 100.0, 140.0, {"Delta": (120.0, 150.0)}, 50)
    assert kind == "warning" and "liegt zu hoch (der wahre Pegel liegt darunter)" in text
    kind, text = ev.verdict_einzelreihe(100, 100.0, 99.0, {"Delta": None, "Profil": (90.0, math.inf)}, 50)
    assert kind == "warning" and "nicht berechenbar" in text and "[90,0; offen]" in text


def test_lage_des_pegels_an_den_grenzen_inklusive():
    assert ev.lage_des_pegels((1.0, 2.0), 1.0) == "drin" and ev.lage_des_pegels((1.0, 2.0), 2.0) == "drin"
    assert ev.lage_des_pegels((1.0, 2.0), 0.999) == "unter" and ev.lage_des_pegels((1.0, 2.0), 2.001) == "ueber"
    assert ev.lage_des_pegels(None, 1.0) == "ohne" and ev.lage_des_pegels((math.nan, 1.0), 1.0) == "ohne"


def test_verdikt_streuband_nennt_band_median_und_anteile():
    band = {"R": 100, "q05": 0.7349, "q50": 0.99, "q95": 1.5021, "unter_1": 0.52, "ueber_2": 0.04, "unter_08": 0.12}
    kind, text = ev.verdict_streuband(band, 100, 50, "ξ = 0,2")
    assert kind == "info" and "In 100 Läufen (ξ = 0,2, m = 50, T = 100)" in text and "**0,73-** und **1,50-Fachen**" in text and "Median 0,99" in text and "52 % der Läufe darunter" in text and "4 % mehr als doppelt" in text
    frozen = {"R": 2000, "q05": 0.74, "q95": 1.50, "ueber_2": 0.003, "unter_1": 0.513}
    kind, text = ev.verdict_streuband(band, 100, 50, "ξ = 0,2", frozen, bekannt=True)
    assert "Eingefroren (2 000 Läufe, anderer Zufalls-Seed): 0,74 bis 1,50" in text and "Bei bekanntem σ und ξ ist der absolute Fehler (in Einheiten σ) für jedes T derselbe; das Verhältnis V hängt über x_T von T ab." in text


def test_verdikt_ueberdeckung_schwelle_und_zahlen():
    z = {"fall": "gev", "xi": 0.2, "m": 50, "T": 100, "R": 1000,
         "delta": {"cov": 0.869, "se": 0.011, "hoch": 0.0, "niedrig": 0.131, "ohne": 0.0, "offen_oben": 0.0, "breite_median": 11.0},
         "profil": {"cov": 0.952, "se": 0.007, "hoch": 0.022, "niedrig": 0.026, "ohne": 0.0, "offen_oben": 0.0, "breite_median": 12.0},
         "bootstrap": {"cov": 0.908, "se": 0.009, "hoch": 0.008, "niedrig": 0.084, "ohne": 0.0, "offen_oben": 0.0, "breite_median": 13.0, "B": 199}}
    kind, text = ev.verdict_ueberdeckung(z)
    assert kind == "warning"                                                            # Delta 86,9 % liegt unter 93 %
    for teil in ("ξ = 0,2, m = 50, T = 100", "Delta 86,9 %", "Perzentil-Bootstrap (B = 199) 90,8 %", "Profil 95,2 %", "R = 1 000 Läufe", "13,1 % zu niedrig, 0,0 % zu hoch", "2,6 % zu niedrig und 2,2 % zu hoch"):
        assert teil in text, teil
    z["delta"]["cov"], z["profil"]["cov"] = 0.93, 0.95                                  # beide mindestens 0,93: nur info
    assert ev.verdict_ueberdeckung(z)[0] == "info"
    z["profil"]["cov"] = 0.9299
    assert ev.verdict_ueberdeckung(z)[0] == "warning"
    z["profil"]["offen_oben"] = 0.091
    assert "Profil-Intervalle ohne Obergrenze: 9,1 %" in ev.verdict_ueberdeckung(z)[1]


def test_verdikt_modellfehler_anker():
    z = {"R": 1000, "m": 50, "T": 100, "x_wahr": 10.5003, "wahre_ueberschreitung": {"anteil_ueber_2": 0.303, "anteil_unter_halb": 0.226, "v_median": 0.97}}
    kind, text = ev.verdict_modellfehler_anker(z)
    assert kind == "info" and "10,500" in text and "30 %" in text and "23 %" in text and "0,97" in text and "m = 50, T = 100" in text


def test_verdikt_probe_zeigt_k_von_R_und_band():
    probe = {"R": 10, "arten": {"delta": {"R": 10, "cov": 0.8, "wilson": (0.49, 0.94)}, "profil": {"R": 10, "cov": 1.0, "wilson": (0.72, 1.0)}}}
    kind, text = ev.verdict_probe(probe)
    assert kind == "info" and "Delta 8 von 10 (80 %; 95-%-Band 49 % bis 94 %)" in text and "Profil 10 von 10 (100 %" in text and "belegt die Überdeckung aber nicht" in text


def test_verdikt_reihenlaenge_mit_und_ohne_band():
    kind, text = ev.verdict_reihenlaenge(100, 230, None)
    assert kind == "info" and "**230 Jahre**" in text and "eingefroren" not in text
    band = {"m": 100, "xi": 0.2, "T": 100, "R": 2000, "q05": 0.801, "q95": 1.282}
    text = ev.verdict_reihenlaenge(100, 230, band)[1]
    assert "Mit 100 Jahren liegt die Schätzung (ξ = 0,2, T = 100, eingefroren, 2 000 Läufe) zu 90 % zwischen 0,80 und 1,28" in text


def test_rand_der_daten_text_und_tabelle():
    text = ev.rand_der_daten_text(50)
    assert "1/51 = 1,96 %" in text and "jenseits dessen, was die Rangordnung der Daten auflöst" in text
    assert "Erst ab T = 73 (das 1,46-Fache von m)" in text and "über dem größten Wert der Reihe" in text and "Extrapolation" not in text
    zeilen = ev.jahre_tabelle()
    assert [q for q, _ in zeilen] == [0.5, 0.9, 0.95, 0.99]
    assert dict(zeilen)[0.9] == [22, 230, 2302] and dict(zeilen)[0.5] == [7, 69, 693] and dict(zeilen)[0.99] == [44, 459, 4603] and dict(zeilen)[0.95] == [29, 299, 2995]

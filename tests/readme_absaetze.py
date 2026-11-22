"""Absätze der README mit Aussagen über viele Zellen, aus den Daten gezählt (Zählung und Ausnahme stehen im selben Satz).

Wie die Tabellen in `readme_tabellen.py` werden sie von der README wörtlich übernommen und von `tests/test_claims.py` wörtlich verlangt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import wkw_returnlevel as RL  # noqa: E402
from readme_tabellen import _cov  # noqa: E402
from wkw_evaluation import de, pct, tausend  # noqa: E402


def _zellen():
    c = _cov()
    return c["ueberdeckung"]["zellen"], c["streuband"]["zellen"], c["modellfehler"]["zellen"], c["ueberdeckung"]["nachmessung"]


def absatz_streuband():
    _, sb, _, _ = _zellen()
    z = {c["id"]: c for c in sb}
    unter = [c["unter_1"] for c in sb]
    med_ueber = sorted(c["id"] for c in sb if c["q50"] >= 1)
    schief = {xi: sum(c["q95"] - 1 > 1 - c["q05"] for c in sb if c["xi"] == xi) for xi in (-0.2, 0.0, 0.2)}
    a, b = z["xi+0.2_m50_T100"], z["xi+0.2_m500_T1000"]
    g = z["xi+0.0_m50_T100"]["robuste_streuung_sigma"]
    g10 = z["xi+0.0_m50_T10"]["robuste_streuung_sigma"]
    g1000 = z["xi+0.0_m50_T1000"]["robuste_streuung_sigma"]
    sk = _cov()["streuband"]["sigma_bekannt_robuste_streuung"]["werte"]
    lo, hi = RL.gumbel_fehler_band(50, 0.25, 0.75)
    assert med_ueber == ["xi+0.2_m100_T100", "xi+0.2_m100_T1000"]
    ausnahmen = "ξ = 0,2, m = 100 bei T = 100 und T = 1000"
    a1000 = z["xi+0.2_m50_T1000"]
    return (
        f"- **Unterschätzung:** der Anteil der Läufe, die unterschätzen (V < 1), liegt in allen 45 Zellen zwischen {pct(min(unter))} und {pct(max(unter))} (Standardfehler 1,1 Punkte); "
        f"in {sum(c['q50'] < 1 for c in sb)} von 45 Zellen liegt der Median von V unter 1; in den übrigen zwei ({ausnahmen}) liegt er bei {de(z['xi+0.2_m100_T100']['q50'], 2)} und {de(z['xi+0.2_m100_T1000']['q50'], 2)} und ist von 1 nicht zu unterscheiden. "
        f"Die Zählung und die Namen der zwei Ausnahmen hängen vom Zufalls-Seed; der kleinste Anteil ({pct(min(unter))}) liegt etwa 2 Standardfehler unter 50 %.\n"
        f"- **Rechtsschief** (q95 − 1 größer als 1 − q05): bei ξ = 0,2 in {schief[0.2]} von 15 Zellen, bei ξ = 0 in {schief[0.0]} von 15 (Ausnahmen: T = 10 ab m = 50), bei ξ = −0,2 in {schief[-0.2]} von 15. "
        f"Das Band liegt also nur bei ξ ≥ 0 meist weiter oberhalb als unterhalb des wahren Pegels.\n"
        f"- **Zu 90 % zwischen:** ξ = 0,2, m = 50: der 100er-Wert zwischen dem {de(a['q05'], 2)}- und {de(a['q95'], 2)}-Fachen, der 1000er-Wert zwischen {de(a1000['q05'], 2)} und {de(a1000['q95'], 2)}, "
        f"in {pct(a1000['ueber_2'])} der Läufe mehr als doppelt so hoch und in {pct(a1000['unter_08'])} unter dem 0,8-Fachen (T = 1000); bei m = 500 sind es {de(z['xi+0.2_m500_T100']['q05'], 2)} bis {de(z['xi+0.2_m500_T100']['q95'], 2)} (T = 100).\n"
        f"- **Das Band hängt nicht nur an T/m:** bei T/m = 2 ist q95/q05 für m = 50, T = 100 gleich {de(a['q95'] / a['q05'], 2)}, für m = 500, T = 1000 gleich {de(b['q95'] / b['q05'], 2)} (ξ = 0,2).\n"
        f"- **Die Extrapolation kostet wegen der geschätzten Parameter** (ξ = 0, m = 50, Streuung des Fehlers in Einheiten σ, robust als Interquartilsabstand/1,349, 2 000 Läufe): für T = 10 / 100 / 1000 "
        f"GEV mit drei freien Parametern {de(g10['gev_frei'], 3)} / {de(g['gev_frei'], 3)} / {de(g1000['gev_frei'], 3)}, Gumbel mit freiem (μ, σ) {de(g10['gumbel_mu_sigma_frei'], 3)} / {de(g['gumbel_mu_sigma_frei'], 3)} / {de(g1000['gumbel_mu_sigma_frei'], 3)}, "
        f"σ bekannt {de(sk['10'], 3)} / {de(sk['100'], 3)} / {de(sk['1000'], 3)} (exakt {de((hi - lo) / 1.349, 3)}). Streuungen aus den Läufen mit ξ = 0,2 und m = 20 werden von Ausreißern bestimmt, darum steht oben das Quantilband."
    )


def absatz_ueberdeckung():
    ue, _, _, nm = _zellen()
    z = {c["id"]: c for c in ue}
    d = [c["delta"] for c in ue]
    p = [c["profil"] for c in ue]
    bo = [c for c in ue if "bootstrap" in c]
    schwach = sorted((c for c in ue if c["profil"]["cov"] < 0.93), key=lambda c: c["profil"]["cov"])
    r20 = sorted(c["profil"]["breite_median"] / c["delta"]["breite_median"] for c in ue if c["m"] == 20)
    r100 = sorted(c["profil"]["breite_median"] / c["delta"]["breite_median"] for c in ue if c["m"] == 100)
    ab = [c["profil"]["cov"] - c["bootstrap"]["cov"] for c in bo]
    haupt = z["xi+0.2_m50_T100"]
    zk = z["xi-0.3_m20_T100"]
    breit = z["xi+0.2_m20_T1000"]
    se95 = (0.95 * 0.05 / ue[0]["R"]) ** 0.5
    deutlich = [c for c in ue if c["profil"]["cov"] < 0.95 - 2 * se95]
    assert len({c["R"] for c in ue}) == 1 and all(c["m"] <= 50 for c in deutlich) and all(c["m"] == 20 for c in schwach)

    def zt(c):
        return f"ξ = {de(c['xi'], 1)}, m = {c['m']}, T = {c['T']}"
    return (
        f"- **Delta unterdeckt in {sum(x['cov'] < 0.95 for x in d)} von {len(d)} Zellen** ({pct(min(x['cov'] for x in d))} bis {pct(max(x['cov'] for x in d))}; diese Delta-Variante, diese Anpassung), "
        f"und fast alle Fehlgriffe liegen **zu niedrig**: zu niedrig {pct(min(x['niedrig'] for x in d))} bis {pct(max(x['niedrig'] for x in d))}, zu hoch höchstens {pct(max(x['hoch'] for x in d))} der Läufe.\n"
        f"- **Profil** liegt in {len(ue)} von {len(ue)} Zellen zwischen {pct(min(x['cov'] for x in p))} und {pct(max(x['cov'] for x in p))} (Standardfehler {de(100 * min(x['se'] for x in p), 1)} bis {de(100 * max(x['se'] for x in p), 1)} Punkte); "
        f"unter 93 % liegen {len(schwach)} Zellen ({'; '.join(zt(c) + ': ' + pct(c['profil']['cov']) for c in schwach)}), beide bei m = 20, und diese Zahl hängt am Zufalls-Seed. "
        f"Mehr als 2 Standardfehler unter 95 % (gegen 95 % gerechnet: Standardfehler {de(100 * se95, 1)} Punkte) liegen {len(deutlich)} von {len(ue)} Zellen, alle bei m ≤ 50; die übrigen {len(ue) - len(deutlich)} sind mit 95 % verträglich. "
        f"Es verfehlt auf **beiden** Seiten (zu niedrig {pct(min(x['niedrig'] for x in p))} bis {pct(max(x['niedrig'] for x in p))}, zu hoch {pct(min(x['hoch'] for x in p))} bis {pct(max(x['hoch'] for x in p))}).\n"
        f"- **Die schwächste Profil-Zelle** (ξ = −0,3, m = 20, T = 100) schwankt mit den Läufen: {pct(zk['profil']['cov'])} bei 1 000 Läufen, {pct(nm['profil']['cov'])} bei {tausend(nm['R'])} frischen Läufen "
        f"(Standardfehler {de(100 * nm['profil']['se'], 1)} Punkte); Delta liegt dort bei {pct(zk['delta']['cov'])} (R = 1 000) und {pct(nm['delta']['cov'])} (R = 3 000).\n"
        f"- **Perzentil-Bootstrap** (B = 199, diese Variante): {pct(min(c['bootstrap']['cov'] for c in bo))} bis {pct(max(c['bootstrap']['cov'] for c in bo))} in {len(bo)} Zellen, **unter dem Profil in {sum(a > 0 for a in ab)} von {len(bo)} Zellen** "
        f"(kleinster Abstand {de(100 * min(ab), 1)} Punkte, größter {de(100 * max(ab), 1)}); Hauptzelle ({zt(haupt)}): {pct(haupt['bootstrap']['cov'])}. Auch er verfehlt überwiegend nach unten: "
        f"zu niedrig {pct(min(c['bootstrap']['niedrig'] for c in bo))} bis {pct(max(c['bootstrap']['niedrig'] for c in bo))}, zu hoch höchstens {pct(max(c['bootstrap']['hoch'] for c in bo))}.\n"
        f"- **Breite** (mediane Breite, Profil durch Delta): bei m = 20 das {de(r20[0], 1)}- bis {de(r20[-1], 1)}-Fache (4 Zellen), bei m = 100 das {de(r100[0], 1)}- bis {de(r100[-1], 1)}-Fache (3 Zellen). "
        f"In keiner der {len(ue)} Zellen (alle mit m ≥ 20) blieb die Obergrenze des Profils offen ({pct(max(c['profil']['offen_oben'] for c in ue))} der Läufe); die Suche geht bis zu 14 Verdopplungen weit. "
        f"In der Zelle ξ = 0,2, m = 20, T = 1000 ist die mediane Breite des Profils {de(breit['profil']['breite_median'], 0)} σ (Delta: {de(breit['delta']['breite_median'], 0)} σ).\n"
        f"- **ξ̂ am Rand:** der Anteil der Läufe, in denen ξ̂ an der Grenze −1 klebt, ist in {sum(c['anpassung_am_rand'] == 0 for c in ue)} von {len(ue)} Zellen {pct(0)}, sonst höchstens {pct(max(c['anpassung_am_rand'] for c in ue))}."
    )


def absatz_modellfehler():
    _, _, mf, _ = _zellen()
    z = {c["id"]: c for c in mf}
    exp0, exp1, nor0, nor1, vb0, vb1 = (z[k] for k in ("exp365_m50_T100", "exp365_m50_T1000", "normal365_m50_T100", "normal365_m50_T1000", "vwz_m50_T100", "vwz_m50_T1000"))
    w = lambda c, k: c["wahre_ueberschreitung"][k]
    se95 = (0.95 * 0.05 / mf[0]["R"]) ** 0.5
    innen = [c for c in mf if c["profil"]["cov"] >= 0.95 - 2 * se95]
    aussen = sorted((c for c in mf if c["profil"]["cov"] < 0.95 - 2 * se95), key=lambda c: c["profil"]["cov"])
    assert len({c["R"] for c in mf}) == 1 and all(c["fall"] == "normal365" for c in aussen) and max(c["delta"]["cov"] for c in mf) < 0.95 and max(c["bootstrap"]["cov"] for c in mf) < 0.95
    return (
        f"- **Bei Modellfehler (Tageswerte statt GEV)** deckt Delta {pct(min(c['delta']['cov'] for c in mf))} bis {pct(max(c['delta']['cov'] for c in mf))} und der Bootstrap {pct(min(c['bootstrap']['cov'] for c in mf))} bis {pct(max(c['bootstrap']['cov'] for c in mf))}: "
        f"beide liegen in allen {len(mf)} Zellen unter 95 % (die kleinste Zahl hat Normal, T = 1000: Delta {pct(nor1['delta']['cov'])}). "
        f"Das Profil ({pct(min(c['profil']['cov'] for c in mf))} bis {pct(max(c['profil']['cov'] for c in mf))}) ist in {len(innen)} von {len(mf)} Zellen von 95 % nicht zu unterscheiden (innerhalb von 2 Standardfehlern, gegen 95 % gerechnet), "
        f"in den übrigen {len(aussen)} (beide Normal: {' und '.join(pct(c['profil']['cov']) for c in aussen)}) liegt es etwas darunter.\n"
        f"- **Das Vorzeichen des Fehlers hängt vom Fall ab:** bei Normal liegt der Median von x̂/x_T unter 1 ({de(w(nor0, 'v_median'), 3)}, {de(w(nor1, 'v_median'), 3)}; Anteil x̂ < x_T {pct(w(nor0, 'unter_1'))} und {pct(w(nor1, 'unter_1'))}), "
        f"bei Vehikel B darüber ({de(w(vb0, 'v_median'), 3)} bei T = 100, {de(w(vb1, 'v_median'), 3)} bei T = 1000; Anteil x̂ < x_T {pct(w(vb0, 'unter_1'))} und {pct(w(vb1, 'unter_1'))}; nur bei diesen Parametern und Tag = 100 Container). "
        f"Bei Exponential ist der Median nur schwach von 1 entfernt ({de(w(exp0, 'v_median'), 3)}, {de(w(exp1, 'v_median'), 3)}; Anteil x̂ < x_T {pct(w(exp0, 'unter_1'))} und {pct(w(exp1, 'unter_1'))}). "
        f"Ob der Rand der Tageswerte die Ursache ist, ist nicht untersucht: die drei Fälle unterscheiden sich zugleich in der Streuung der Tageswerte und in der Modellform.\n"
        f"- **Wahre Überschreitungswahrscheinlichkeit des Schätzwerts** (aus F^n, in Vielfachen des Ziels 1/T): Exponential, T = 100: in {pct(w(exp0, 'anteil_ueber_2'))} der Läufe tritt der geschätzte Wert real mindestens doppelt so oft auf wie das Ziel (wahre Wiederkehrperiode höchstens 50 Jahre), "
        f"in {pct(w(exp0, 'anteil_unter_halb'))} höchstens halb so oft; Normal, T = 1000: in {pct(w(nor1, 'anteil_ueber_2'))} mindestens doppelt so oft (Median {de(w(nor1, 'p_wahr_mal_T_median'), 2)} · 1/T); "
        f"Vehikel B, T = 1000: in {pct(w(vb1, 'anteil_unter_halb'))} höchstens halb so oft (Median {de(w(vb1, 'p_wahr_mal_T_median'), 2)} · 1/T). "
        f"Diese Mediane und Anteile sind je eine Ziehung von {tausend(mf[0]['R'])} Läufen und haben hier keinen Standardfehler; mit anderen Zufalls-Seeds streuen sie deutlich. "
        f"Der Median bei Normal, T = 1000 ({de(w(nor1, 'p_wahr_mal_T_median'), 2)}) ist nach Nachläufen mit anderen Zufalls-Seeds eine hohe Ziehung (die Nachläufe, nicht eingefroren, lagen meist darunter): als Größenordnung lesen."
    )


def preset_fazit_kurz():
    ue, _, mf, _ = _zellen()
    k = next(z for z in ue if z["id"] == "xi+0.2_m20_T100")
    vb = next(z for z in mf if z["id"] == "vwz_m50_T1000")
    return (de(100 * k["delta"]["cov"], 1), de(k["profil"]["breite_median"] / k["delta"]["breite_median"], 1), de(vb["wahre_ueberschreitung"]["v_median"], 2))


def alle():
    return {"streuband_absatz": absatz_streuband(), "ueberdeckung_absatz": absatz_ueberdeckung(), "modellfehler_absatz": absatz_modellfehler()}


if __name__ == "__main__":
    for name, text in alle().items():
        print(f"### {name}\n{text}\n")
    print(preset_fazit_kurz())

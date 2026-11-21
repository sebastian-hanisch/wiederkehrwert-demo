# Wiederkehrwert: was „100 Jahre“ heißt und wie sicher man ihn kennt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-wiederkehrwert-demo.streamlit.app/)**

Demo der Linie **„Extremwerte erklärt“** in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“). Sie setzt voraus, dass die Parameter (μ, σ, ξ) aus Blockmaxima geschätzt werden (Gegenstand der Demo „GEV an Blockmaxima“), und hebt die Annahme „der angepasste Pegel ist **die** Zahl“ auf: er ist eine Zufallsgröße mit **Streuband**, ein nominell 95-%-Intervall hält sein Versprechen bei kurzen Reihen oft **nicht** (Delta: in allen 13 gemessenen Zellen unter 95 %; Profil: bei m = 20 zwischen 92,4 % und 94,5 %, sonst zwischen 93,5 % und 95,9 %), und die Wiederkehrperiode ist ein **Mittel, keine Frist**. Danach folgt in der Linie die Demo „Überschreitungen und die GPD“ mit der Gabelung „Block gegen Schwelle“. Die Blöcke sind **ausgedacht**: GEV direkt aus der Extremwertverteilung gezogen, oder das Jahresmaximum aus 365 Tageswerten von Exponential- und Normalverteilung, oder das Tagesmaximum aus 100 Containern eines erfundenen Verweilzeit-Modells („Vehikel B“). Zähleinheiten stehen an jeder Zahl: **Läufe** (unabhängige Wiederholungen eines ganzen Experiments mit m Blöcken), **Reihen** (dasselbe bei den exakten Zahlen), **Blöcke** (m) und **Jahre** (T, N; ein Block ist ein Jahr).

## Warum dieses Problem

Ein „100-Jahres-Wert“ klingt nach einer Frist und nach einer Zahl. Die Wiederkehrperiode ist ein Mittel, keine Frist, und der aus Daten geschätzte Wert ist keine feste Zahl, sondern eine Zufallsgröße mit Streuband: Bei T = 100 wird der Pegel innerhalb von 100 Jahren nur mit 63,4 % mindestens einmal überschritten (Median der Wartezeit 69 Jahre), eine 50-Jahres-Reihe erreicht ihn in 39,5 % der Reihen, und aus 50 Jahresmaxima geschätzt liegt er zu 90 % zwischen dem 0,73- und 1,49-Fachen des wahren Werts (ξ = 0,2, μ = 100, σ = 20, 2 000 Läufe). Die Kernfragen der App: (1) Was heißt „100 Jahre“? (2) Wie weit liegt die Schätzung daneben, und nach welcher Seite? (3) Hält das 95-%-Intervall? (4) Wie lang muss die Reihe sein? Die Falle ist, das Intervall für die Unsicherheit zu halten und die Zahl für den Pegel: die Überdeckung ist nur über viele Wiederholungen messbar, und sie liegt bei kurzen Reihen unter 95 % (beim Delta-Intervall deutlich). Alles gilt in diesen Modellen, bei dieser Anpassung (Maximum Likelihood mit ξ ≥ −1) und bei diesen drei Intervallvarianten; es gibt in den gelesenen Quellen keine reale Reihe, an der das Gezeigte geprüft wäre, und die Demo sagt nichts über Nutzen oder Wirkung in Logistik, Lager oder Hafen (reines Lehrbuch).

## Befunde und Korrekturen gegenüber der Vormessung

Die Zahlen des Plans stammen aus einer Vormessung mit `numpy.default_rng` (Python 3.14, scipy-Anpassung); dieser Bau rechnet alles mit SplitMix64 und eigener Anpassung **neu** (anderer Zufalls-Seed, Zufallsanteile als Band mit Standardfehler). Alte Zahlen stehen unten in Code-Schrift. Was sich verschoben oder widersprochen hat:

- **Profil, schwächste Zelle (ξ = −0,3, m = 20, T = 100):** die Vormessung nannte `88,4 %` bei 1 000 Läufen und `92,8 %` bei 3 000 frischen Läufen. Hier sind es 92,5 % (R = 1 000) und 92,0 % (R = 3 000): die `88,4 %` weichen von den `92,8 %` (3 000 frische Läufe der Vormessung) und von diesem Bau ab; die Ursache ist nicht untersucht (naheliegend: eine andere Anpassung oder ein einfacher Start der Profilsuche in der Vormessung, eine niedrige Ziehung allein ist bei etwa 4 Standardfehlern der Differenz unwahrscheinlich). Hier liegt die Zelle bei etwa 92 bis 93 % und bleibt die schwächste Profil-Zelle (zusammen mit ξ = 0,2, m = 20, T = 1000: 92,4 %).
- **Profil-Spanne:** `93,3 bis 95,2 %` in 12 Zellen → 92,4 bis 95,9 % in 13 Zellen. Die größten Verschiebungen: (ξ = 0, m = 50, T = 1000) `93,8 %` → 95,9 %, (ξ = 0,2, m = 20, T = 1000) `94,5 %` → 92,4 %; beides etwa 2 Standardfehler der Differenz, also im Rahmen des Ziehungsrauschens.
- **Profil oben offen:** in der Vormessung blieb die Obergrenze in der Zelle ξ = 0,2, m = 20, T = 1000 in `9,1 %` der Läufe offen (Breite `186 σ`). Hier ist sie in **keinem** Lauf offen (Breite 235 σ). Ursache nicht untersucht; naheliegend ist der Mehrfachstart der Profilsuche in diesem Bau, der den Schnittpunkt auch dort findet, wo ein einzelner Start ihn verfehlt (gemessen ist nur: 0 von 13 Zellen mit offener Obergrenze).
- **Delta-Spanne:** `70,4 bis 91,4 %` → 76,0 bis 91,9 %. Die Zelle ξ = −0,3, m = 20 hat dort `70,4 %` (frische 3 000 Läufe: `74,4 %`), hier 76,0 % (R = 1 000) und 72,1 % (R = 3 000). „Delta unterdeckt“ gilt weiter in 13 von 13 Zellen.
- **Bootstrap (B = 199, R = 1 000):** Spanne `84,6 bis 91,5 %` → 82,2 bis 91,2 %; Hauptzelle `90,8 %` → 91,2 %; „unter dem Profil“ bleibt in 6 von 6 Zellen, der kleinste Abstand wächst von `3,2` auf 3,8 Punkte; die Zelle ξ = 0, m = 20 liegt mit `84,6 %` → 82,2 % am Rand der Spanne. Bei Modellfehler rechnet dieser Bau mit B = 199 und R = 1 000 statt `B = 99` und `R = 500` (Standardfehler 0,9 statt `1,3` Punkte).
- **Streuband:** Verschiebungen höchstens im Rahmen des Standardfehlers der Quantile: ξ = 0,2, m = 50, T = 100 `0,735 bis 1,502` → 0,731 bis 1,490, T = 1000 `0,569 bis 2,215` → 0,565 bis 2,211; m = 20, T = 100 `0,624 bis 2,206` → 0,628 bis 2,120; m = 20, T = 1000 `0,436 bis 5,106` → 0,430 bis 4,921. Anteile: m = 50, T = 1000 mehr als doppelt so hoch `7,4 %` → 6,9 %, unter dem 0,8-Fachen `30,1 %` → 28,5 %; m = 20, T = 1000 unter dem 0,8-Fachen `38,7 %` → 40,3 %.
- **Unterschätzung:** `50 bis 62 %` in 45 Zellen → 47,8 bis 61,8 %. Unter 50 % liegen zwei Zellen (ξ = 0,2, m = 100, T = 100 und T = 1000).
- **Modellfehler:** die wahre Überschreitungswahrscheinlichkeit des Schätzwerts verschiebt sich für Normal, T = 1000 von `1,79` auf 2,25 (Median) und von `48,5 %` auf 52,5 % der Läufe mit mindestens doppelt so oft; Vehikel B, T = 1000: Median der wahren Überschreitungswahrscheinlichkeit `0,42` → 0,47, Median von x̂/x_T `1,23` → 1,20. Delta Normal, T = 1000: `80,0 %` → 82,6 %.
- **Nicht übernommen:** „die Zelle ξ = −0,3, m = 20 habe `88,4 %`“ (siehe oben). Die Zahlen von Huang u. a. und genEVA sind weiter nicht nachgerechnet.


## Modell

Ein Block ist ein Jahr; die m Blockmaxima sind unabhängig und gleich verteilt. **Wiederkehrwert** (Pegel) x_T ist der Wert, den ein Block mit Wahrscheinlichkeit p = 1/T überschreitet, G(x_T) = 1 − 1/T. Für die GEV in der Vorzeichenkonvention der Linie (ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ mit endlichem Rand) gilt x_T = μ + σ((−ln(1−p))^(−ξ) − 1)/ξ, bei ξ = 0: μ − σ ln(−ln(1−p)) (Coles 2001, Gl. 3.4, in eigenen Worten). **`scipy.stats.genextreme` hat c = −ξ**; im Test steht diese Umrechnung (Pegel-Formel, Likelihood und Anpassung gegen `genextreme` mit c = −ξ). ξ ist hier der Formparameter (Extremwertindex), nie der Extremalindex θ. Die App und die Formeln im 📐-Expander: Pegel, Wartezeit (geometrisch, Mittel T, Median das kleinste k mit (1 − 1/T)^k < 1/2; bei T = 2 sind 1 und 2 beide Medianwerte, die Tabelle nimmt 2), Delta-, Profil- und Bootstrap-Intervall sowie der exakte Ein-Parameter-Fall (Gumbel, σ bekannt, Gamma-Verteilung). Der Block „Modell“ im Expander ist vollständig für diese Demo (Variablenbereiche, Definition jeder Größe, drei Intervallvarianten); nicht enthalten sind abhängige Blöcke, Trend, Schwellenmodelle und andere Niveaus als 95 %.

**Bezugsgröße „Jahr“:** ein Block ist ein Jahr. Bei Exponential und Normal sind es **n = 365 Tageswerte je Jahr** (die 100-Jahres-Werte anderer Demos der Linie mit anderem n je Jahr sind nicht dieselbe Zahl), bei Vehikel B ein Tag aus 100 Containern. Der wahre Pegel kommt exakt aus F^n (nicht aus GEV-Parametern: die GEV ist hier nur das Grenzgesetz): Exponential 10,500 / 12,807 (T = 100 / 1 000), Normal 4,033 / 4,545, Vehikel B 155,128 / 287,123 Tage (Zielgröße „Tagesmaximum aus 100 Containern“; je Container zur Überschreitungswahrscheinlichkeit 10⁻⁵ sind es 287,158 Tage). Die Gewichtstabelle von Vehikel B (1 224 verschiedene Komponenten aus 4 704 Aufzählungsschritten, davon 2 376 mit Gewicht) ist eine wörtliche Kopie der gemeinsamen Definition der Linie (Stand 2026-11-20, SHA-256 `669f821c78e0bf38c47fa846232f27cd2ff126f4fceadca5369a5c9af5551c9f`); `tests/test_vehikel_b.py` prüft sie gegen eine eigene Aufzählung aus der Definition; ein Vergleich mit der Datei des Eigentümer-Stücks gehört nicht zu den Tests (geprüft sind Hash und Aufzählung). „Tag = 100 Container“ ist eine Annahme.

## Methodik

- **Zufall:** nur der reine SplitMix64-Generator `wkw_rng.py` (Kopie der Vorlage des Portfolios, `tests/test_copies.py`); GEV und F^n-Blöcke per Inversion (ein Zug je Jahr), Vehikel B per Umkehrung der exakten Verteilung des Tagesmaximums (Tabelle, ein Zug je Tag). Jeder Lauf hat einen eigenen Strom aus (Basis, Zelle, Lauf). Es läuft kein `numpy.random` und kein scipy in der App.
- **Anpassung:** Maximum Likelihood mit ξ ≥ −1 in numpy und Standardbibliothek (eigenes Nelder-Mead, Start aus L-Momenten, Neustart am Ergebnis); gegen `genextreme.fit` getestet.
- **Intervalle (alle nominell 95 %):** *Delta* mit der beobachteten Information (numerische Hesse-Matrix, symmetrisch um x̂), *Profil-Likelihood* (Devianz höchstens 3,8415, Mehrfachstart je Pegel, Obergrenze unendlich, wenn die Kurve die Schwelle im Suchbereich nicht erreicht) und *parametrischer Perzentil-Bootstrap* (B = 199 Anpassungen an Stichproben aus der angepassten GEV, keine BCa- oder Studentisierungs-Korrektur: Aussagen gelten nur für diese Variante).
- **Überdeckung** = Anteil der Läufe, in denen das Intervall den wahren Pegel enthält. „Zu niedrig“: der wahre Pegel liegt **über** dem Intervall; „zu hoch“: **darunter**. Ein Delta-Intervall ohne positiv definite Hesse-Matrix zählt als verfehlt, ein Profil-Intervall ohne Obergrenze (Obergrenze unendlich) als überdeckend, wenn der wahre Pegel über der Untergrenze liegt, sonst als zu hoch.
- **Eingefroren** sind die schweren Messungen (`wkw_coverage.json`, Erzeuger `tools/wkw_make_coverage.py` mit `--check`, der drei Stichproben als Band nachrechnet) und die exakten Tabellen (`wkw_reference.json`, `tools/wkw_make_reference.py --check`). Live rechnet die App je Experiment höchstens: Experiment 1 bis 2 Millionen Zufallszahlen, Experiment 3 bis 150 Läufe, Experiment 4 bis 20 Läufe.

## Befunde (gemessen, keine Behauptungen)

Jede Zahl steht mit ihrer Zähleinheit; die Tabelle „Zahl → Test“ am Ende dieses Abschnitts nennt zu jeder Zahl den Test. Alle Tabellen dieses Abschnitts erzeugt `tests/readme_tabellen.py` aus der Rechnung und aus `wkw_coverage.json`; `tests/test_claims.py` prüft, dass sie wörtlich hier stehen.

### A. Was „T Jahre“ heißt (exakt, unabhängige Jahre, Zähleinheit Jahre und Reihen)

| T | P(mindestens eine Überschreitung in T Jahren) | Median der Wartezeit (Jahre) |
|---|---|---|
| 2 | 75,00 % | 2 |
| 10 | 65,13 % | 7 |
| 50 | 63,58 % | 35 |
| 100 | 63,40 % | 69 |
| 1000 | 63,23 % | 693 |

Der Grenzwert für große T ist 1 − 1/e = 63,21 %. Die Wiederkehrperiode ist das **Mittel** der geometrischen Wartezeit (Mittel T, der Median liegt ab T = 3 darunter; bei T = 2 ist er 2, gleich dem Mittel). Die Nutzungsdauer N und die Reihenlänge m sind dasselbe Zählen: der 100er-Wert wird in N = 50 Jahren mit 39,50 % mindestens einmal überschritten, und genauso erreicht das größte Jahr einer 50-Jahres-Reihe ihn in 39,50 % der Reihen (für jede stetige Verteilung unabhängiger, gleich verteilter Jahre).

**Eine Reihe von m Jahren erreicht den Pegel** (Anteil der Reihen, 1 − (1 − 1/T)^m; die Simulation mit 4 000 gezogenen Reihen je Zelle liegt in allen 15 Zellen (T = 10, 100, 1 000, m = 20 bis 500) innerhalb von 4 Standardfehlern):

| Reihenlänge m (Jahre) | T = 10 | T = 100 | T = 1000 |
|---|---|---|---|
| 20 | 87,84 % | 18,21 % | 1,98 % |
| 50 | 99,48 % | 39,50 % | 4,88 % |
| 100 | 100,00 % | 63,40 % | 9,52 % |

**Jahre Reihe**, damit das Maximum den Wert mit Sicherheit q erreicht (exakt, kleinste Länge):

| Sicherheit q | T = 10 | T = 100 | T = 1000 |
|---|---|---|---|
| 50 % | 7 | 69 | 693 |
| 90 % | 22 | 230 | 2 302 |
| 95 % | 29 | 299 | 2 995 |
| 99 % | 44 | 459 | 4 603 |

Eine Sicherheit q über N Jahre verlangt T = 1/(1 − q^(1/N)): für q = 95 % und N = 50 Jahre ist T = 975,3, für N = 100 Jahre T = 1 950,1 (Rechnung in diesem Modell; die Meterangaben des Deich-Beispiels bei Embrechts u. a. sind nicht übernommen). Das Maximum aus m Jahren wird vom nächsten Jahr mit der Wahrscheinlichkeit genau 1/(m + 1) übertroffen, für jede stetige Verteilung (m = 50: 1/51 = 1,96 %, Aufzählung der Rangordnungen im Test). Eine Wiederkehrperiode über m + 1 Jahren liegt darum jenseits dessen, was die Rangordnung der Daten auflöst. Dass der Pegel auch über dem größten Wert der Reihe liegt, gilt erst später: in mehr als der Hälfte der Reihen für m = 50 ab T = 73, bei langen Reihen ab etwa dem 1,4-Fachen von m (stetige Verteilung, unabhängige, gleich verteilte Jahre).

### B. Ein Parameter bekannt: alles exakt (Gumbel, σ bekannt, ξ = 0; Zähleinheit Reihen)

Für x_i ~ Gumbel(μ, σ) ist exp(−(x_i − μ)/σ) ~ Exp(1); der ML-Schätzer ist μ̂ = −σ ln(Mittel von exp(−x_i/σ)), und mit G = Σ exp(−(x_i − μ)/σ) ~ Gamma(m, 1) gilt **μ̂ − μ = −σ ln(G/m)**: der Fehler **jedes** Wiederkehrwerts ist dieser eine Fehler und hängt nicht von T ab. Die beobachtete Information ist m (exakt), das Delta-Intervall ist μ̂ ± 1,96 σ/√m, das Profil-Intervall μ̂ + ln r mit r − 1 − ln r = 3,8415/(2m); die Überdeckung folgt exakt aus der Gamma-Verteilungsfunktion:

| m (Reihen) | Delta deckt (exakt) | Profil deckt (exakt) | Fehler μ̂ − μ, 5 bis 95 % (Einheit σ) |
|---|---|---|---|
| 5 | 93,220 % | 94,619 % | −0,605 bis 0,931 |
| 10 | 94,102 % | 94,809 % | −0,451 bis 0,611 |
| 20 | 94,549 % | 94,905 % | −0,332 bis 0,411 |
| 50 | 94,819 % | 94,962 % | −0,218 bis 0,249 |
| 500 | 94,982 % | 94,996 % | −0,072 bis 0,075 |

**Aussage mit Voraussetzung:** sind σ und ξ bekannt, kostet die Extrapolation von T = 100 auf 1 000 **nichts** an Streuung des absoluten Fehlers (in Einheiten σ ist er für jedes T derselbe); das Verhältnis V = x̂/x_T hängt dagegen über x_T von T ab. Schon im einfachsten Fall liegt Delta unter Profil und beide unter 95 % (m = 5 bis 500, 5 von 5 Zeilen). Handbeispiel (fünf ausgedachte Jahresmaxima ln 2, …, ln 6, σ = 1): μ̂ = −ln 0,29 = 1,2379, x̂₁₀₀ = 5,8380; die drei Intervalle gelten für μ (für x₁₀₀ jeweils um +4,6001 verschieben): Delta [0,3614; 2,1144], Profil [0,2123; 2,0030], exakt über den Pivot [0,1130; 1,9549] (`tests/test_handbeispiele.py`). Das ist der Ein-Parameter-Fall; er prüft die Wartezeit-Arithmetik, die Routinen im einfachsten Modell und die Gamma-Überdeckung, **nicht** die Dreiparameter-Anpassung.

### C. Streuband des geschätzten Pegels (eingefroren, Zähleinheit Läufe)

Blöcke exakt GEV(μ = 100, σ = 20, ξ), Maximum Likelihood mit ξ ≥ −1, **R = 2 000 Läufe je Zelle**, V = x̂_T / x_T (wahrer Pegel exakt, GEV-Quantil); 90-%-Band q05 bis q95 von V (Median in Klammern), ξ = 0,2:

| m (Blöcke) | T = 10 | T = 100 | T = 1000 |
|---|---|---|---|
| 20 | 0,83 bis 1,22 (Median 0,98) | 0,63 bis 2,12 (Median 0,96) | 0,43 bis 4,92 (Median 0,95) |
| 50 | 0,89 bis 1,13 (Median 0,99) | 0,73 bis 1,49 (Median 0,99) | 0,56 bis 2,21 (Median 0,98) |
| 100 | 0,92 bis 1,09 (Median 1,00) | 0,80 bis 1,29 (Median 1,00) | 0,67 bis 1,62 (Median 1,01) |
| 200 | 0,94 bis 1,06 (Median 1,00) | 0,85 bis 1,19 (Median 0,99) | 0,74 bis 1,41 (Median 0,99) |
| 500 | 0,97 bis 1,04 (Median 1,00) | 0,91 bis 1,10 (Median 1,00) | 0,84 bis 1,21 (Median 1,00) |

Für andere ξ (m = 50):

| ξ (m = 50) | T = 100 | T = 1000 |
|---|---|---|
| −0,2 | 0,90 bis 1,09 | 0,86 bis 1,17 |
| 0,0 | 0,83 bis 1,23 | 0,73 bis 1,48 |

- **Unterschätzung:** der Anteil der Läufe, die unterschätzen (V < 1), liegt in allen 45 Zellen zwischen 47,8 % und 61,8 % (Standardfehler 1,1 Punkte); in 43 von 45 Zellen liegt der Median von V unter 1; in den übrigen zwei (ξ = 0,2, m = 100 bei T = 100 und T = 1000) liegt er bei 1,00 und 1,01 und ist von 1 nicht zu unterscheiden. Die Zählung und die Namen der zwei Ausnahmen hängen vom Zufalls-Seed; der kleinste Anteil (47,8 %) liegt etwa 2 Standardfehler unter 50 %.
- **Rechtsschief** (q95 − 1 größer als 1 − q05): bei ξ = 0,2 in 15 von 15 Zellen, bei ξ = 0 in 11 von 15 (Ausnahmen: T = 10 ab m = 50), bei ξ = −0,2 in 3 von 15. Das Band liegt also nur bei ξ ≥ 0 meist weiter oberhalb als unterhalb des wahren Pegels.
- **Zu 90 % zwischen:** ξ = 0,2, m = 50: der 100er-Wert zwischen dem 0,73- und 1,49-Fachen, der 1000er-Wert zwischen 0,56 und 2,21, in 6,9 % der Läufe mehr als doppelt so hoch und in 28,5 % unter dem 0,8-Fachen (T = 1000); bei m = 500 sind es 0,91 bis 1,10 (T = 100).
- **Das Band hängt nicht nur an T/m:** bei T/m = 2 ist q95/q05 für m = 50, T = 100 gleich 2,04, für m = 500, T = 1000 gleich 1,45 (ξ = 0,2).
- **Die Extrapolation kostet wegen der geschätzten Parameter** (ξ = 0, m = 50, Streuung des Fehlers in Einheiten σ, robust als Viertelspanne/1,349, 2 000 Läufe): für T = 10 / 100 / 1000 GEV mit drei freien Parametern 0,358 / 1,127 / 2,465, Gumbel mit freiem (μ, σ) 0,338 / 0,580 / 0,820, σ bekannt 0,144 / 0,144 / 0,144 (exakt 0,142). Streuungen aus den Läufen mit ξ = 0,2 und m = 20 werden von Ausreißern bestimmt, darum steht oben das Quantilband.

### D. Überdeckung des nominell 95-%-Intervalls (eingefroren, Zähleinheit Läufe)

Blöcke exakt GEV(0, 1, ξ) (die Überdeckung hängt nicht an Ort und Skala, `tests/test_properties.py`); **R = 1 000 Läufe je Zelle** (Standardfehler je nach Zelle 0,6 bis 0,8 Punkte beim Profil, 0,9 bis 1,4 Punkte bei Delta und 0,9 bis 1,2 Punkte beim Bootstrap; Zufalls-Seeds je Zelle in der Datei); der wahre Pegel ist das GEV-Quantil. Bootstrap: B = 199 Anpassungen, nur bei T = 100 und ξ = 0 oder 0,2 (daher „-“ sonst). „Zu niedrig“: der wahre Pegel liegt über dem Intervall.

| ξ | m | T | Delta | Profil | Perzentil-Bootstrap (B = 199) | Delta: zu niedrig / zu hoch |
|---|---|---|---|---|---|---|
| −0,3 | 20 | 100 | 76,0 % | 92,5 % | - | 23,9 % / 0,0 % |
| −0,3 | 50 | 100 | 84,2 % | 94,3 % | - | 15,8 % / 0,0 % |
| −0,3 | 100 | 100 | 87,3 % | 94,7 % | - | 12,7 % / 0,0 % |
| 0,0 | 20 | 100 | 79,4 % | 94,5 % | 82,2 % | 20,5 % / 0,0 % |
| 0,0 | 50 | 100 | 86,8 % | 93,5 % | 87,5 % | 13,2 % / 0,0 % |
| 0,0 | 100 | 100 | 90,5 % | 95,4 % | 90,0 % | 9,4 % / 0,1 % |
| 0,2 | 20 | 100 | 80,0 % | 93,1 % | 84,8 % | 20,0 % / 0,0 % |
| 0,2 | 50 | 100 | 89,1 % | 95,1 % | 91,2 % | 10,9 % / 0,0 % |
| 0,2 | 100 | 100 | 90,0 % | 94,2 % | 90,4 % | 10,0 % / 0,0 % |
| 0,2 | 50 | 10 | 91,9 % | 95,5 % | - | 8,1 % / 0,0 % |
| 0,2 | 50 | 1000 | 86,8 % | 94,8 % | - | 13,2 % / 0,0 % |
| 0,0 | 50 | 1000 | 85,9 % | 95,9 % | - | 14,1 % / 0,0 % |
| 0,2 | 20 | 1000 | 77,7 % | 92,4 % | - | 22,3 % / 0,0 % |

- **Delta unterdeckt in 13 von 13 Zellen** (76,0 % bis 91,9 %; diese Delta-Variante, diese Anpassung), und fast alle Fehlgriffe liegen **zu niedrig**: zu niedrig 8,1 % bis 23,9 %, zu hoch höchstens 0,1 % der Läufe.
- **Profil** liegt in 13 von 13 Zellen zwischen 92,4 % und 95,9 % (Standardfehler 0,6 bis 0,8 Punkte); unter 93 % liegen 2 Zellen (ξ = 0,2, m = 20, T = 1000: 92,4 %; ξ = −0,3, m = 20, T = 100: 92,5 %), beide bei m = 20, und diese Zahl hängt am Zufalls-Seed. Mehr als 2 Standardfehler unter 95 % (gegen 95 % gerechnet: Standardfehler 0,7 Punkte) liegen 4 von 13 Zellen, alle bei m ≤ 50; die übrigen 9 sind mit 95 % verträglich. Es verfehlt auf **beiden** Seiten (zu niedrig 2,5 % bis 6,5 %, zu hoch 1,0 % bis 3,2 %).
- **Die schwächste Profil-Zelle** (ξ = −0,3, m = 20, T = 100) schwankt mit den Läufen: 92,5 % bei 1 000 Läufen, 92,0 % bei 3 000 frischen Läufen (Standardfehler 0,5 Punkte); Delta liegt dort bei 76,0 % (R = 1 000) und 72,1 % (R = 3 000).
- **Perzentil-Bootstrap** (B = 199, diese Variante): 82,2 % bis 91,2 % in 6 Zellen, **unter dem Profil in 6 von 6 Zellen** (kleinster Abstand 3,8 Punkte, größter 12,3); Hauptzelle (ξ = 0,2, m = 50, T = 100): 91,2 %. Auch er verfehlt überwiegend nach unten: zu niedrig 7,8 % bis 17,4 %, zu hoch höchstens 1,5 %.
- **Breite** (mediane Breite, Profil durch Delta): bei m = 20 das 1,9- bis 4,8-Fache (4 Zellen), bei m = 100 das 1,1- bis 1,2-Fache (3 Zellen). In keiner der 13 Zellen (alle mit m ≥ 20) blieb die Obergrenze des Profils offen (0,0 % der Läufe); die Suche geht bis zu 14 Verdopplungen weit. In der Zelle ξ = 0,2, m = 20, T = 1000 ist die mediane Breite des Profils 235 σ (Delta: 49 σ).
- **ξ̂ am Rand:** der Anteil der Läufe, in denen ξ̂ an der Grenze −1 klebt, ist in 10 von 13 Zellen 0,0 %, sonst höchstens 2,3 %.

### E. Modellfehler: Blöcke aus Tageswerten, wahrer Pegel exakt aus F^n (eingefroren, m = 50, R = 1 000 Läufe, Bootstrap B = 199)

| Fall (m = 50) | T | wahrer Pegel | Delta | Profil | Perzentil-Bootstrap (B = 199) |
|---|---|---|---|---|---|
| Exponential, Jahr aus 365 Tagen | 100 | 10,500 | 88,2 % | 94,7 % | 88,2 % |
| Exponential, Jahr aus 365 Tagen | 1000 | 12,807 | 87,0 % | 94,1 % | 89,7 % |
| Normal, Jahr aus 365 Tagen | 100 | 4,033 | 84,8 % | 93,1 % | 84,2 % |
| Normal, Jahr aus 365 Tagen | 1000 | 4,545 | 82,6 % | 93,6 % | 85,6 % |
| Vehikel B, Tag aus 100 Containern | 100 | 155,128 | 92,3 % | 95,8 % | 93,9 % |
| Vehikel B, Tag aus 100 Containern | 1000 | 287,123 | 93,4 % | 93,9 % | 93,9 % |

Die **wahre** Überschreitungswahrscheinlichkeit des geschätzten Pegels (aus F^n), in Vielfachen des Ziels 1/T:

| Fall (m = 50) | T | Median p_wahr · T | p_wahr ≥ 2/T | p_wahr ≤ 0,5/T | Median x̂/x_T | Anteil x̂ < x_T |
|---|---|---|---|---|---|---|
| Exponential, Jahr aus 365 Tagen | 100 | 1,12 | 30,1 % | 24,8 % | 0,99 | 54,8 % |
| Exponential, Jahr aus 365 Tagen | 1000 | 1,17 | 42,0 % | 37,7 % | 0,99 | 52,3 % |
| Normal, Jahr aus 365 Tagen | 100 | 1,29 | 33,7 % | 19,9 % | 0,98 | 60,2 % |
| Normal, Jahr aus 365 Tagen | 1000 | 2,25 | 52,5 % | 29,2 % | 0,96 | 62,2 % |
| Vehikel B, Tag aus 100 Containern | 100 | 0,86 | 22,1 % | 30,8 % | 1,04 | 44,8 % |
| Vehikel B, Tag aus 100 Containern | 1000 | 0,47 | 24,5 % | 51,4 % | 1,20 | 35,1 % |

- **Bei Modellfehler (Tageswerte statt GEV)** deckt Delta 82,6 % bis 93,4 % und der Bootstrap 84,2 % bis 93,9 %: beide liegen in allen 6 Zellen unter 95 % (die kleinste Zahl hat Normal, T = 1000: Delta 82,6 %). Das Profil (93,1 % bis 95,8 %) ist in 4 von 6 Zellen von 95 % nicht zu unterscheiden (innerhalb von 2 Standardfehlern, gegen 95 % gerechnet), in den übrigen 2 (beide Normal: 93,1 % und 93,6 %) liegt es etwas darunter.
- **Das Vorzeichen des Fehlers hängt vom Fall ab:** bei Normal liegt der Median von x̂/x_T unter 1 (0,985, 0,962; Anteil x̂ < x_T 60,2 % und 62,2 %), bei Vehikel B darüber (1,044 bei T = 100, 1,200 bei T = 1000; Anteil x̂ < x_T 44,8 % und 35,1 %; nur bei diesen Parametern und Tag = 100 Container). Bei Exponential ist der Median nur schwach von 1 entfernt (0,989, 0,988; Anteil x̂ < x_T 54,8 % und 52,3 %). Ob der Rand der Tageswerte die Ursache ist, ist nicht untersucht: die drei Fälle unterscheiden sich zugleich in der Streuung der Tageswerte und in der Modellform.
- **Wahre Überschreitungswahrscheinlichkeit des Schätzwerts** (aus F^n, in Vielfachen des Ziels 1/T): Exponential, T = 100: in 30,1 % der Läufe tritt der geschätzte Wert real mindestens doppelt so oft auf wie das Ziel (wahre Wiederkehrperiode höchstens 50 Jahre), in 24,8 % höchstens halb so oft; Normal, T = 1000: in 52,5 % mindestens doppelt so oft (Median 2,25 · 1/T); Vehikel B, T = 1000: in 51,4 % höchstens halb so oft (Median 0,47 · 1/T). Diese Mediane und Anteile sind je eine Ziehung von 1 000 Läufen und haben hier keinen Standardfehler; mit anderen Zufalls-Seeds streuen sie deutlich. Der Median bei Normal, T = 1000 (2,25) ist nach Nachläufen mit anderen Zufalls-Seeds eine hohe Ziehung (die Nachläufe, nicht eingefroren, lagen meist darunter): als Größenordnung lesen.

### Zahl → Test

**Zahl → Test.** Jede Zahl der README steht entweder in einer der obigen Tabellen (der Code erzeugt sie aus der Rechnung und aus `wkw_coverage.json`; `tests/test_claims.py` verlangt sie wörtlich), ist ein Wert der eingefrorenen Dateien oder steht in dieser Tabelle mit dem Test, der genau diese Zahl prüft. Alles übrige in `test_jede_zahl_der_readme_ist_belegt` ist eine benannte Strukturzahl (Reglergrenzen, Zellen, Zählungen).

| Zahl | Test |
|---|---|
| 63,4 | `test_handbeispiele.py::test_h1_wahrscheinlichkeit_in_T_jahren_exakt_als_bruch` |
| 63,2 | `test_handbeispiele.py::test_h1_grenzwert_und_mittel_der_wartezeit` |
| 69 | `test_handbeispiele.py::test_h1_median_der_wartezeit_69_und_30_jahre` |
| 39,5 | `test_handbeispiele.py::test_h2_reihe_von_50_jahren_und_jahre_fuer_sicherheit` |
| 4,9 | `test_handbeispiele.py::test_h2_reihe_von_50_jahren_und_jahre_fuer_sicherheit` |
| 975,3 | `test_handbeispiele.py::test_h2_sicherheit_ueber_nutzungsdauer_deich_beispiel` |
| 1 950,1 | `test_handbeispiele.py::test_h2_sicherheit_ueber_nutzungsdauer_deich_beispiel` |
| 475,06 | `test_claims.py::test_ekm_475_jahre_beispiel` |
| 1,96 | `test_oracle_wiederkehrwert.py::test_f_rand_der_daten_exakt_1_durch_m_plus_1_fuer_stetige_verteilungen` |
| 10,500 | `test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel` |
| 12,807 | `test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel` |
| 4,033 | `test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel` |
| 4,545 | `test_oracle_wiederkehrwert.py::test_b_handwerte_exakter_pegel` |
| 155,128 | `test_vehikel_b.py::test_pegel_gegen_scipy_mischung` |
| 287,123 | `test_vehikel_b.py::test_pegel_gegen_scipy_mischung` |
| 287,158 | `test_vehikel_b.py::test_pegel_gegen_scipy_mischung` |
| 1 224 | `test_vehikel_b.py::test_tabelle_gleich_eigener_aufzaehlung` |
| 4 704 | `test_vehikel_b.py::test_tabelle_gleich_eigener_aufzaehlung` |
| 2 376 | `test_vehikel_b.py::test_zahl_der_komponenten_mit_gewicht_2376` |
| 0,29 | `test_handbeispiele.py::test_h4_ml_und_pegel` |
| 1,2379 | `test_handbeispiele.py::test_h4_ml_und_pegel` |
| 5,8380 | `test_handbeispiele.py::test_h4_ml_und_pegel` |
| 0,3614 | `test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu` |
| 2,1144 | `test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu` |
| 0,2123 | `test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu` |
| 2,0030 | `test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu` |
| 0,1130 | `test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu` |
| 1,9549 | `test_handbeispiele.py::test_h4_drei_intervalle_fuer_mu` |
| 3,8415 | `test_oracle_wiederkehrwert.py::test_e_profil_endpunkte_haben_devianz_3_8415` |
| 475 | `test_claims.py::test_ekm_475_jahre_beispiel` |
| 63,21 | `test_handbeispiele.py::test_h1_grenzwert_und_mittel_der_wartezeit` |
| 43 | `test_claims.py::test_streuband_aussagen` |
| 11 | `test_claims.py::test_streuband_aussagen` |
| 2,04 | `test_claims.py::test_streuband_aussagen` |
| 1,45 | `test_claims.py::test_streuband_aussagen` |
| 73 | `test_claims.py::test_pegel_liegt_erst_ab_etwa_1_4_m_ueber_dem_reihenmaximum` |
| 1,4 | `test_claims.py::test_pegel_liegt_erst_ab_etwa_1_4_m_ueber_dem_reihenmaximum` |
| 4,6001 | `test_handbeispiele.py::test_h4_ml_und_pegel` |


## Ehrliche Grenzen

- **Alles ausgedacht.** Es gibt keine reale Reihe, an der das Gezeigte geprüft ist. Das Deich-Beispiel (Embrechts u. a. 8.1.5) steht nur als Lehrbuchbeispiel; seine Meterangaben sind Annahmen des Buchs und nicht übernommen. Was die Zahlen für einen echten Betrieb heißen, wird nicht behauptet.
- **Unabhängige, gleich verteilte Jahre.** Bei Abhängigkeit (Extremalindex θ < 1) ist die Wiederkehrperiode clusterweise zu lesen, bei Trend gilt „Risiko in n Jahren ≠ Wiederkehrwert“: beides hier nicht gemessen.
- **Negative Befunde stehen gleichberechtigt:** auch das Profil-Intervall liegt bei kurzen Reihen unter 95 % (siehe die schwächste Zelle ξ = −0,3, m = 20); es ist bei kurzen Reihen viel breiter als Delta (in der Vormessung war die Obergrenze manchmal offen, in diesem Bau in den 13 gemessenen Zellen mit m ≥ 20 in keinem Lauf; in einer Nebenmessung ohne eingefrorene Datei blieb sie bei m = 10 (ξ = 0,2, T = 1000) in einzelnen Läufen offen). Mehr Jahre helfen, aber das Streuband hängt nicht nur an T/m. Der Bootstrap liegt in dieser Variante unter dem Profil, ist aber nicht „schlecht“ im Allgemeinen.
- **Aussagen gelten für die gemessene Variante**, nicht für die Methodenklasse: „Delta unterdeckt“ gilt für die beobachtete Information mit numerischer Hesse-Matrix und die Maximum-Likelihood-Anpassung dieser Demo; „Bootstrap liegt unter Profil“ nur für den parametrischen Perzentil-Bootstrap mit B = 199 (nicht BCa, nicht Bootstrap-t).
- **Die Überdeckung hängt nicht an Ort und Skala der GEV** (Äquivarianz, im Test gezeigt), das Streuband von x̂/x_T schon: es gilt für GEV(100, 20, ξ).
- **Zufallsanteile sind Bänder.** Alle Zufallszahlen der README nennen ihre Zahl der Läufe; einen Standardfehler haben die Überdeckungen und Anteile, nicht aber die Quantile des Streubands (am stärksten streut q95 bei T = 1000) und die Mediane der wahren Überschreitung in Abschnitt E, die mit anderen Zufalls-Seeds merklich anders ausfallen können. Die Zahlen stammen aus Läufen mit dem Generator SplitMix64 unter Windows (Python 3.14, numpy 2.5); Ganzzahlen und Brüche sind exakt, Läufe nur als Band wiederholbar (Gleitkomma-Letztstelle plattformabhängig).
- **Browser:** Die App wurde im Browser durchgeklickt (Streamlit gestartet, Seite in einem Browser-Fenster): alle sieben Presets, die Probe in Experiment 4 und der Formel-Expander liefen ohne Fehlermeldung der Seite und ohne KaTeX-Fehler (`.katex-error`); AppTest sieht solche Skriptfehler nicht. Nicht geprüft: Streamlit Cloud, schmale Telefon-Fenster, das dunkle Design.
- **Rechenzeit:** Zeiten sind kein Befund. Die App begrenzt die Live-Rechnung (Experiment 1: 2 Millionen Zufallszahlen, Experiment 3: 150 Läufe, Experiment 4: 20 Läufe); der Bootstrap kostet je Lauf ein Vielfaches von Delta und Profil und ist in der Probe abschaltbar. Die Laufzeit auf Streamlit Cloud ist nicht gemessen.


## 🚧 Bewusst nicht enthalten / nicht gemessen

- Intervalle: BCa, Bootstrap-t und ein **nichtparametrischer** Bootstrap für den Pegel; die Delta-Methode auf der ln-Skala oder mit erwarteter Information; Bayes-Intervalle; prädiktive Wiederkehrwerte (Sisson, Pericchi, Coles: nur als Suchtreffer bekannt, nicht gelesen).
- Intervalle für Pegel aus **Schwellenmodellen** (Überschreitungen und GPD): nicht gemessen, nur der Blockweg.
- **Abhängige Blöcke**, Trend, gleitende Blöcke, Kovariaten: nicht gemessen.
- Zellen: ξ = −0,3 mit Bootstrap, m < 20 und m > 100 für die Überdeckung (für das Streuband gibt es m = 200 und 500), Intervall-Breite und Überdeckung für T = 10 und 1 000 nur in den eingefrorenen Zellen, andere Niveaus als 95 % (90 % und 99 % nicht gemessen), die Zeile „ξ̂ klebt am Rand“ als Intervallfehler.
- Die Zahlen von Huang u. a. (Tabelle 1; nominell 90 %, Log-Histospline gegen POT/GPD an Überschreitungen, anderes Verfahren) und genEVA (nominell 50 %, Quantilregression) sind **nicht nachgerechnet** und nicht mit den Zahlen dieser Demo vergleichbar.
- Rechenzeit auf Streamlit Cloud und der Lauf der Tests unter Linux (Docker) sind nicht gemessen; der Browser-Durchlauf steht oben unter „Ehrliche Grenzen“.


## Quellen und Stufen

Stufen: W Wortlaut gelesen, A nur Abstract, S Suchtreffer oder Zusammenfassung, sonst „nicht gelesen“. Zahlen ohne Quelle heißen „eigene Nachrechnung in diesem Modell“.

| Quelle | Stufe | Wofür |
|---|---|---|
| Coles 2001, *An Introduction to Statistical Modeling of Extreme Values* (Pegel-Formel Gl. 3.4, Delta Gl. 3.11, Profil Gl. 3.12) | W (nur Gleichungen und Befunde in eigenen Worten) | Pegel, Intervalle |
| Embrechts, Klüppelberg, Mikosch 1997, *Modelling Extremal Events* (6.2.4, Bsp. 6.2.19, 8.1.5) | W | Wiederkehrperiode als Mittel, 475-Jahres-Beispiel (P(L ≤ 50) = 0,1 ⇒ T = 475,06 bestätigt), Deich als Lehrbuchbeispiel |
| Huang, Nychka, Zhang, arXiv 1802.09387, Tabelle 1 | W (laut Plan gelesen, im Bau nicht erneut geöffnet) | andere Methode (Log-Histospline, verglichen mit POT/GPD), anderes Niveau (90 %), hier nicht nachgerechnet |
| Huang u. a., arXiv 1512.08775 | A | Warnzeichen: Reihe so lang wie die Wiederkehrperiode oder kürzer |
| genEVA (Buriticá u. a.), arXiv 2401.14910 | S, nach Zusammenfassung, ungeprüft | anderes Problem (Quantilregression, nominell 50 %), nur genannt |
| Rootzén, Katz | W | Wiederkehrperiode gegen Risiko über die Nutzungsdauer |
| Smith 1985, Hosking/Wallis, Fisher/Tippett | nicht gelesen | hier nicht zitiert (die Bedingung ξ ≥ −1 für die Likelihood ist ein Suchtreffer zu Smith 1985) |

Keine Neuheit: alles Lehrbuch; neu ist die Vorführbarkeit mit Zähleinheit, die Überdeckungstabelle und die einseitige Aufteilung der Fehlgriffe. Zahlen ohne Quelle sind eigene Nachrechnungen in diesem Modell.


## Tests

**221 Tests**, serielle Laufzeit der Suite zuletzt 448 Sekunden bei hoher Rechnerlast durch parallele Sitzungen (davon 141 Sekunden der Test `test_check_der_eingefrorenen_messung_als_band`, der drei Zellen neu rechnet); ohne Last war sie früher kürzer (174 Sekunden). Zusammensetzung:

- **Handbeispiele von Hand** (`tests/test_handbeispiele.py`, 12 Tests): Brüche, Aufzählung aller 32 Ausgänge, Tafelwerte, die fünf Jahresmaxima ln 2 bis ln 6.
- **Orakel** (`tests/test_oracle_wiederkehrwert.py`, 71 Tests; scipy nur hier): Pegel-Formel gegen `genextreme.isf` mit c = −ξ, wahre Pegel aus F^n auf zwei Wegen, Simulation der Reihen im Band, Ein-Parameter-Fall (allgemeine Delta- und Profil-Routine gegen geschlossene Endpunkte, Überdeckung gegen die Gamma-Verteilung), Anpassung gegen `genextreme.fit`, Profil-Endpunkte mit Devianz 3,8415 über eine unabhängige Mehrfachstart-Minimierung, Ziehungen gegen die exakten Verteilungen, verteilungsfreie Probe (Beta).
- **Vehikel B** (`tests/test_vehikel_b.py`, 7 Tests): kopierte Tabelle gegen eine eigene Aufzählung aus der Definition, Pegel gegen scipy, Ziehung des Tagesmaximums gegen F^100 und gegen Einzelzüge.
- **Eigenschaften** (`tests/test_properties.py`, 7 Tests), **Randfälle** (`tests/test_edge_cases.py`, 14), **Zahlenwerkzeuge** (`tests/test_numeric.py`, 10), **Kopier-Tests** (`tests/test_copies.py`, 4).
- **Claims und Daten** (`tests/test_claims.py`, 8 Tests; `tests/test_coverage_data.py`, 7): jede README-Zahl belegt, Tabellen wörtlich, eingefrorene Daten gegen die Neurechnung als Band.
- **Verdikte** (`tests/test_verdicts.py`, 12), **Presets** (`tests/test_presets.py`, 7), **Oberfläche**: AppTest (`tests/test_app.py`, 32 Tests, und `tests/test_app_smoke.py`, 3), Layout (`tests/test_layout.py`, 24), Footer (2), Laufzeit-Importe (1).


## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfragen, vier Experimente, Grenzen, Formeln |
| `requirements.txt` | Laufzeit: streamlit, numpy, plotly, pandas (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (scipy nur als zweiter Rechenweg in den Tests) |
| `wkw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `wkw_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `wkw_returnlevel.py` | Pegel, Wartezeit, Reihenlänge, exakte Ein-Parameter-Rechnung (ohne Zufall) |
| `wkw_gev.py` | ML-Anpassung, Delta, Profil, parametrischer Bootstrap (numpy, kein scipy) |
| `wkw_numeric.py` | Nelder-Mead, Brent, Gamma-Verteilung, Wilson-Band |
| `wkw_scenario.py` | Die Blockquellen (GEV, Exponential, Normal, Vehikel B), exakte Pegel, Zufalls-Seeds je Lauf |
| `wkw_vehikel_b.py` | Vehikel B: Mischung aus der Gewichtstabelle, exakter Pegel, Ziehung des Tagesmaximums |
| `wkw_vehikel_b_tabelle.json` | Gewichtstabelle von Vehikel B (wörtliche Kopie der gemeinsamen Definition) |
| `wkw_rng.py` | SplitMix64 (Kopie der Vorlage des Portfolios) |
| `wkw_study.py` | Ein Lauf (Anpassung, Pegel, drei Intervalle) und die Kennzahlen über Läufe |
| `wkw_evaluation.py` | Experimente und Verdikte als reine Funktionen |
| `wkw_reference.py` | lädt die eingefrorenen Daten |
| `wkw_coverage.json` | eingefrorene Messungen (Überdeckung, Streuband, Modellfehler) |
| `wkw_reference.json` | exakte Tabellen und Quellen mit Stufe |
| `wkw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `wkw_layout.py` | Fester Kopf (Stylesheet) |
| `tools/wkw_make_coverage.py` | erzeugt und prüft `wkw_coverage.json` (`--check`, `--lang`) |
| `tools/wkw_make_reference.py` | erzeugt und prüft `wkw_reference.json` (`--check`) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopier-Tests |


## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py --server.port 10085
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html).

"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Alle Blöcke sind ausgedacht (kein Echtdatenbezug). Ein Block ist ein Jahr; „Pegel“ ist der Wiederkehrwert; T ist die Wiederkehrperiode (ein Mittel, keine Frist)."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
T_MIN, T_MAX, DEFAULT_T = 2, 1000, 100
M_MIN, M_MAX, DEFAULT_M = 1, 500, 50                  # Zahl der Blöcke (Jahre) der Reihe
N_MIN, N_MAX, DEFAULT_N = 1, 200, 100                 # Nutzungsdauer in Jahren (Experiment 1)
R1_MIN, R1_MAX, DEFAULT_R1 = 100, 4000, 2000          # gezogene Reihen in Experiment 1; R1 · Reihenlänge ist auf SIM_LIMIT begrenzt
R3_MIN, R3_MAX, DEFAULT_R3 = 10, 150, 100             # Läufe in Experiment 3 (live; darüber nur eingefroren)
R4_MIN, R4_MAX, DEFAULT_R4 = 5, 20, 10                # Läufe der Probe in Experiment 4 (live)
SEED_MAX, DEFAULT_SEED = 999999, 20
SIM_LIMIT = 2_000_000                                 # höchstens so viele gleichverteilte Zahlen in Experiment 1 (rund 3 s)
XI_WERTE = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
DEFAULT_XI = 0.2
FAELLE = ("gev", "exp365", "normal365", "vwz")
DEFAULT_FALL = "gev"
BEKANNT_FREI = "ξ frei (ML mit drei Parametern)"
BEKANNT_SIGMA = "σ und ξ bekannt (nur μ geschätzt)"
BEKANNT_WERTE = (BEKANNT_FREI, BEKANNT_SIGMA)
BOOT_B = 199                                          # Bootstrap-Wiederholungen (wie in den eingefrorenen Zellen)

# --- feste Annahmen -------------------------------------------------------------------------------------------------------------------------
MU_GEV, SIGMA_GEV = 100.0, 20.0                       # Ort und Skala der GEV-Blöcke in der App (die Überdeckung hängt nicht daran, das Streuband von V = x̂/x_T schon)
NOMINAL = 0.95
UEBERDECKUNG_WARNUNG = 0.93                          # unter dieser Überdeckung (Delta oder Profil) zeigt das Verdikt eine Warnung statt eines Hinweises
Q_STUFEN = (0.5, 0.9, 0.95, 0.99)                     # Sicherheiten der Tabelle „Wie lang muss die Reihe sein?“
T_STUFEN = (10, 100, 1000)

COLOR_WAHR = "#2F6B65"
COLOR_SCHAETZ = "#d95f02"
COLOR_DELTA = "#4c78a8"
COLOR_PROFIL = "#7b3294"
COLOR_BOOT = "#8c8c8c"
COLOR_HELL = "rgba(150,150,150,0.55)"

# --- Presets (Schnellstart) -----------------------------------------------------------------------------------------------------------------
# Schlüssel: kurze Namen; was ein Preset nicht nennt, hat den Standardwert.
PRESETS = {
    "100 Jahre in 100 Jahren": {},
    "50 Jahre Reihe, 100er-Wert": {"N": 50, "m": 50, "T": 100},
    "Eine Reihe, drei Intervalle": {"m": 50, "T": 100, "seed": 3},
    "Normal, 1000er-Wert": {"fall": "normal365", "T": 1000},
    "Vehikel B, 1000er-Wert": {"fall": "vwz", "T": 1000},
    "Kurze Reihe m = 20": {"m": 20, "T": 100},
    "σ bekannt": {"xi": 0.0, "m": 20, "T": 1000, "bekannt": BEKANNT_SIGMA},
}
PRESET_HELP = {
    "100 Jahre in 100 Jahren": "T = 100, Nutzungsdauer 100 Jahre, Reihe von 50 Jahren: was die Wiederkehrperiode wirklich heißt (Experiment 1) und wie weit die Schätzung danebenliegt (Experimente 2 bis 4).",
    "50 Jahre Reihe, 100er-Wert": "Nutzungsdauer 50 Jahre: dieselbe Zahl wie die Frage, ob eine 50-Jahres-Reihe den 100er-Wert schon erreicht hat.",
    "Eine Reihe, drei Intervalle": "Eine Reihe aus 50 Jahresmaxima, 100er-Wert, mit Delta-, Profil- und Bootstrap-Intervall (anderer Zufalls-Seed als der Standard).",
    "Normal, 1000er-Wert": "Jahresmaximum aus 365 Normal-Tageswerten, 1000er-Wert: das Modell passt nicht genau, der wahre Pegel kommt exakt aus F^365.",
    "Vehikel B, 1000er-Wert": "Tagesmaximum aus 100 Containern eines ausgedachten Verweilzeit-Modells, 1000er-Wert: der Median des Schätzwerts liegt hier beim 1,20-Fachen des wahren Pegels (eingefroren, 1 000 Läufe), bei Exponential und Normal darunter.",
    "Kurze Reihe m = 20": "Nur 20 Jahre Reihe (ξ = 0,2, T = 100): das Delta-Intervall deckt in der eingefrorenen Messung nur 80,0 % der Läufe, das Profil-Intervall ist 2,6-mal so breit wie Delta (Experiment 4).",
    "σ bekannt": "Gumbel mit bekannter Skala (ξ = 0): nur die Lage wird geschätzt; dann sind Überdeckung und Fehlerband exakt berechenbar (Gamma-Verteilung) und der Fehler in Einheiten σ hängt nicht an T; das Streudiagramm zeigt die live gezogenen Läufe.",
}

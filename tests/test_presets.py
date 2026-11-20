"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, paarweise verschieden; Einstellungen kommen nur aus SETTING_SPECS."""

import wkw_constants as C
import wkw_evaluation as ev
import wkw_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[name] for name in C.PRESETS)


def test_preset_keys_are_known_and_values_are_valid():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            elif isinstance(spec.default, bool):
                assert isinstance(value, bool)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}                                            # nennt nichts: alles Standard
    assert ev.Settings() == ev.Settings(**{k: v for k, v in C.PRESETS[first].items()})
    for state_key, spec in P.SETTING_SPECS.items():
        assert spec.default is not None


def test_presets_are_pairwise_different():
    def wert(preset):
        return tuple(preset.get(k, P.SETTING_SPECS[sk].default) for k, sk in P.PRESET_KEYS.items())
    liste = [wert(p) for p in C.PRESETS.values()]
    assert len(set(liste)) == len(liste)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f for f in ev.Settings.__dataclass_fields__}      # jedes Feld von Settings ist ein Regler und umgekehrt


def test_defaults_lie_inside_the_bounds_and_choices():
    for spec in P.SETTING_SPECS.values():
        if spec.choices is not None:
            assert spec.default in spec.choices
        elif spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi


def test_preset_names_match_their_content():
    """Namen gegen Werte lesen: „Normal“ = Fall normal365, „Vehikel B“ = vwz, „1000er“ = T = 1000, „m = 20“ = 20 Blöcke, „σ bekannt“ = bekannter Fall."""
    P_ = C.PRESETS
    assert P_["Normal, 1000er-Wert"]["fall"] == "normal365" and P_["Normal, 1000er-Wert"]["T"] == 1000
    assert P_["Vehikel B, 1000er-Wert"]["fall"] == "vwz" and P_["Vehikel B, 1000er-Wert"]["T"] == 1000
    assert P_["Kurze Reihe m = 20"]["m"] == 20
    assert P_["σ bekannt"]["bekannt"] == C.BEKANNT_SIGMA and P_["σ bekannt"]["xi"] == 0.0
    assert P_["50 Jahre Reihe, 100er-Wert"]["N"] == 50 and P_["50 Jahre Reihe, 100er-Wert"]["m"] == 50
    assert P_["100 Jahre in 100 Jahren"] == {} and ev.Settings().T == 100 and ev.Settings().N == 100

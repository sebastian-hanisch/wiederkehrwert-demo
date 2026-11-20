"""Gemeinsame Fixtures und Hilfen: Handbeispiel-Daten, Zufalls-Seeds und die Pfade der Demo."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import wkw_scenario as S  # noqa: E402
from wkw_rng import SplitMix64  # noqa: E402

WURZEL = Path(__file__).resolve().parent.parent


@pytest.fixture
def maxima_h4():
    """Fünf ausgedachte Jahresmaxima x_i = ln 2, ln 3, ln 4, ln 5, ln 6 (Handbeispiel H4: exp(-x_i) = 1/2, 1/3, 1/4, 1/5, 1/6)."""
    import math
    return [math.log(k) for k in (2, 3, 4, 5, 6)]


def stichprobe(art: str, xi: float, m: int, *seed_teile, mu: float = 0.0, sigma: float = 1.0):
    """m Blöcke aus dem Fall `art` mit eigenem SplitMix64-Strom (für Tests)."""
    rng = SplitMix64(S.stream_seed(*seed_teile))
    return np.array(S.ziehe_bloecke(S.Fall(art, xi, mu, sigma), rng, m), float)

"""Cross-checks of the TF models against reference values from the C++ generator.

inputs/belle_cache.root holds the Belle D0 -> KS pi pi amplitude computed by the
C++ code in d02kspipi_toys on a 500x500 grid in (m2(KS pi+), m2(KS pi-)).
"""

import numpy as np
import pytest
import uproot

import amplitf.interface as atfi

from tfaex import paths
from tfaex.io import load_belle_model
from tfaex.models import belle2018
from tfaex.models.d02kspipi import dalitz_phase_space


@pytest.mark.xfail(
    strict=True,
    reason="belle2018.amplitude differs from the C++ cache by O(100%) after removing a "
    "global complex factor, for every axis/conjugation convention tried; under study "
    "in macros/strong_phases.py",
)
def test_belle2018_matches_cpp_cache():
    with uproot.open(paths.BELLE_CACHE) as f:
        amp, xbins, ybins = f["histo_amplitude"].to_numpy()
        phase, _, _ = f["histo_phase"].to_numpy()
    xc = 0.5 * (xbins[:-1] + xbins[1:])
    yc = 0.5 * (ybins[:-1] + ybins[1:])
    # every 5th bin is enough for the comparison
    X, Y = np.meshgrid(xc[::5], yc[::5], indexing="ij")
    ref = (amp * np.exp(1j * phase))[::5, ::5].ravel()
    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    phsp = dalitz_phase_space()
    inside = phsp.inside(atfi.const(points)).numpy() & (np.abs(ref) > 0)
    tf_amp = belle2018.amplitude(atfi.const(points[inside]), phsp, load_belle_model()).numpy()

    ratio = tf_amp / ref[inside]
    ratio /= np.median(ratio)  # conventions may differ by a global complex factor
    assert np.median(np.abs(ratio - 1)) < 0.01

"""Cross-check of the TF Belle model against the cached C++ amplitude used by the toys.

inputs/belle_cache.root holds the Belle D0 -> KS pi pi amplitude computed by the
C++ code in d02kspipi_toys on a 500x500 grid in (m2(KS pi+), m2(KS pi-)).
"""

import numpy as np
import uproot

import amplitf.interface as atfi

from tfaex import paths
from tfaex.io import load_belle_model
from tfaex.models import belle2018
from tfaex.models.d02kspipi import dalitz_phase_space


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
    # The cache x axis is m2(KS pi-): make_amplitude_hists.cpp fills it with
    # calc_amp(x, y), whose first argument is m2(KS pi-). The residual differences
    # come from the K-matrix multi-meson phase space below 1 GeV^2 (see
    # test_cpp_reference.py) and from the PDG masses used on the TF side.
    assert np.median(np.abs(ratio - 1)) < 5e-3
    assert abs(np.median(ratio.real) - 1) < 5e-3 and abs(np.median(ratio.imag)) < 5e-3

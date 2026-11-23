"""Component-by-component comparison with the Belle C++ implementation.

Reference values: inputs/d02kspipi_cpp_reference.npz (macros/make_cpp_reference.py).
"""

import numpy as np
import pytest

from tfaex.models.d02kspipi import BABAR2008_COMPONENTS
from tfaex.reference import load_cpp_reference, tf_components

REF = load_cpp_reference()
TF = tf_components(REF)
BELOW_1GEV2 = REF["m2_pipi"] <= 1.0

EXACT = [c for c in BABAR2008_COMPONENTS if c in REF and c != "Kmatrix"]


def rel_dev(a, b):
    return np.abs(a - b) / np.abs(b)


def test_reference_contents():
    # every C++ term has a TF counterpart; the TF model has two extra terms
    cpp_terms = {k for k in REF if k in BABAR2008_COMPONENTS}
    assert set(BABAR2008_COMPONENTS) - cpp_terms == {"Kstar1680plus", "nonres"}
    np.testing.assert_allclose(sum(REF[k] for k in cpp_terms), REF["total"], rtol=1e-12)


@pytest.mark.parametrize("name", EXACT)
def test_component(name):
    assert rel_dev(TF[name], REF[name]).max() < 1e-9


def test_kmatrix_above_1gev2():
    sel = ~BELOW_1GEV2
    # The TF side uses the PDG eta mass (0.547862) instead of 0.54775 as in the
    # C++ code: this moves the eta-eta threshold (s ~ 1.20 GeV^2), where the
    # difference peaks at ~0.5%; elsewhere it is < 1e-3.
    assert rel_dev(TF["Kmatrix"][sel], REF["Kmatrix"][sel]).max() < 1e-2


@pytest.mark.xfail(
    strict=True,
    reason="AmpliTF kmatrix_phsp_multimeson uses a different 4pi phase-space "
    "approximation below 1 GeV^2 than Resonance.cpp (up to ~7% on the K-matrix)",
)
def test_kmatrix_below_1gev2():
    sel = BELOW_1GEV2
    assert rel_dev(TF["Kmatrix"][sel], REF["Kmatrix"][sel]).max() < 1e-3


def test_unused_components_are_zero():
    # Kstar1680plus is not in the C++ model; its Belle coupling is zero
    assert np.all(TF["Kstar1680plus"] == 0)


def test_total_above_1gev2():
    sel = ~BELOW_1GEV2
    assert np.median(rel_dev(TF["total"][sel], REF["total"][sel])) < 1e-4

"""Component-by-component comparison with the Belle C++ implementation.

Reference values: inputs/d02kspipi_cpp_reference.npz (macros/make_cpp_reference.py).
"""

import numpy as np
import pytest

from tfaex.models import belle2018
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


def test_kmatrix_with_cpp_masses():
    """With the K-matrix channel masses of the C++ code the K-matrix is reproduced exactly."""
    tf = tf_components(REF, kmatrix_masses=belle2018.CPP_KMATRIX_MASSES)
    dev = rel_dev(tf["Kmatrix"], REF["Kmatrix"])
    assert np.median(dev) < 1e-12
    assert dev.max() < 1e-4
    total = rel_dev(tf["total"], REF["total"])
    assert np.median(total) < 1e-12
    assert total.max() < 1e-4


def test_kmatrix_needs_belle_multimeson():
    """With the AmpGen 4pi phase space the K-matrix differs from Belle below 1 GeV^2 only."""
    tf = tf_components(REF, kmatrix_masses=belle2018.CPP_KMATRIX_MASSES, kmatrix_multimeson="ampgen")
    dev = rel_dev(tf["Kmatrix"], REF["Kmatrix"])
    assert dev[BELOW_1GEV2].max() > 0.03
    assert dev[~BELOW_1GEV2].max() < 1e-6


def test_kmatrix_with_pdg_masses():
    # With the PDG eta mass (0.547862 instead of 0.54775) the eta-eta channel
    # differs, mostly at its threshold (s ~ 1.20 GeV^2): ~0.5% there, < 0.2% below 1 GeV^2.
    assert rel_dev(TF["Kmatrix"][BELOW_1GEV2], REF["Kmatrix"][BELOW_1GEV2]).max() < 5e-3
    assert rel_dev(TF["Kmatrix"][~BELOW_1GEV2], REF["Kmatrix"][~BELOW_1GEV2]).max() < 1e-2


def test_unused_components_are_zero():
    # Kstar1680plus is not in the C++ model; its Belle coupling is zero
    assert np.all(TF["Kstar1680plus"] == 0)


def test_total_with_pdg_masses():
    dev = rel_dev(TF["total"], REF["total"])
    assert np.median(dev) < 1e-3

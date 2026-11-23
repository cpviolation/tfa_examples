"""Smoke tests for the D0 -> KS pi pi models and helpers."""

import numpy as np
import pytest

import amplitf.interface as atfi
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from particle.particle import literals as lp

from tfaex import paths
from tfaex.io import add_binflip_binid, load_belle_model
from tfaex.models import belle2018
from tfaex.models.couplings import coherent_sum, switches_for
from tfaex.models.d02kspipi import (
    BABAR2008_COMPONENTS,
    babar2008_components,
    babar2008_model,
    babar2008_model_amp,
    belle_cache_amp,
)


@pytest.fixture(scope="module")
def phsp():
    mkz = atfi.const(lp.K_S_0.mass / 1000)
    mpi = atfi.const(lp.pi_plus.mass / 1000)
    md = atfi.const(lp.D_0.mass / 1000)
    return DalitzPhaseSpace(mpi, mkz, mpi, md)


@pytest.fixture(scope="module")
def sample(phsp):
    atfi.set_seed(1)
    return phsp.uniform_sample(2000)


@pytest.fixture(scope="module")
def belle_model():
    return load_belle_model()


def test_input_files_exist():
    for f in (paths.BELLE_MODEL, paths.BELLE_MODEL_DEFINITIONS, paths.BELLE_CACHE, paths.BINFLIP_LUT):
        assert f.is_file(), f


def test_load_belle_model(belle_model):
    assert belle_model["Kstar892_Mass"][0] == pytest.approx(0.8937, abs=0.01)
    assert "Kmatrix_beta1_realpart" in belle_model


def test_belle2018_amplitude_finite(phsp, sample, belle_model):
    amp = belle2018.amplitude(sample, phsp, belle_model).numpy()
    assert amp.shape == (sample.shape[0],)
    assert np.all(np.isfinite(amp))
    assert np.any(np.abs(amp) > 0)


def _legacy_kwargs(couplings):
    names = BABAR2008_COMPONENTS[:14]
    kwargs = {}
    for i, n in enumerate(names, start=1):
        kwargs[f"a{i}r"] = atfi.real(couplings[n])
        kwargs[f"a{i}i"] = atfi.imaginary(couplings[n])
    return kwargs


def test_legacy_interface_matches_named(phsp, sample, belle_model):
    """babar2008_model_amp(a1r, ...) and the named interface give the same amplitude."""
    args = belle2018.lineshape_args(belle_model)
    cpl = belle2018.couplings(belle_model)
    named = belle2018.amplitude(sample, phsp, belle_model).numpy()
    legacy = babar2008_model_amp(sample, phsp, *args)(**_legacy_kwargs(cpl)).numpy()
    np.testing.assert_allclose(legacy, named, rtol=1e-12)
    # real parameters <name>_re / <name>_im, as passed by a TFA2 likelihood
    params = {}
    for n, c in cpl.items():
        params[f"{n}_re"], params[f"{n}_im"] = atfi.real(c), atfi.imaginary(c)
    from_params = belle2018.amplitude_model(sample, phsp, belle_model)(**params).numpy()
    np.testing.assert_allclose(from_params, named, rtol=1e-12)


def test_density_is_amplitude_squared(phsp, sample, belle_model):
    """babar2008_model must stay equal to |babar2008_model_amp|^2."""
    amp = belle2018.amplitude(sample, phsp, belle_model).numpy()
    kwargs = _legacy_kwargs(belle2018.couplings(belle_model))
    density = babar2008_model(sample, phsp, *belle2018.lineshape_args(belle_model))(**kwargs).numpy()
    np.testing.assert_allclose(density, np.abs(amp) ** 2, rtol=1e-10)


def test_switches_select_components(phsp, sample, belle_model):
    comps = babar2008_components(sample, phsp, *belle2018.lineshape_args(belle_model))
    cpl = belle2018.couplings(belle_model)
    only = switches_for(BABAR2008_COMPONENTS, ["Kstar892minus"])
    amp = belle2018.amplitude(sample, phsp, belle_model, switches=only).numpy()
    np.testing.assert_allclose(amp, (cpl["Kstar892minus"] * comps["Kstar892minus"]).numpy(), rtol=1e-12)
    with pytest.raises(KeyError):
        switches_for(BABAR2008_COMPONENTS, ["not_a_component"])
    with pytest.raises(KeyError):
        coherent_sum(comps, {}, only)


def test_belle_cache_amp(phsp, sample):
    amp = belle_cache_amp(sample, phsp).numpy()
    assert amp.shape == (sample.shape[0],)
    assert np.all(np.isfinite(amp))


def test_mixing_density_positive(phsp, sample):
    amp = belle_cache_amp(sample, phsp)
    ampb = belle_cache_amp(sample[:, ::-1], phsp)
    t = atfi.const(np.linspace(0.1, 5.0, sample.shape[0]))
    one = atfi.const(1.0)
    dens = atfm.mixing_density(
        amp, ampb, atfi.complex(one, atfi.const(0.0)),
        atfm.psip(t, atfi.const(0.0064), one),
        atfm.psim(t, atfi.const(0.0064), one),
        atfm.psii(t, atfi.const(0.004), one),
    ).numpy()
    assert np.all(dens >= 0)


def test_add_binflip_binid(sample):
    data = add_binflip_binid(sample.numpy())
    assert data.shape == (sample.shape[0], 3)
    assert set(np.unique(np.abs(data[:, 2]))) <= set(range(0, 9))

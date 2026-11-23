"""Decay-time distribution of the mixing toys and an end-to-end fit of x and y."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import amplitf.interface as atfi
import amplitf.likelihood as atfl
import tfa.optimisation as tfo

MACRO = Path(__file__).resolve().parents[1] / "macros" / "d02kspipi_toys.py"


@pytest.fixture(scope="module")
def toys():
    spec = importlib.util.spec_from_file_location("d02kspipi_toys", MACRO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_mixing_is_exponential(toys):
    t = toys.generate(50000, x_mix=0.0, y_mix=0.0, qop=1.0, qop_phase=0.0, seed=3)[:, 2]
    # exp(-t/tau) with tau = 1: mean 1 and std 1 (previously exp(-2t): mean 0.5)
    assert t.mean() == pytest.approx(1.0, abs=0.02)
    assert t.std() == pytest.approx(1.0, abs=0.03)


def test_relative_density_times_exponential_is_full_density(toys):
    x = toys.c_phsp.uniform_sample(1000)
    pars = dict(x_mix_par=atfi.const(0.05), y_mix_par=atfi.const(0.05),
                qoverp_re=atfi.const(0.9), qoverp_im=atfi.const(0.2))
    rel = toys.mixing_model(x, relative_to_decay=True)(**pars).numpy()
    full = toys.mixing_model(x, relative_to_decay=False)(**pars).numpy()
    t = x.numpy()[:, 2]
    np.testing.assert_allclose(rel * np.exp(-t), full, rtol=1e-12)


def test_fit_recovers_mixing_parameters(toys):
    """Toys and normalisation both use DecayTimePhaseSpace samples (exponential in t)."""
    true = {"x_mix_par": 0.05, "y_mix_par": 0.05}  # exaggerated to get a precise check
    data = atfi.const(toys.generate(50000, x_mix=true["x_mix_par"], y_mix=true["y_mix_par"],
                                    qop=1.0, qop_phase=0.0, seed=4))
    norm = toys.c_phsp.uniform_sample(300000)
    data_model, norm_model = toys.mixing_model(data), toys.mixing_model(norm)
    fixed = {"qoverp_re": atfi.const(1.0), "qoverp_im": atfi.const(0.0)}

    @atfi.function
    def nll(pars):
        p = dict(pars, **fixed)
        return atfl.unbinned_nll(data_model(**p), atfl.integral(norm_model(**p)))

    pars = [tfo.FitParameter("x_mix_par", 0.0, -0.3, 0.3),
            tfo.FitParameter("y_mix_par", 0.0, -0.3, 0.3)]
    result = tfo.run_minuit(nll, pars)
    for name, value in true.items():
        fitted, error = result["params"][name]
        assert abs(fitted - value) < 5 * error, (name, fitted, error)

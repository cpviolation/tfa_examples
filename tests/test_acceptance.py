import numpy as np
import pytest

from tfaex.acceptance import apply_efficiency, decay_time_efficiency

PARS = dict(t0=0.3, a=1.5, b=2.0, m=1.0, n=1.0)


def test_efficiency_shape():
    t = np.array([0.0, 0.3, 0.5, 1.0, 5.0])
    eff = decay_time_efficiency(t, **PARS)
    assert np.all(eff[:2] == 0)
    dt = t[2:] - PARS["t0"]
    np.testing.assert_allclose(eff[2:], dt**1.5 / (2.0 + dt))


def test_apply_efficiency_fraction():
    rng = np.random.default_rng(0)
    data = np.column_stack([np.zeros(200000), np.zeros(200000), rng.exponential(1.0, 200000)])
    eff = decay_time_efficiency(data[:, 2], **PARS)
    kept = apply_efficiency(data, eff, rng=1)
    assert len(kept) / len(data) == pytest.approx(eff.mean() / eff.max(), rel=0.02)
    assert np.all(kept[:, 2] > PARS["t0"])
    with pytest.raises(ValueError):
        apply_efficiency(data, np.zeros(len(data)))


def test_acceptance_macro_matches_exploratory_function():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "macros" / "acceptance.py"
    spec = importlib.util.spec_from_file_location("acceptance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = np.linspace(0, 10, 101)
    t0, a, b, beta, m, n = 0.335, 1.14, 6.73, 1.27, 2.97, 0.17
    dt = t - t0
    expected = np.where(dt >= 0, (np.clip(dt, 0, None)**a / (b + np.clip(dt, 0, None)**n))**m * np.exp(-beta * dt), 0)
    np.testing.assert_allclose(mod.dt_acceptance(t, t0, a, b, beta, m, n), expected, rtol=1e-12, atol=0)

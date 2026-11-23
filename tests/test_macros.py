"""The macros can be imported without side effects and their steps run on a small sample."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MACROS = Path(__file__).resolve().parents[1] / "macros"


def load(name):
    spec = importlib.util.spec_from_file_location(name, MACROS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["d02kspipi_toys", "binflipfitter", "strong_phases", "acceptance"])
def test_import(name):
    load(name)


@pytest.mark.parametrize("name", ["d02kspipi_toys", "binflipfitter", "strong_phases"])
def test_dryrun(name, capsys):
    load(name).main(["--dryrun"])
    assert "Namespace" in capsys.readouterr().out


def test_toys_and_binflip():
    toys = load("d02kspipi_toys")
    sample = toys.generate(5000, x_mix=0.004, y_mix=0.0064, qop=1.0, qop_phase=0.0, seed=1)
    assert sample.shape == (5000, 3)
    assert np.all(sample[:, 2] > 0)
    # both Dalitz coordinates inside the kinematic range
    assert np.all((sample[:, :2] > 0.3) & (sample[:, :2] < 3.1))

    binflip = load("binflipfitter")
    data = binflip.add_binflip_binid(binflip.order_dataset(sample))
    ratios, errors = binflip.calculate_ratios(data, nbins=5)
    assert ratios.shape == (8, 5)
    assert np.all(np.isfinite(errors))

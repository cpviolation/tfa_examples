"""D0 -> K+ K- pi+ pi- model defined with AmpGen, against amplitudes computed with AmpGen.

Reference: inputs/ampgen_d02kkpipi_stage1_reference.npz, produced with
macros/ampgen/DumpTerms.cpp from params/ampgen/DtoKKpipi_stage1.opt (2000
phase-space events; amplitude of every decay chain without couplings, couplings,
total amplitude).
"""

import cmath

import numpy as np
import pytest

import amplitf.interface as atfi
import amplitf.likelihood as atfl
import tfa.optimisation as tfo
import tfa.toymc as tft

from tfaex import paths
from tfaex.ampgen import parse_descriptor, read_options
from tfaex.models import d02kkpipi

REF = np.load(paths.INPUTS_DIR / "ampgen_d02kkpipi_stage1_reference.npz")
MODEL = d02kkpipi.load_model()
EVENTS = [atfi.const(REF["events"][:, 4 * i:4 * i + 4]) for i in range(4)]
REF_NAMES = list(REF["names"])


def rms(x):
    return np.sqrt(np.mean(np.abs(x) ** 2))


def test_read_options():
    opts = read_options(d02kkpipi.OPTIONS)
    assert opts.event_type == ["D0", "K+", "K-", "pi+", "pi-"]
    params = {k: v for k, (v, _) in opts.parameters.items()}
    assert params["D0_radius"] == 1.21
    assert params["K*(892)bar0_radius"] == params["K*(892)0_radius"] == 1.13  # alias
    assert params["a(1)(1260)-_mass"] == params["a(1)(1260)+_mass"] == 1.19505
    c, free = opts.couplings["D0[D]{K*(892)0{K+,pi-},K*(892)bar0{K-,pi+}}"]
    assert c == pytest.approx(cmath.rect(0.693083, 2.85743)) and free  # polar, radians
    assert not opts.couplings["D0{phi(1020)0{K+,K-},rhoOmega00}"][1]  # fixed reference


def test_descriptor_round_trip():
    text = "D0[D]{K*(1680)0{K+,pi-},K*(892)bar0{K-,pi+}}"
    assert parse_descriptor(text).descriptor() == text


def test_chains_match_ampgen():
    """Expansion of pseudo-particles, daughter ordering and orbital labels as in AmpGen."""
    assert sorted(MODEL.chain_names) == sorted(REF_NAMES)


@pytest.mark.parametrize("index", range(len(REF_NAMES)))
def test_chain_amplitude(index):
    name = REF_NAMES[index]
    chain = next(c for c in MODEL.chains if c.name == name)
    amp = chain.particle.amplitude(EVENTS, MODEL.mpi).numpy()
    ref = REF["amplitudes"][index]
    # AmpGen evaluates its amplitudes with fast-math: agreement ~1e-6 - 1e-5 of the rms
    assert np.max(np.abs(amp - ref)) < 1e-4 * rms(ref)
    assert MODEL.chain_coupling(chain) == pytest.approx(REF["couplings"][index], abs=1e-12)


def test_total_amplitude():
    total = MODEL.amplitude(EVENTS).numpy()
    assert np.max(np.abs(total - REF["total"])) < 1e-4 * rms(REF["total"])


def test_phase_space_momenta():
    phsp = d02kkpipi.phase_space(MODEL)
    x = phsp.uniform_sample(1000)
    p = [v.numpy() for v in d02kkpipi.momenta(phsp, x)]
    masses = [MODEL.table[n]["mass"] for n in d02kkpipi.FINAL_STATE]
    for pi, m in zip(p, masses):
        np.testing.assert_allclose(pi[:, 3] ** 2 - np.sum(pi[:, :3] ** 2, axis=1), m * m, rtol=1e-9)
    total = sum(p)
    np.testing.assert_allclose(total[:, :3], 0, atol=1e-12)
    np.testing.assert_allclose(total[:, 3], MODEL.table["D0"]["mass"], rtol=1e-12)


def test_small_fit():
    """Toys from the model; fit two couplings with the others fixed."""
    phsp = d02kkpipi.phase_space(MODEL)

    def pdf(x):
        return d02kkpipi.pdf_model(MODEL, phsp, x)()

    data = tft.run_toymc(pdf, phsp, 5000, maximum=1.0e-20, chunk=200000, seed=3, components=False)
    norm = phsp.uniform_sample(200000)
    rules = ["D0{K*(892)0{K+,pi-},K*(892)bar0{K-,pi+}}", "D0{K(1)(1400)+{K*(892)0{K+,pi-},pi+},K-}"]
    pars, names = d02kkpipi.fit_parameters(MODEL, rules)
    for p in pars:
        p.update(p.init_value * 1.2)
    data_pdf, norm_pdf = d02kkpipi.pdf_model(MODEL, phsp, data), d02kkpipi.pdf_model(MODEL, phsp, norm)

    @atfi.function
    def nll(values):
        c = d02kkpipi.couplings_from_parameters(values, names)
        return atfl.unbinned_nll(data_pdf(c), atfl.integral(norm_pdf(c)))

    result = tfo.run_minuit(nll, pars)
    for name, rule in names.items():
        true = MODEL.couplings[rule]
        for part, t in (("re", true.real), ("im", true.imag)):
            v, e = result["params"][f"{name}_{part}"]
            assert abs(v - t) < 5 * e, (rule, part, v, e, t)

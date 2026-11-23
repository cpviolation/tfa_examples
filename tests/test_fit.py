"""End-to-end check of the named-coupling interface with TFA2 toys and Minuit."""

import numpy as np

import amplitf.dynamics as atfd
import amplitf.interface as atfi
import amplitf.likelihood as atfl
import tfa.optimisation as tfo
import tfa.toymc as tft
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from particle.particle import literals as lp

from tfaex.io import load_belle_model
from tfaex.models import belle2018
from tfaex.models.couplings import couplings_from_params, fit_parameters

FLOATED = "Kstar892minus"


def test_fit_recovers_coupling():
    phsp = DalitzPhaseSpace(
        atfi.const(lp.pi_plus.mass / 1000), atfi.const(lp.K_S_0.mass / 1000),
        atfi.const(lp.pi_plus.mass / 1000), atfi.const(lp.D_0.mass / 1000),
    )
    belle_model = load_belle_model()
    true = belle2018.couplings(belle_model)

    def pdf(x, switches=belle2018.DEFAULT_SWITCHES):
        return atfd.density(belle2018.amplitude(x, phsp, belle_model, switches))

    atfi.set_seed(5)
    data = tft.run_toymc(pdf, phsp, 20000, maximum=1.0e-20, chunk=200000, components=False)
    norm = phsp.uniform_sample(200000)

    data_model = belle2018.amplitude_model(data, phsp, belle_model)
    norm_model = belle2018.amplitude_model(norm, phsp, belle_model)
    fixed = {n: c for n, c in true.items() if n != FLOATED}

    @atfi.function
    def nll(pars):
        cpl = dict(fixed, **couplings_from_params(pars, [FLOATED]))
        return atfl.unbinned_nll(
            atfd.density(data_model(couplings=cpl)),
            atfl.integral(atfd.density(norm_model(couplings=cpl))),
        )

    start = 0.8 * complex(true[FLOATED].numpy())
    result = tfo.run_minuit(nll, fit_parameters({FLOATED: start}))
    for part, f in (("re", np.real), ("im", np.imag)):
        value, error = result["params"][f"{FLOATED}_{part}"]
        assert abs(value - f(true[FLOATED].numpy())) < 5 * error

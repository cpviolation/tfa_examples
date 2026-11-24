"""D0 -> K+ K- pi+ pi- model as defined in AmpGen (LHCb, JHEP 02 (2019) 126).

The model is read from ``params/ampgen/DtoKKpipi_v2.opt`` (AmpGen's
``options/DtoKKpipi_v2.opt``, all 55 decay chains) with
:class:`tfaex.ampgen.AmpGenModel`.
The phase space is :class:`FourBodyHelicityPhaseSpace` with pairs
(K+ K-)(pi+ pi-), sampled uniformly in (m(KK), m(pipi), cos theta_K,
cos theta_pi, phi). That is not uniform in phase space: the PDF given to TFA2
(toys and normalisation integrals) is |A|^2 times the phase-space density
``phsp.density(x)``.
"""

import tensorflow as tf

import amplitf.interface as atfi
import tfa.optimisation as tfo
from amplitf.phasespace.four_body_helicity_phasespace import FourBodyHelicityPhaseSpace

from tfaex import paths
from tfaex.ampgen import AmpGenModel

OPTIONS = paths.PARAMS_DIR / "ampgen" / "DtoKKpipi_v2.opt"
PARTICLES = paths.PARAMS_DIR / "ampgen" / "particles.json"
FINAL_STATE = ["K+", "K-", "pi+", "pi-"]


def load_model(options=OPTIONS, particles=PARTICLES):
    model = AmpGenModel(options, particles)
    if model.final_state != FINAL_STATE:
        raise ValueError(f"Expected final state {FINAL_STATE}, got {model.final_state}")
    return model


def phase_space(model):
    """Four-body phase space with AmpGen's masses; momenta come out as (K+, K-, pi+, pi-)."""
    table = model.table
    mk, mpi = atfi.const(table["K+"]["mass"]), atfi.const(table["pi+"]["mass"])
    md = atfi.const(table[model.head]["mass"])
    return FourBodyHelicityPhaseSpace(mk, mk, mpi, mpi, md)


def momenta(phsp, x):
    return list(phsp.final_state_momenta(x))


def pdf_model(model, phsp, x):
    """Function of the couplings ({rule: complex}) returning |A|^2 x phase-space density at x.

    Chain amplitudes are computed once; only the couplings change between calls.
    """
    p = momenta(phsp, x)
    amplitudes = model.chain_amplitudes(p)
    density = phsp.density(x)

    def _pdf(couplings=None):
        a = model.amplitude(p, couplings, amplitudes)
        return (atfi.real(a) ** 2 + atfi.imaginary(a) ** 2) * density

    return _pdf


def compiled_pdf(model, phsp, couplings=None):
    """|A|^2 x phase-space density as a function of the points x, compiled once
    (``tf.function`` with an unknown number of points). Evaluating the 55 decay
    chains in graph mode is ~30 times faster than eagerly, which matters for toy
    generation (many chunks of points); the first call takes ~20 s to trace."""
    spec = tf.TensorSpec([None, phsp.dimensionality()], atfi.fptype())

    @tf.function(input_signature=[spec])
    def _pdf(x):
        return pdf_model(model, phsp, x)(couplings)

    return _pdf


def generate(model, phsp, size, couplings=None, seed=None):
    """``size`` events generated from the model (``couplings`` overriding the options file)."""
    import tfa.toymc as tft

    # maximum=1e-20: run_toymc finds the maximum itself
    return tft.run_toymc(compiled_pdf(model, phsp, couplings), phsp, size, maximum=1.0e-20, chunk=200000,
                         seed=seed, components=False)


def normalisation_sample(model, phsp, size, couplings=None, seed=None, uniform=False):
    """Points for the normalisation integral and the weights to apply to the PDF.

    By default the points are generated from the model itself (with ``couplings``
    overriding the options file), i.e. importance sampling: the integral of the
    PDF is then the mean of ``pdf(x) * weight`` with weight = 1 / pdf_gen(x). This
    is much more precise than uniform points for a model with narrow structures
    (phi(1020), K*(892)). With ``uniform=True`` the points are uniform in the
    phase-space variables and the weights are 1.
    """
    if uniform:
        x = phsp.uniform_sample(size)
        return x, atfi.ones(x[:, 0])
    x = generate(model, phsp, size, couplings, seed)
    return x, 1.0 / pdf_model(model, phsp, x)(couplings)


def fit_parameters(model, rules, limit=10.0):
    """Cartesian fit parameters ``<name>_re``/``<name>_im`` for the given coupling rules,
    starting from the values in the options file. Returns (parameters, {name: rule})."""
    pars, names = [], {}
    for rule in rules:
        name = model.parameter_name(rule)
        names[name] = rule
        c = model.couplings[rule]
        pars += [tfo.FitParameter(name + "_re", c.real, -limit, limit),
                 tfo.FitParameter(name + "_im", c.imag, -limit, limit)]
    return pars, names


def couplings_from_parameters(pars, names):
    """{rule: complex tensor} from fit parameter values ``<name>_re``/``<name>_im``."""
    return {rule: atfi.complex(pars[name + "_re"], pars[name + "_im"]) for name, rule in names.items()}

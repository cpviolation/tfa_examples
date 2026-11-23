"""D0 -> KS pi+ pi- amplitude with the Belle 2018 parameters.

Uses the BaBar 2008 parameterisation (:func:`babar2008_components`) with the
resonance parameters and couplings read from ``params/belle_model.txt``.
"""

import amplitf.interface as atfi
from particle.particle import literals as lp

from tfaex.models.d02kspipi import (
    BABAR2008_COMPONENTS,
    BABAR2008_DEFAULT_SWITCHES,
    BABAR2008_FIXED,
    babar2008_named_model_amp,
)

#: Default component switches (see ``BABAR2008_COMPONENTS``): constant term off.
DEFAULT_SWITCHES = BABAR2008_DEFAULT_SWITCHES

#: Components with a free complex coupling; rho(770) is the reference, fixed to 1.
FREE_COMPONENTS = tuple(c for c in BABAR2008_COMPONENTS if c not in BABAR2008_FIXED)
REFERENCE = "rho770"


def lineshape_args(belle_model):
    """Positional arguments of :func:`babar2008_components` after ``(x, phsp)``."""
    # the KKbar channel of the K-matrix uses the charged kaon mass, as in the Belle code
    mk = atfi.const(lp.K_plus.mass / 1000.0)
    mpi = atfi.const(lp.pi_plus.mass / 1000.0)
    meta = atfi.const(lp.eta.mass / 1000.0)
    metap = atfi.const(lp.etap_958.mass / 1000.0)

    def p(name):
        return atfi.const(belle_model[name][0])

    def cplx(prefix):
        return atfi.stack(
            [
                belle_model[f"{prefix}{i}_realpart"][0]
                + 1.0j * belle_model[f"{prefix}{i}_imaginarypart"][0]
                for i in range(1, 6)
            ]
        )

    return [
        p("rho770_Mass"), p("rho770_Width"),
        p("Kstar892_Mass"), p("Kstar892_Width"),
        p("Kstartwo1430_Mass"), p("Kstartwo1430_Width"),
        p("Kstar1410_Mass"), p("Kstar1410_Width"),
        p("Kstar1680_Mass"), p("Kstar1680_Width"),
        p("omega_Mass"), p("omega_Width"),
        p("ftwo1270_Mass"), p("ftwo1270_Width"),
        p("rho1450_Mass"), p("rho1450_Width"),
        # LASS
        p("LASS_a"), p("LASS_r"),
        p("Kstarzero1430_Mass"), p("Kstarzero1430_Width"),
        p("LASS_R"), p("LASS_phi_R"),
        p("LASS_F"), p("LASS_phi_F"),
        # K-matrix
        atfi.const([0.651, 1.2036, 1.55817, 1.21, 1.82206]),
        atfi.const([[0.22889, -0.55377, 0, -0.39899, -0.34639],
                    [0.94128, 0.55095, 0, 0.39065, 0.31503],
                    [0.36856, 0.23888, 0.55639, 0.18340, 0.18681],
                    [0.33650, 0.40907, 0.85679, 0.19906, -0.00984],
                    [0.18171, -0.17558, -0.79658, -0.00355, 0.22358]]),
        atfi.const(-3.92637),
        atfi.const([[0.23399, 0.15044, -0.20545, 0.32825, 0.35412],
                    [0.15044, 0, 0, 0, 0],
                    [-0.20545, 0, 0, 0, 0],
                    [0.32825, 0, 0, 0, 0],
                    [0.35412, 0, 0, 0, 0]]),
        cplx("Kmatrix_beta"),
        atfi.const(-0.07),
        cplx("Kmatrix_f_prod_1"),
        [[mpi, mpi], [mk, mk], [mpi], [meta, meta], [meta, metap]],
    ]


def amplitude_model(x, phsp, belle_model):
    """Named-coupling model (:func:`babar2008_named_model_amp`) with the Belle lineshapes."""
    return babar2008_named_model_amp(x, phsp, *lineshape_args(belle_model))


def couplings(belle_model):
    """Belle 2018 complex couplings, {component name: complex constant}."""
    result = {REFERENCE: atfi.complex(atfi.const(1.0), atfi.const(0.0))}
    for name in FREE_COMPONENTS:
        if name != REFERENCE:
            result[name] = atfi.complex(
                atfi.const(belle_model[f"{name}_realpart"][0]),
                atfi.const(belle_model[f"{name}_imaginarypart"][0]),
            )
    return result


def amplitude(x, phsp, belle_model, switches=DEFAULT_SWITCHES):
    """Complex D0 -> KS pi+ pi- amplitude at the points ``x``."""
    return amplitude_model(x, phsp, belle_model)(couplings=couplings(belle_model), switches=switches)

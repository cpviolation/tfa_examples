"""Reference values of the Belle D0 -> KS pi pi amplitude from the C++ implementation.

``inputs/d02kspipi_cpp_reference.npz`` is produced by ``macros/make_cpp_reference.py``
from Resonance.cpp / d02kspipi_amplitudes.cpp in d02kspipi_toys. It holds, at 2000
Dalitz points, the total amplitude and every term of the coherent sum (couplings
included), keyed by the names in ``BABAR2008_COMPONENTS``.
"""

import numpy as np

import amplitf.interface as atfi
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

from tfaex import paths
from tfaex.io import load_belle_model
from tfaex.models import belle2018
from tfaex.models.d02kspipi import babar2008_components


def load_cpp_reference(file_name=paths.CPP_REFERENCE):
    """Reference arrays as a dict; adds ``m2_pipi`` computed with the C++ masses."""
    with np.load(file_name) as f:
        ref = {k: f[k] for k in f.files}
    msqsum = ref["md"] ** 2 + 2 * ref["mpi"] ** 2 + ref["mkz"] ** 2
    ref["m2_pipi"] = msqsum - ref["m2_kspim"] - ref["m2_kspip"]
    return ref


def cpp_phase_space(ref):
    """Dalitz phase space with the masses hard-coded in the C++ code."""
    mpi, mkz, md = (atfi.const(float(ref[k])) for k in ("mpi", "mkz", "md"))
    return DalitzPhaseSpace(mpi, mkz, mpi, md)


def tf_components(ref, belle_model=None):
    """TF terms of the coherent sum (coupling x component) at the reference points.

    Returns {name: complex numpy array}, with the same keys as the C++ terms plus
    ``total`` (the Belle 2018 amplitude with default switches).
    """
    belle_model = belle_model or load_belle_model()
    phsp = cpp_phase_space(ref)
    x = atfi.const(np.stack([ref["m2_kspim"], ref["m2_kspip"]], axis=1))
    comps = babar2008_components(x, phsp, *belle2018.lineshape_args(belle_model))
    cpl = belle2018.couplings(belle_model)
    out = {k: (cpl[k] * v if k in cpl else v).numpy() for k, v in comps.items()}
    out["total"] = belle2018.amplitude(x, phsp, belle_model).numpy()
    return out

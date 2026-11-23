"""Readers for model parameters and binning look-up tables."""

import numpy as np
import uproot

from tfaex import paths


def decode_model(fname, definitions=paths.BELLE_MODEL_DEFINITIONS):
    """Read a parameter file whose keys are the numeric codes in ``definitions``.

    Returns a dict mapping parameter name to the list of values on its line.
    """
    parameter_names = {}
    with open(definitions, "r") as f:
        for line in f:
            values = line.split()
            if len(values) == 0:
                continue
            parameter_names[values[1]] = values[0]
    model = {}
    with open(fname, "r") as f:
        for line in f:
            values = line.split()
            if len(values) == 0:
                continue
            model[parameter_names[values[0]]] = [float(v) for v in values[1:]]
    return model


def load_belle_model():
    """Parameters of the Belle 2018 D0 -> KS pi pi model."""
    return decode_model(paths.BELLE_MODEL)


def add_binflip_binid(data, lut_file=paths.BINFLIP_LUT, histo=paths.BINFLIP_LUT_HISTO):
    """Append the signed BaBar-2008 equal-delta-deltaD bin index as a new column.

    The sign is flipped for events with m2(KS pi-) bin index > m2(KS pi+) bin index.
    """
    with uproot.open(lut_file) as hf:
        hnp = hf[histo].to_numpy()
    ix = np.digitize(data[:, 0], hnp[1]) - 1
    iy = np.digitize(data[:, 1], hnp[2]) - 1
    binid = hnp[0][ix, iy] * (2 * (ix > iy) - 1)
    return np.column_stack((data, binid))

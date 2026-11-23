"""Decay-time efficiency and its application to toy samples."""

import numpy as np


def decay_time_efficiency(t, t0, a, b, m, n):
    r"""Decay-time efficiency (not normalised)

    .. math::

        \varepsilon(t) = \left(\frac{(t-t_0)^a}{b + (t-t_0)^n}\right)^m
        \quad (t > t_0), \qquad \varepsilon(t) = 0 \quad (t \le t_0)

    ``t`` and ``t0`` are in the units of the sample (t/tau for the toys of
    macros/d02kspipi_toys.py). For a > n the function grows without bound, so
    with accept-reject the efficiency is relative to its largest value in the
    sample.
    """
    dt = np.asarray(t, dtype=float) - t0
    dtp = np.clip(dt, 0.0, None)
    return np.where(dt > 0, (dtp**a / (b + dtp**n)) ** m, 0.0)


def apply_efficiency(data, efficiency, rng=None):
    """Keep each row of ``data`` with probability efficiency / max(efficiency)."""
    efficiency = np.asarray(efficiency, dtype=float)
    if efficiency.shape[0] != len(data):
        raise ValueError("one efficiency value per event is needed")
    emax = efficiency.max()
    if emax <= 0:
        raise ValueError("efficiency is zero for all events")
    rng = np.random.default_rng(rng)
    return data[rng.uniform(0.0, emax, len(efficiency)) < efficiency]

"""Generate a D0 -> KS pi+ pi- toy sample with mixing (Dalitz variables and decay time).

The D0 and D0bar amplitudes are read from the binned Belle model in
inputs/belle_cache.root. The sample is saved as a numpy array with columns
(m2(KS pi-), m2(KS pi+), t/tau) in $TFAEX_OUTPUT or <repo>/output.

With --acceptance T0 A B M N, a decay-time efficiency
((t-t0)^a / (b + (t-t0)^n))^m (tfaex.acceptance) is applied by accept-reject,
so fewer than --ntoys events are kept.
"""

import argparse
import time

import numpy as np

import amplitf.interface as atfi
import amplitf.mixing as atfm
import tfa.toymc as tft
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace

from tfaex import paths
from tfaex.acceptance import apply_efficiency, decay_time_efficiency
from tfaex.models.d02kspipi import belle_cache_amp, dalitz_phase_space

# Decay time in units of the D0 lifetime
phsp = dalitz_phase_space()
tphsp = DecayTimePhaseSpace(atfi.const(1.0))
c_phsp = CombinedPhaseSpace(phsp, tphsp)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ntoys", type=int, default=100000, help="Number of events to generate")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument("-x", type=float, default=0.004, help="Mixing parameter x")
    parser.add_argument("-y", type=float, default=0.0064, help="Mixing parameter y")
    parser.add_argument("-qop", type=float, default=1.0, help="|q/p|")
    parser.add_argument("-qop_phase", type=float, default=0.0, help="arg(q/p)")
    parser.add_argument("--acceptance", type=float, nargs=5, default=None,
                        metavar=("T0", "A", "B", "M", "N"),
                        help="Apply the decay-time efficiency with these parameters (t in units of tau)")
    parser.add_argument("--output", type=str, default="d02kspipi_toy",
                        help="Output file name, without extension")
    parser.add_argument("--dryrun", action="store_true", help="Parse the arguments and exit")
    return parser.parse_args(argv)


def mixing_model(x):
    """Time-dependent density; returns a function of the mixing parameters."""
    ampl_dz = belle_cache_amp(c_phsp.data1(x), phsp)
    # D0bar: swap m2(KS pi+) <-> m2(KS pi-) only (not the decay time column)
    ampl_dzb = belle_cache_amp(c_phsp.data1(x)[:, ::-1], phsp)
    t = c_phsp.phsp2.t(c_phsp.data2(x))

    def _model(x_mix_par, y_mix_par, qoverp_re, qoverp_im):
        tep = atfm.psip(t, y_mix_par, atfi.const(1.0))
        tem = atfm.psim(t, y_mix_par, atfi.const(1.0))
        tei = atfm.psii(t, x_mix_par, atfi.const(1.0))
        return atfm.mixing_density(
            ampl_dz, ampl_dzb, atfi.complex(qoverp_re, qoverp_im), tep, tem, tei
        )

    return _model


def toymc_mixing_model(x_mix, y_mix, qop, qop_phase):
    """PDF for tfa.toymc.run_toymc with fixed mixing parameters."""

    def pdf(x):
        return mixing_model(x)(
            x_mix_par=atfi.const(x_mix),
            y_mix_par=atfi.const(y_mix),
            qoverp_re=atfi.const(qop) * atfi.cos(atfi.const(qop_phase)),
            qoverp_im=atfi.const(qop) * atfi.sin(atfi.const(qop_phase)),
        )

    return pdf


def generate(ntoys, x_mix, y_mix, qop, qop_phase, seed=None):
    """Generate ``ntoys`` events; returns a numpy array (m2ab, m2bc, t)."""
    pdf = toymc_mixing_model(x_mix, y_mix, qop, qop_phase)
    # maximum=1e-20: run_toymc raises the majorant to 1.5x the largest PDF value
    # it finds and starts over, so the maximum does not need to be known.
    sample = tft.run_toymc(pdf, c_phsp, ntoys, maximum=1.0e-20, chunk=1000000,
                           seed=seed, components=False)
    return sample.numpy()


def apply_acceptance(sample, t0, a, b, m, n, seed=None):
    """Keep events with probability proportional to the decay-time efficiency."""
    eff = decay_time_efficiency(sample[:, 2], t0, a, b, m, n)
    return apply_efficiency(sample, eff, rng=seed)


def main(argv=None):
    args = parse_args(argv)
    if args.dryrun:
        print(args)
        return
    start_time = time.time()
    sample = generate(args.ntoys, args.x, args.y, args.qop, args.qop_phase, seed=args.seed)
    print(f"Generated {args.ntoys} events in {time.time() - start_time:.2f} seconds.")
    if args.acceptance is not None:
        sample = apply_acceptance(sample, *args.acceptance, seed=args.seed)
        print(f"Kept {len(sample)} events after the decay-time acceptance.")
    file_name = paths.output_dir() / (args.output + ".npy")
    np.save(file_name, sample)
    print(f"Saved toy sample to {file_name}")


if __name__ == "__main__":
    main()

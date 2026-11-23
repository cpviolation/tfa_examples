"""Generate and fit D0 -> K+ K- pi+ pi- toys with the AmpGen-defined model.

The model (params/ampgen/DtoKKpipi_stage1.opt) is the LHCb amplitude model as
defined in AmpGen, restricted for now to its Breit-Wigner / Gounaris-Sakurai /
rho-omega terms. Toys are generated with the AmpGen couplings; the free
couplings of the options file are then fitted, starting from shifted values.
The normalisation integral uses points generated from the model at the starting
couplings, weighted by 1/pdf (importance sampling); --uniform-norm uses uniform
points instead, which need many more points for the same precision.
Results (JSON) and projection plots are written to $TFAEX_OUTPUT or <repo>/output.
"""

import argparse
import json
import time

import numpy as np

import amplitf.interface as atfi
import amplitf.likelihood as atfl
import tfa.optimisation as tfo
import tfa.toymc as tft

from tfaex import paths
from tfaex.models import d02kkpipi


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ntoys", type=int, default=20000, help="Number of toy events")
    parser.add_argument("--nnorm", type=int, default=200000, help="Number of normalisation points")
    parser.add_argument("--uniform-norm", action="store_true",
                        help="Uniform normalisation points instead of points generated from the model "
                             "at the starting couplings (importance sampling)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--shift", type=float, default=0.2,
                        help="Relative random shift of the starting values of the free couplings")
    parser.add_argument("--output", type=str, default="d02kkpipi_fit", help="Output file prefix")
    parser.add_argument("--no-plots", action="store_true", help="Do not make projection plots")
    parser.add_argument("--dryrun", action="store_true", help="Parse the arguments and exit")
    return parser.parse_args(argv)


def free_head_rules(model):
    """Free couplings of D0 decays in the options file (sub-decay couplings stay fixed)."""
    return [r for r in model.free_couplings if r.startswith(model.head)]


def generate(model, phsp, ntoys, seed=None):
    def pdf(x):
        return d02kkpipi.pdf_model(model, phsp, x)()

    # maximum=1e-20: run_toymc finds the maximum itself
    return tft.run_toymc(pdf, phsp, ntoys, maximum=1.0e-20, chunk=200000, seed=seed, components=False)


def starting_couplings(model, rules, shift, seed):
    """Couplings of ``rules`` shifted randomly by a relative amount ``shift``."""
    rng = np.random.default_rng(seed)
    start = {}
    for rule in rules:
        c = model.couplings[rule]
        start[rule] = complex(c.real * (1 + shift * rng.normal()), c.imag * (1 + shift * rng.normal()))
    return start


def fit(model, phsp, data, norm, norm_weights, rules, start):
    pars, names = d02kkpipi.fit_parameters(model, rules)
    for p in pars:
        rule = names[p.name.rsplit("_", 1)[0]]
        p.update(start[rule].real if p.name.endswith("_re") else start[rule].imag)
    data_pdf = d02kkpipi.pdf_model(model, phsp, data)
    norm_pdf = d02kkpipi.pdf_model(model, phsp, norm)

    @atfi.function
    def nll(values):
        couplings = d02kkpipi.couplings_from_parameters(values, names)
        return atfl.unbinned_nll(data_pdf(couplings), atfl.weighted_integral(norm_pdf(couplings), norm_weights))

    return tfo.run_minuit(nll, pars), names


def plot_projections(model, phsp, data, norm, norm_weights, couplings, file_name):
    import matplotlib.pyplot as plt

    def masses(x):
        pkp, pkm, ppp, ppm = (v.numpy() for v in d02kkpipi.momenta(phsp, x))

        def m(*ps):
            s = sum(ps)
            return np.sqrt(np.maximum(s[:, 3] ** 2 - np.sum(s[:, :3] ** 2, axis=1), 0))

        return {r"$m(K^+K^-)$": m(pkp, pkm), r"$m(\pi^+\pi^-)$": m(ppp, ppm),
                r"$m(K^+\pi^-)$": m(pkp, ppm), r"$m(K^-\pi^+)$": m(pkm, ppp)}

    w = (d02kkpipi.pdf_model(model, phsp, norm)(couplings) * norm_weights).numpy()
    md, mn = masses(data), masses(norm)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key in zip(axes.flat, md):
        h, edges, _ = ax.hist(md[key], bins=60, histtype="step", color="k", label="toy data")
        ax.hist(mn[key], bins=edges, weights=w * len(md[key]) / w.sum(), histtype="step", label="fit")
        ax.set_xlabel(key + " [GeV]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(file_name)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.dryrun:
        print(args)
        return
    model = d02kkpipi.load_model()
    phsp = d02kkpipi.phase_space(model)
    rules = free_head_rules(model)
    print(f"{len(model.chains)} decay chains, {len(rules)} free D0 couplings")

    start = time.time()
    data = generate(model, phsp, args.ntoys, seed=args.seed)
    start_couplings = starting_couplings(model, rules, args.shift, args.seed)
    norm, norm_weights = d02kkpipi.normalisation_sample(
        model, phsp, args.nnorm, start_couplings, seed=args.seed + 1000, uniform=args.uniform_norm)
    print(f"Generated {len(data)} toys and {len(norm)} normalisation points in {time.time() - start:.1f} s")

    result, names = fit(model, phsp, data, norm, norm_weights, rules, start_couplings)
    print(f"Fit: {result['func_calls']} calls, {result['time']:.1f} s, -log L = {result['loglh']:.2f}")

    rows = []
    for name, rule in names.items():
        true = model.couplings[rule]
        for part, t in (("re", true.real), ("im", true.imag)):
            v, e = result["params"][f"{name}_{part}"]
            rows.append({"rule": rule, "part": part, "true": t, "fit": v, "error": e, "pull": (v - t) / e})
            print(f"{rows[-1]['pull']:+6.2f}  {part} {v:+.4f} +- {e:.4f} (true {t:+.4f})  {rule}")
    pulls = np.array([r["pull"] for r in rows])
    print(f"Pulls: mean {pulls.mean():+.2f}, std {pulls.std(ddof=1):.2f} over {len(pulls)} parameters")

    out = paths.output_dir()
    with open(out / f"{args.output}.json", "w") as f:
        json.dump({"ntoys": len(data), "fit": rows, "loglh": result["loglh"]}, f, indent=1)
    if not args.no_plots:
        fitted = d02kkpipi.couplings_from_parameters(
            {k: atfi.const(v[0]) for k, v in result["params"].items()}, names)
        plot_projections(model, phsp, data, norm, norm_weights, fitted, out / f"{args.output}.png")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()

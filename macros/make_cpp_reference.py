"""Produce reference values of the Belle D0 -> KS pi+ pi- amplitude from the C++ code
in d02kspipi_toys (generator/src/Resonance.cpp and d02kspipi_amplitudes.cpp).

Runs outside the uv environment, with a Python that has PyROOT and numpy, e.g.

    /opt/homebrew/bin/python3 macros/make_cpp_reference.py ../d02kspipi_toys

The C++ sources are compiled into a temporary shared library (only ROOT is needed;
the unused boost include is stubbed). Writes inputs/d02kspipi_cpp_reference.npz with

    m2_kspim, m2_kspip : Dalitz points (GeV^2)
    total              : D02KsPiPiAmp::calc_amp(m2_kspim, m2_kspip)
    <component>        : Resonance::contribution for each term of calc_amp
                         (coupling included), keyed by BABAR2008_COMPONENTS names
    md, mpi, mkz       : masses hard-coded in Resonance.cpp
    commit             : d02kspipi_toys commit used
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "inputs" / "d02kspipi_cpp_reference.npz"
NPOINTS = 2000
SEED = 12345

# Masses hard-coded in Resonance.cpp
MD, MPI, MKZ = 1.86483, 0.13957, 0.49761


def build_library(toys, workdir):
    stub = workdir / "stub" / "boost"
    stub.mkdir(parents=True)
    (stub / "lexical_cast.hpp").touch()
    lib = workdir / "libd02kspipi_ref.so"
    cflags = subprocess.check_output(["root-config", "--cflags"], text=True).split()
    cflags = [f for f in cflags if not f.startswith("-std=")]
    libs = subprocess.check_output(["root-config", "--libs"], text=True).split()
    src = toys / "generator" / "src"
    cmd = (["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-undefined", "dynamic_lookup"]
           + cflags + [f"-I{workdir / 'stub'}", f"-I{src}", f"-I{toys / 'common' / 'src'}",
                       str(src / "Resonance.cpp"), str(src / "d02kspipi_amplitudes.cpp")]
           + libs + ["-o", str(lib)])
    subprocess.check_call(cmd)
    return lib, [workdir / "stub", src, toys / "common" / "src"]


def read_parameters():
    names = {}
    for line in (REPO / "inputs" / "belle_model_definitions.txt").read_text().splitlines():
        v = line.split()
        if len(v) >= 2 and not line.startswith("#"):
            names[v[1]] = v[0]
    pars = {}
    for line in (REPO / "params" / "belle_model.txt").read_text().splitlines():
        v = line.split()
        if v:
            pars[names[v[0]]] = float(v[1])
    return pars


def main(toys):
    import ROOT

    toys = Path(toys).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        lib, includes = build_library(toys, Path(tmp))
        for inc in includes:
            ROOT.gInterpreter.AddIncludePath(str(inc))
        ROOT.gInterpreter.Declare('#include "Resonance.h"\n#include "d02kspipi_amplitudes.h"')
        if ROOT.gSystem.Load(str(lib)) < 0:
            sys.exit(f"cannot load {lib}")

        p = read_parameters()
        model = ROOT.D02KsPiPiAmp(str(REPO / "params" / "belle_model.txt"))

        def polar(name):
            c = complex(p[f"{name}_realpart"], p[f"{name}_imaginarypart"])
            return abs(c), np.degrees(np.angle(c))

        def bw(name, mass, spin, axis):
            amp, phase = polar(name) if name else (1.0, 0.0)
            return ROOT.Resonance("BW", amp, phase, p[f"{mass}_Mass"], p[f"{mass}_Width"],
                                  spin, 1.5, 5.0, axis)

        def lass(name, axis):
            amp, phase = polar(name)
            return ROOT.Resonance("LASS", amp, phase, p["Kstarzero1430_Mass"],
                                  p["Kstarzero1430_Width"], 0, 1.5, 5.0, axis,
                                  p["LASS_F"], p["LASS_phi_F"], p["LASS_R"], p["LASS_phi_R"],
                                  p["LASS_a"], p["LASS_r"])

        km_args = []
        for name in [f"Kmatrix_beta{i}" for i in range(1, 6)] + [f"Kmatrix_f_prod_1{i}" for i in range(1, 6)]:
            c = complex(p[f"{name}_realpart"], p[f"{name}_imaginarypart"])
            km_args += [abs(c), float(np.angle(c))]  # K-matrix phases in radians, as in calc_amp

        # Same terms as D02KsPiPiAmp::calc_amp (there is no K*(1680)+ term)
        resonances = {
            "rho770": bw(None, "rho770", 1, "Y"),
            "omega": bw("omega", "omega", 1, "Y"),
            "Kstar892minus": bw("Kstar892minus", "Kstar892", 1, "X"),
            "Kstarzero1430minus": lass("Kstarzero1430minus", "X"),
            "Kstartwo1430minus": bw("Kstartwo1430minus", "Kstartwo1430", 2, "X"),
            "Kstar1680minus": bw("Kstar1680minus", "Kstar1680", 1, "X"),
            "Kstar1410minus": bw("Kstar1410minus", "Kstar1410", 1, "X"),
            "Kstar892plus": bw("Kstar892plus", "Kstar892", 1, "Z"),
            "Kstarzero1430plus": lass("Kstarzero1430plus", "Z"),
            "Kstartwo1430plus": bw("Kstartwo1430plus", "Kstartwo1430", 2, "Z"),
            "Kstar1410plus": bw("Kstar1410plus", "Kstar1410", 1, "Z"),
            "ftwo1270": bw("ftwo1270", "ftwo1270", 2, "Y"),
            "rho1450": bw("rho1450", "rho1450", 1, "Y"),
            "Kmatrix": ROOT.Resonance("KMatrix", "Y", *km_args, p["Kmatrix_s_prod_0"]),
        }

        rng = np.random.default_rng(SEED)
        msqsum = MD**2 + 2 * MPI**2 + MKZ**2
        points, total = [], []
        comps = {k: [] for k in resonances}
        while len(points) < NPOINTS:
            m2m, m2p = rng.uniform(0.3, 3.0, 2)
            a = model.calc_amp(m2m, m2p)
            if a.real == 0 and a.imag == 0:  # outside the Dalitz plot
                continue
            points.append((m2m, m2p))
            total.append(complex(a.real, a.imag))
            m2pipi = msqsum - m2m - m2p
            for k, r in resonances.items():
                c = r.contribution(m2m, m2pipi)
                comps[k].append(complex(c.Re(), c.Im()))

        total = np.array(total)
        for k, v in {"total": total, **comps}.items():
            bad = ~np.isfinite(np.array(v))
            if bad.any():
                print(f"{k}: {bad.sum()} non-finite values, e.g. at {points[np.argmax(bad)]}")
        comp_sum = sum(np.array(v) for v in comps.values())
        dev = np.max(np.abs(comp_sum - total) / np.abs(total))
        print(f"max |sum of components - calc_amp| / |calc_amp| = {dev:.2e}")
        assert dev < 1e-10, "components do not reproduce calc_amp"

        commit = subprocess.check_output(["git", "-C", str(toys), "rev-parse", "HEAD"], text=True).strip()
        points = np.array(points)
        out = OUTPUT
        np.savez_compressed(out, m2_kspim=points[:, 0], m2_kspip=points[:, 1], total=total,
                            md=MD, mpi=MPI, mkz=MKZ, commit=commit,
                            **{k: np.array(v) for k, v in comps.items()})
        print(f"wrote {out} ({NPOINTS} points, d02kspipi_toys {commit[:8]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else REPO.parent / "d02kspipi_toys")

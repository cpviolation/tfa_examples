# tfa_examples

Example amplitude models, toy studies and tests built on
[AmpliTF](https://github.com/cpviolation/AmpliTF) and
[TFA2](https://github.com/cpviolation/TFA2).

## Layout

```
external/AmpliTF   git submodule (cpviolation fork, master)
external/TFA2      git submodule (cpviolation fork, master)
src/tfaex/         installable package with the models and helpers
  models/          amplitude models (d02kspipi, belle2018, d02kkpipi)
  acceptance.py    decay-time efficiency for toys
  io.py            parameter files and binning look-up tables
  paths.py         locations of input files and of the output directory
  plotting.py      standard plots
  reference.py     comparison with the C++ reference values
  ampgen.py        models from AmpGen options files
macros/            command-line scripts (toys, fits, studies)
notebooks/         exploratory notebooks
params/            model parameters
inputs/            cached model, C++ reference values, parameter definitions, binning LUT
tests/             pytest suite
```

## Installation

Clone with the submodules:

```bash
git clone --recurse-submodules git@github.com:cpviolation/tfa_examples.git
# or, in an existing clone
git submodule update --init
```

Create the environment with [uv](https://docs.astral.sh/uv/). AmpliTF and TFA2
are installed in editable mode from `external/`:

```bash
uv sync                    # CPU
uv sync --extra metal      # Apple Silicon GPU
uv sync --extra cuda       # NVIDIA GPU (Linux)
```

With plain pip instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e external/AmpliTF -e external/TFA2 -e . pytest jupyter
```

## Usage

```bash
uv run pytest                                   # run the tests (also run by CI on every push/PR)
uv run pre-commit install                       # once per clone: strip notebook outputs on commit
uv run python macros/d02kspipi_toys.py --ntoys 100000
uv run python macros/d02kspipi_toys.py --acceptance 0.3 1.5 2.0 1.0 1.0   # with decay-time efficiency
uv run python macros/binflipfitter.py
uv run jupyter lab notebooks/
```

### Models with named couplings

Components are identified by name (`tfaex.models.d02kspipi.BABAR2008_COMPONENTS`)
and couplings are passed as a dict, or as real fit parameters
`<name>_re`/`<name>_im`:

```python
from tfaex.models import belle2018
from tfaex.models.couplings import fit_parameters, switches_for
from tfaex.models.d02kspipi import BABAR2008_COMPONENTS

model = belle2018.amplitude_model(x, phsp, belle_model)
amp = model(couplings=belle2018.couplings(belle_model))
amp_kst = model(couplings=belle2018.couplings(belle_model), switches=switches_for(BABAR2008_COMPONENTS, ["Kstar892minus"]))
pars = fit_parameters({"Kstar892minus": -1.2 + 1.1j})   # Kstar892minus_re, Kstar892minus_im
```

`switches` is kept because TFA2 (`run_toymc`, `calculate_fit_fractions`) uses
it to identify components. `tests/test_fit.py` shows a complete fit. The
positional `babar2008_model_amp(...)(a1r, a1i, ...)` interface still works.

Samples have columns m²(KSπ⁻), m²(KSπ⁺) [, t/τ] in GeV²
(`tfaex.models.d02kspipi.dalitz_phase_space`).

Macros write to `output/` (git-ignored); set `TFAEX_OUTPUT` to use another
directory. Some notebooks still read `TFAEX_ROOT`: run `source setup.sh`
before starting Jupyter for those.

## D0 -> K+ K- pi+ pi- with the AmpGen model

`tfaex.ampgen` builds an amplitude model directly from an
[AmpGen](https://github.com/GooFit/AmpGen) options file: decay descriptors and
couplings, particle parameters, pseudo-particles, AmpGen's daughter ordering
and orbital rules, with covariant spin factors (`amplitf.covariant`) and
AmpGen-convention lineshapes (`amplitf.ampgen_lineshapes`).
`tfaex.models.d02kkpipi` uses it for the LHCb D0 -> K+ K- pi+ pi- model
(AmpGen `options/DtoKKpipi_v2.opt`, copied unmodified to
`params/ampgen/DtoKKpipi_v2.opt`): 55 decay chains with Breit-Wigner,
Gounaris-Sakurai, rho-omega, a0(980) Flatte, FOCUS K pi S-wave, AmpGen's
pi pi / K K K-matrix and spline running-width (K1(1270), a1(1260)) lineshapes.
Every chain agrees with AmpGen to ~1e-5 (`tests/test_ampgen_d02kkpipi.py`,
reference values described in `macros/ampgen/README.md`).

```bash
uv run python macros/d02kkpipi_fit.py      # generate toys and fit the 25 free D0 couplings
```

## Reference values from the C++ code

`inputs/d02kspipi_cpp_reference.npz` holds the Belle amplitude and each of its
terms from the C++ implementation in
[d02kspipi_toys](https://gitlab.cern.ch/mamartin/d02kspipi_toys)
(`Resonance.cpp`, `d02kspipi_amplitudes.cpp`) at 2000 Dalitz points.
`tests/test_cpp_reference.py` compares the TF model with it term by term, and
`tfaex.reference` gives the same comparison for notebooks. To regenerate it you
need ROOT with PyROOT (e.g. Homebrew `root`) and a d02kspipi_toys checkout:

```bash
/opt/homebrew/bin/python3 macros/make_cpp_reference.py ../d02kspipi_toys
```

## Working on AmpliTF / TFA2

Changes to the core libraries go in the submodules, which are normal git
clones:

```bash
cd external/AmpliTF
git switch master            # submodules start in detached HEAD
# ... edit, commit, push to the cpviolation fork ...
cd ../..
git add external/AmpliTF     # record the new commit in tfa_examples
git commit -m "Bump AmpliTF"
```

To update both to the latest commit of their tracked branch:
`git submodule update --remote`.

It is advisable to add the upstream repositories as extra remotes:

```bash
git -C external/AmpliTF remote add upstream git@github.com:apoluekt/AmpliTF.git
git -C external/TFA2 remote add upstream git@github.com:apoluekt/TFA2.git
```

## Licences

tfa_examples is distributed under the MIT licence (`LICENSE`), except:

- `params/ampgen/DtoKKpipi_v2.opt` (copy of an AmpGen file) and
  `macros/ampgen/DumpTerms.cpp` (builds against AmpGen): GPL-2.0-only, as
  [AmpGen](https://github.com/GooFit/AmpGen); see `LICENSES/GPL-2.0-only.txt`;
- `params/ampgen/particles.json`: particle properties from the PDG (2008 edition), see
  `params/ampgen/README.md`.

AmpliTF and TFA2 (`external/`) are separate projects with their own licences.


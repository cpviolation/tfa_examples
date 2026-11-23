# tfa_examples

Example amplitude models, toy studies and tests built on
[AmpliTF](https://github.com/cpviolation/AmpliTF) and
[TFA2](https://github.com/cpviolation/TFA2).

## Layout

```
external/AmpliTF   git submodule (cpviolation fork, master)
external/TFA2      git submodule (cpviolation fork, master)
src/tfaex/         installable package with the models and helpers
  models/          amplitude models (d02kspipi, belle2018)
  io.py            parameter files and binning look-up tables
  paths.py         locations of input files and of the output directory
  plotting.py      standard plots
macros/            command-line scripts (toys, fits, studies)
notebooks/         exploratory notebooks
params/            model parameters
inputs/            cached model, parameter definitions, binning look-up tables
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

Macros write to `output/` (git-ignored); set `TFAEX_OUTPUT` to use another
directory. Some notebooks still read `TFAEX_ROOT`: run `source setup.sh`
before starting Jupyter for those.

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

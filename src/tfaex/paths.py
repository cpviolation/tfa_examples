"""Locations of the input files shipped with the repository.

The repository root is inferred from the location of this file (the package
is meant to be installed in editable mode from ``src/``). It can be
overridden with the ``TFAEX_ROOT`` environment variable.

Output files are written to ``TFAEX_OUTPUT`` if set, otherwise to
``<repo>/output`` (ignored by git).
"""

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("TFAEX_ROOT", Path(__file__).resolve().parents[2]))

PARAMS_DIR = REPO_ROOT / "params"
INPUTS_DIR = REPO_ROOT / "inputs"
FILES_DIR = REPO_ROOT / "notebooks" / "files"

BELLE_MODEL = PARAMS_DIR / "belle_model.txt"
BELLE_MODEL_DEFINITIONS = FILES_DIR / "belle_model_definitions.txt"
BELLE_CACHE = INPUTS_DIR / "belle_cache.root"
BINFLIP_LUT = FILES_DIR / "BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD.root"
BINFLIP_LUT_HISTO = "h_BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD"


def output_dir(create=True):
    """Directory where macros write their results."""
    path = Path(os.environ.get("TFAEX_OUTPUT", REPO_ROOT / "output"))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

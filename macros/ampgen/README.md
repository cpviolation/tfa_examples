# AmpGen reference values

`inputs/ampgen_d02kkpipi_v2_reference.npz` holds amplitudes computed with
[AmpGen](https://github.com/GooFit/AmpGen) (commit `fc651a7`) for the model in
`params/ampgen/DtoKKpipi_v2.opt`, and `params/ampgen/particles.json` holds
AmpGen's default particle properties. `tests/test_ampgen_d02kkpipi.py` compares
the TensorFlow model (`tfaex.ampgen`) with them, decay chain by decay chain.

To regenerate them (needs ROOT, e.g. Homebrew `root`, CMake, e.g. `uvx cmake`,
GSL and nlohmann/json; on a machine with CVMFS but without the GSL headers, add
`-DGSL_ROOT_DIR=/cvmfs/sft.cern.ch/lcg/releases/GSL/2.7-30ba4/x86_64-el9-gcc11-opt`
and `-DCMAKE_BUILD_RPATH=<same>/lib` to the first `cmake` command):

```bash
git clone https://github.com/GooFit/AmpGen.git ampgen && cd ampgen
cp <tfa_examples>/macros/ampgen/DumpTerms.cpp examples/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target DumpTerms -j 8
./build/bin/DumpTerms <tfa_examples>/params/ampgen/DtoKKpipi_v2.opt <tfa_examples>/macros/ampgen/dump_settings.opt
```

This writes `ampgen_d02kkpipi_v2_reference.json`: 2000 phase-space events
(four-momenta of K+, K-, pi+, pi-), for every decay chain its AmpGen descriptor,
amplitude without coupling and coupling, the total amplitude, and the particle
properties. Convert it to the `.npz` and `particles.json` used by the tests with:

```python
import json, numpy as np
d = json.load(open("ampgen_d02kkpipi_v2_reference.json"))
amp = np.array([np.array(t["amp"]) @ [1, 1j] for t in d["terms"]])
np.savez_compressed("inputs/ampgen_d02kkpipi_v2_reference.npz", events=np.array(d["events"]),
                    names=np.array([t["name"] for t in d["terms"]]), amplitudes=amp,
                    couplings=np.array([complex(*t["coupling"]) for t in d["terms"]]),
                    total=np.array(d["total"]) @ [1, 1j], ampgen_commit="<commit>")
json.dump({"_source": "AmpGen <commit>", **d["particles"]}, open("params/ampgen/particles.json", "w"), indent=1)
```

The `Fitter` application of AmpGen `fc651a7` does not compile with recent clang; it is not needed.

## Licence

`DumpTerms.cpp` builds against AmpGen, which is distributed under the GPL-2.0, so
this file is distributed under the GPL-2.0-only
([`LICENSES/GPL-2.0-only.txt`](../../LICENSES/GPL-2.0-only.txt)), not under the MIT
licence of the rest of tfa_examples. `dump_settings.opt` is a plain list of settings.

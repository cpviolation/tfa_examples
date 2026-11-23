# AmpGen reference values

`inputs/ampgen_d02kkpipi_stage1_reference.npz` holds amplitudes computed with
[AmpGen](https://github.com/GooFit/AmpGen) (commit `fc651a7`) for the model in
`params/ampgen/DtoKKpipi_stage1.opt`, and `params/ampgen/particles.json` holds
AmpGen's default particle properties. `tests/test_ampgen_d02kkpipi.py` compares
the TensorFlow model (`tfaex.ampgen`) with them, decay chain by decay chain.

To regenerate them (needs ROOT, e.g. Homebrew `root`, and CMake, e.g. `uvx cmake`):

```bash
git clone https://github.com/GooFit/AmpGen.git ampgen && cd ampgen
cp <tfa_examples>/macros/ampgen/DumpTerms.cpp examples/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target DumpTerms -j 8
./build/bin/DumpTerms <tfa_examples>/params/ampgen/DtoKKpipi_stage1.opt <tfa_examples>/macros/ampgen/dump_settings.opt
```

This writes `ampgen_d02kkpipi_stage1_reference.json`: 2000 phase-space events
(four-momenta of K+, K-, pi+, pi-), for every decay chain its AmpGen descriptor,
amplitude without coupling and coupling, the total amplitude, and the particle
properties. Convert it to the `.npz` and `particles.json` used by the tests with:

```python
import json, numpy as np
d = json.load(open("ampgen_d02kkpipi_stage1_reference.json"))
amp = np.array([np.array(t["amp"]) @ [1, 1j] for t in d["terms"]])
np.savez_compressed("inputs/ampgen_d02kkpipi_stage1_reference.npz", events=np.array(d["events"]),
                    names=np.array([t["name"] for t in d["terms"]]), amplitudes=amp,
                    couplings=np.array([complex(*t["coupling"]) for t in d["terms"]]),
                    total=np.array(d["total"]) @ [1, 1j], ampgen_commit="<commit>")
json.dump({"_source": "AmpGen <commit>", **d["particles"]}, open("params/ampgen/particles.json", "w"), indent=1)
```

The `Fitter` application of AmpGen `fc651a7` does not compile with recent clang; it is not needed.

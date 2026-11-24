"""Evaluate amplitude models defined with AmpGen options files, in TensorFlow.

The model is read from an AmpGen options file (decay descriptors with
couplings, particle parameters and aliases) and evaluated with the covariant
spin factors of :mod:`amplitf.covariant` and the AmpGen-convention lineshapes of
:mod:`amplitf.ampgen_lineshapes`. The conventions follow AmpGen
(https://github.com/GooFit/AmpGen):

- daughters are ordered as in ``Particle::operator<`` (higher spin first,
  unstable before stable, heavier first, then PDG id and charge; antiparticles
  are ordered as their conjugates);
- the default orbital angular momentum is the lowest allowed one, with parity
  conservation for strong decays (all but the head);
- narrow particles (width < 1 keV, e.g. the head) get the ``FormFactor``
  lineshape, other resonances ``BW`` unless a lineshape is given in brackets;
- pseudo-particles with a single daughter (e.g. ``rhoOmega00``) have no lineshape;
  their decays are expanded with their own couplings.

Only the lineshapes and spin factors validated against AmpGen are supported;
anything else raises ``NotImplementedError``.
"""

import cmath
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf

import amplitf.ampgen_lineshapes as agl
import amplitf.covariant as cov
import amplitf.interface as atfi

STABLE_THRESHOLD = 1e-6  # GeV, AmpGen Particle::StableThreshold (1 keV)
SPIN_LETTER = {0: "S", 2: "V", 4: "T"}
SUPPORTED_LINESHAPES = {"BW", "GounarisSakurai", "FormFactor"}


# ----------------------------------------------------------------------------
# Particle properties

def load_particles(file_name):
    """AmpGen particle properties (``two_spin``, ``parity``, ``pdg_id``, ``charge``,
    ``mass``, ``width``, ``radius``) keyed by AmpGen name."""
    with open(file_name) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------
# Decay descriptors

@dataclass
class Decay:
    name: str
    modifiers: list = field(default_factory=list)
    daughters: list = field(default_factory=list)

    def descriptor(self):
        mods = f"[{';'.join(self.modifiers)}]" if self.modifiers else ""
        ds = "{" + ",".join(d.descriptor() for d in self.daughters) + "}" if self.daughters else ""
        return self.name + mods + ds


def parse_descriptor(text):
    """Parse ``'D0[D]{K*(892)0{K+,pi-},K*(892)bar0{K-,pi+}}'`` into a :class:`Decay` tree."""
    m = re.match(r"([^\[\]{},]+)(\[([^\]]*)\])?", text)
    if m is None:
        raise ValueError(f"Cannot parse decay descriptor '{text}'")
    decay = Decay(m.group(1), [x for x in (m.group(3) or "").split(";") if x])
    rest = text[m.end():]
    if rest:
        if not (rest.startswith("{") and rest.endswith("}")):
            raise ValueError(f"Cannot parse decay descriptor '{text}'")
        depth, start = 0, 1
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    decay.daughters.append(parse_descriptor(rest[start:i]))
            elif ch == "," and depth == 1:
                decay.daughters.append(parse_descriptor(rest[start:i]))
                start = i + 1
    return decay


# ----------------------------------------------------------------------------
# Options files

@dataclass
class Options:
    event_type: list
    couplings: dict  # descriptor -> (complex value, is_free)
    parameters: dict  # name -> (value, is_free)
    ignored: list


def _resolve_expression(expr, values):
    expr = expr.strip()
    m = re.fullmatch(r"(\S+)\s*([+-])\s*(PI|[0-9.eE+-]+)", expr)
    if m:
        base = _resolve_expression(m.group(1), values)
        offset = math.pi if m.group(3) == "PI" else float(m.group(3))
        return base + offset if m.group(2) == "+" else base - offset
    if expr in values:
        return values[expr]
    raise KeyError(f"Unknown parameter in expression '{expr}'")


def read_options(file_name):
    """Read an AmpGen options file: event type, couplings and parameters.

    Coupling values are complex (polar or Cartesian according to
    ``CouplingConstant::Coordinates``); flag 2 means fixed. Aliases ``A = B``
    (optionally ``B +- number|PI``) are resolved, also for the real and
    imaginary (or magnitude and phase) parts of couplings via ``_Re``/``_Im``.
    """
    event_type, polar = None, False
    raw_couplings, raw_params, aliases, ignored = {}, {}, [], []
    for line in Path(file_name).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            lhs, rhs = (x.strip() for x in line.split("=", 1))
            aliases.append((lhs, rhs))
            continue
        tokens = line.split()
        key = tokens[0]
        if key == "EventType":
            event_type = tokens[1:]
        elif key == "CouplingConstant::Coordinates":
            polar = tokens[1] == "polar"
        elif "::" in key:
            ignored.append(line)
        elif len(tokens) == 7:
            raw_couplings[key] = (float(tokens[2]), float(tokens[5]), tokens[1] != "2" or tokens[4] != "2")
        elif len(tokens) == 4:
            raw_params[key] = (float(tokens[2]), tokens[1] != "2")
        else:
            ignored.append(line)

    values = {k: v[0] for k, v in raw_params.items()}
    for k, (re_, im_, _) in raw_couplings.items():
        values[k + "_Re"], values[k + "_Im"] = re_, im_
    for lhs, rhs in aliases:  # in file order
        values[lhs] = _resolve_expression(rhs, values)
        if lhs.endswith("_Re") or lhs.endswith("_Im"):
            name = lhs[:-3]
            re_, im_ = values.get(name + "_Re"), values.get(name + "_Im")
            if re_ is not None and im_ is not None:
                raw_couplings[name] = (re_, im_, False)
        else:
            raw_params[lhs] = (values[lhs], False)

    couplings = {}
    for name, (a, b, free) in raw_couplings.items():
        couplings[name] = (cmath.rect(a, b) if polar else complex(a, b), free)
    params = {k: (values[k], free) for k, (_, free) in raw_params.items()}
    return Options(event_type, couplings, params, ignored)


# ----------------------------------------------------------------------------
# Model

class Particle:
    """A node of an expanded decay chain, with AmpGen's ordering and orbital rules."""

    def __init__(self, decay, table, params, final_state, is_head=False):
        self.name = decay.name
        self.modifiers = list(decay.modifiers)
        if self.name not in table:
            raise KeyError(f"No properties for particle '{self.name}'")
        props = table[self.name]
        self.two_spin, self.parity = props["two_spin"], props["parity"]
        self.pdg_id, self.charge = props["pdg_id"], props["charge"]
        self.mass = params.get(self.name + "_mass", props["mass"])
        self.width = params.get(self.name + "_width", props["width"])
        self.radius = params.get(self.name + "_radius", props["radius"])
        self.is_head = is_head
        self.daughters = [Particle(d, table, params, final_state) for d in decay.daughters]
        self.index = final_state.index(self.name) if not self.daughters else None
        self._sort_daughters()
        self.L = self._orbital()

    # --- AmpGen ordering (Particle::operator< and sortDaughters)
    def _key(self, conj):
        pdg = -self.pdg_id if conj else self.pdg_id
        charge = -self.charge if conj else self.charge
        return self, pdg, charge

    def _sort_daughters(self):
        import functools

        conj = self.pdg_id < 0

        def less(a, b):
            (pa, ida, qa), (pb, idb, qb) = a._key(conj), b._key(conj)
            if pa.two_spin != pb.two_spin:
                return pa.two_spin > pb.two_spin
            if pa.daughters and not pb.daughters:
                return True
            if not pa.daughters and pb.daughters:
                return False
            if pa.mass != pb.mass:
                return pa.mass > pb.mass
            if abs(ida) == abs(idb) and ida != idb:
                return ida > idb
            if qa != qb:
                return qa > qb
            if ida != idb:
                return ida > idb
            return (pa.index or 0) < (pb.index or 0)

        def cmp(a, b):
            return -1 if less(a, b) else (1 if less(b, a) else 0)

        self.daughters = sorted(self.daughters, key=functools.cmp_to_key(cmp))

    # --- orbital angular momentum
    def _orbital(self):
        for m in self.modifiers:
            if m in ("S", "P", "D", "F"):
                return "SPDF".index(m)
        if len(self.daughters) != 2:
            return 0
        j, j1, j2 = self.two_spin, self.daughters[0].two_spin, self.daughters[1].two_spin
        smin, smax = abs(j1 - j2), j1 + j2
        lmin = min(abs(j - s) for s in range(smin, smax + 1, 2)) // 2
        lmax = (j + smax) // 2
        for L in range(lmin, lmax + 1):
            if self.is_head or self.parity == self.daughters[0].parity * self.daughters[1].parity * (-1) ** L:
                return L
        raise ValueError(f"No allowed orbital angular momentum for {self.name}")

    @property
    def lineshape_name(self):
        named = [m for m in self.modifiers if m not in ("S", "P", "D", "F")]
        if named:
            return named[0]
        return "FormFactor" if self.width < STABLE_THRESHOLD else "BW"

    def descriptor(self):
        mods = [m for m in self.modifiers]
        if len(self.daughters) == 2 and self.L != self._default_orbital() and not any(m in "SPDF" for m in mods):
            mods.append("SPDF"[self.L])
        m = f"[{';'.join(mods)}]" if mods else ""
        ds = "{" + ",".join(d.descriptor() for d in self.daughters) + "}" if self.daughters else ""
        return self.name + m + ds

    def _default_orbital(self):
        saved = self.modifiers
        self.modifiers = [m for m in saved if m not in ("S", "P", "D", "F")]
        L = self._orbital()
        self.modifiers = saved
        return L

    # --- evaluation
    def momentum(self, p):
        if not self.daughters:
            return p[self.index]
        return tf.add_n([d.momentum(p) for d in self.daughters])

    def mass_squared(self, p):
        if not self.daughters:
            return atfi.ones(p[0][..., 0]) * self.mass ** 2
        return cov.dot(self.momentum(p), self.momentum(p))

    def lineshape(self, p, mpi):
        if not self.daughters:
            return atfi.complex(atfi.ones(p[0][..., 0]), atfi.zeros(p[0][..., 0]))
        if len(self.daughters) == 1:
            return self.daughters[0].lineshape(p, mpi)
        s = self.mass_squared(p)
        s1, s2 = self.daughters[0].mass_squared(p), self.daughters[1].mass_squared(p)
        name = self.lineshape_name
        if name == "FormFactor":
            own = atfi.cast_complex(agl.form_factor_signed(s, s1, s2, self.radius, self.L))
        elif name == "BW":
            own = agl.breit_wigner(s, s1, s2, self.mass, self.width, self.radius, self.L)
        elif name == "GounarisSakurai":
            own = agl.gounaris_sakurai(s, s1, s2, self.mass, self.width, self.radius, self.L, mpi)
        else:
            raise NotImplementedError(f"Lineshape '{name}' of {self.name} is not implemented")
        return own * self.daughters[0].lineshape(p, mpi) * self.daughters[1].lineshape(p, mpi)

    def spin_tensor(self, p):
        if not self.daughters:
            return atfi.ones(p[0][..., 0])
        if len(self.daughters) == 1:
            return self.daughters[0].spin_tensor(p)
        d0, d1 = self.daughters
        name = f"{SPIN_LETTER[self.two_spin]}_{SPIN_LETTER[d0.two_spin]}{SPIN_LETTER[d1.two_spin]}_{'SPDF'[self.L]}"
        return cov.vertex(name, self.momentum(p), d0.momentum(p) - d1.momentum(p),
                          d0.spin_tensor(p), d1.spin_tensor(p))

    def amplitude(self, p, mpi):
        """Lineshapes x spin factor, without couplings (AmpGen ``MatrixElement``)."""
        return self.lineshape(p, mpi) * atfi.cast_complex(self.spin_tensor(p))


@dataclass
class Chain:
    """A fully expanded decay chain and the couplings that multiply it."""
    particle: Particle
    rules: list  # names of the coupling rules whose product is the chain coupling

    @property
    def name(self):
        return self.particle.descriptor()


class AmpGenModel:
    """Coherent sum of the decay chains defined in an AmpGen options file."""

    def __init__(self, options_file, particles_file):
        self.options = read_options(options_file)
        self.table = load_particles(particles_file)
        self.head, *self.final_state = self.options.event_type
        self.params = {k: v for k, (v, _) in self.options.parameters.items()}
        self.couplings = {k: v for k, (v, _) in self.options.couplings.items()}
        self.free_couplings = [k for k, (_, free) in self.options.couplings.items() if free]
        self.mpi = self.table["pi+"]["mass"]
        self.chains = []
        for rule in self.options.couplings:
            decay = parse_descriptor(rule)
            if decay.name == self.head:
                for expanded, rules in self._expand(decay):
                    particle = Particle(expanded, self.table, self.params, self.final_state, is_head=True)
                    self.chains.append(Chain(particle, [rule] + rules))
        if not self.chains:
            raise ValueError(f"No decays of {self.head} in {options_file}")

    def _rules_for(self, name):
        return [r for r in self.options.couplings if parse_descriptor(r).name == name]

    def _expand(self, decay):
        """Expand particles without explicit daughters using their own decay rules."""
        if not decay.daughters:
            rules = self._rules_for(decay.name) if decay.name not in self.final_state else []
            if not rules:
                return [(decay, [])]
            out = []
            for rule in rules:
                sub = parse_descriptor(rule)
                sub.modifiers = decay.modifiers + sub.modifiers
                for expanded, used in self._expand(sub):
                    out.append((expanded, [rule] + used))
            return out
        combos = [(Decay(decay.name, list(decay.modifiers)), [])]
        for d in decay.daughters:
            new = []
            for parent, used in combos:
                for dexp, dused in self._expand(d):
                    new.append((Decay(parent.name, parent.modifiers, parent.daughters + [dexp]), used + dused))
            combos = new
        return combos

    @staticmethod
    def parameter_name(rule):
        """A name for ``rule`` usable as a TensorFlow variable / fit parameter name."""
        name = rule.replace("*", "st").replace("+", "p").replace("-", "m").replace("~", "")
        return re.sub(r"[^A-Za-z0-9_.]", "_", name).strip("_")

    @property
    def chain_names(self):
        return [c.name for c in self.chains]

    def chain_amplitudes(self, p):
        """{chain name: amplitude without couplings} at final-state momenta ``p``
        (a list of ``[N, 4]`` tensors ordered as the event type)."""
        return {c.name: c.particle.amplitude(p, self.mpi) for c in self.chains}

    def chain_coupling(self, chain, couplings=None):
        couplings = self.couplings if couplings is None else {**self.couplings, **couplings}
        value = couplings[chain.rules[0]]
        for rule in chain.rules[1:]:
            value = value * couplings[rule]
        return value

    def amplitude(self, p, couplings=None, amplitudes=None):
        """Total amplitude. ``couplings`` overrides rule couplings by name (complex
        constants or tensors); ``amplitudes`` can pass pre-computed chain amplitudes."""
        amplitudes = self.chain_amplitudes(p) if amplitudes is None else amplitudes
        total = None
        for chain in self.chains:
            c = self.chain_coupling(chain, couplings)
            c = c if isinstance(c, tf.Tensor) else atfi.complex(atfi.const(c.real), atfi.const(c.imag))
            term = c * amplitudes[chain.name]
            total = term if total is None else total + term
        return total

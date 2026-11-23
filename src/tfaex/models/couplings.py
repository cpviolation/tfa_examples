"""Coherent sums of named amplitude components.

A model is described by a dict ``{name: complex amplitude tensor}`` of its
components and a dict ``{name: complex coupling}``. Fit parameters follow the
convention ``<name>_re`` / ``<name>_im``, so that a TFA2 likelihood can pass
them as keyword arguments::

    pars = fit_parameters(initial_couplings, fixed=["rho770"])
    def _nll(pars):
        couplings = couplings_from_params(pars, names)
        ...

TFA2 (``run_toymc``, ``calculate_fit_fractions``) identifies components with a
``switches`` list; :func:`switches_for` builds one from component names.
"""

import amplitf.interface as atfi


def coherent_sum(components, couplings, switches=None, fixed=()):
    """Sum of ``coupling * component`` over the enabled components.

    Args:
        components (dict): name -> complex amplitude tensor, in a fixed order.
        couplings (dict): name -> complex coupling, for every enabled component
            that is not in ``fixed``.
        switches (sequence, optional): 0/1 flag per component, in the order of
            ``components``. All enabled if None.
        fixed (sequence): components entering with coupling 1.
    """
    names = list(components)
    if switches is None:
        switches = len(names) * [1]
    if len(switches) != len(names):
        raise ValueError(f"{len(switches)} switches for {len(names)} components")
    ampl = atfi.complex(atfi.const(0.0), atfi.const(0.0)) * atfi.cast_complex(
        atfi.ones(atfi.real(next(iter(components.values()))))
    )
    for name, on in zip(names, switches):
        if not on:
            continue
        if name in fixed:
            ampl += components[name]
        elif name in couplings:
            ampl += couplings[name] * components[name]
        else:
            raise KeyError(f"No coupling given for component '{name}'")
    return ampl


def switches_for(names, enabled):
    """0/1 switches over ``names`` with only the ``enabled`` components on."""
    unknown = set(enabled) - set(names)
    if unknown:
        raise KeyError(f"Unknown components: {sorted(unknown)}")
    return [int(n in enabled) for n in names]


def couplings_from_params(params, names):
    """Complex couplings from real parameters ``<name>_re`` and ``<name>_im``.

    Names whose parameters are not in ``params`` are skipped.
    """
    return {
        n: atfi.complex(params[f"{n}_re"], params[f"{n}_im"])
        for n in names
        if f"{n}_re" in params and f"{n}_im" in params
    }


def fit_parameters(initial, limit=10.0, fixed=()):
    """TFA2 fit parameters ``<name>_re`` / ``<name>_im`` for each coupling.

    Args:
        initial (dict): name -> initial complex value (python or numpy complex).
        limit (float): parameters are bounded to [-limit, limit].
        fixed (sequence): names whose real and imaginary parts are fixed,
            typically the reference component.
    """
    import tfa.optimisation as tfo

    pars = []
    for name, value in initial.items():
        value = complex(value)
        for suffix, v in (("re", value.real), ("im", value.imag)):
            p = tfo.FitParameter(f"{name}_{suffix}", v, -limit, limit)
            if name in fixed:
                p.fix()
            pars.append(p)
    return pars

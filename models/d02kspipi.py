"""
Various models to describe D0->KSππ decay
- Simple
- Belle
- BaBar
"""
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm


def simple(x, phsp,
           md, mpi, mkz, 
           mkst, wkst,
           mrho, wrho,
           rd, rr):
    """A simple model in which amplitudes parameters are fixed

    Args:
        x (tf.tensor): the input data
        phsp (DalitzPhaseSpace): Dalitz phase space class to calculate invariant masses and angles
        md (tf.constant): the invariant mass of the D0 candidate
        mpi (tf.constant): the mass of the pion
        mkz (tf.constant): mass of the K0
        mkst (tf.constant): mass of the K*(892) resonance
        wkst (tf.constant): width of the K*(892) resonance
        mrho (tf.constant): mass of the rho(770) resonance
        wrho (tf.constant): width of the rho(770) resonance
        rd (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (D0)
        rr (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (resonance)

    Returns:
        tf.tensor: the value(s) of the amplitude
    """
    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    hel_ab = atfd.helicity_amplitude(phsp.cos_helicity_ab(x), 1)
    hel_bc = atfd.helicity_amplitude(phsp.cos_helicity_bc(x), 1)
    hel_ac = atfd.helicity_amplitude(phsp.cos_helicity_ac(x), 1)

    bw1 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw2 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw3 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, mpi, mpi, mkz, md, rd, rr, 1, 1)

    def _model(a1r, a1i, a2r, a2i, a3r, a3i, switches=4 * [1]):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl += a1 * bw1 * hel_ab
        if switches[1]:
            ampl += a2 * bw2 * hel_bc
        if switches[2]:
            ampl += a3 * bw3 * hel_ac
        if switches[3]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        return atfd.density(ampl)

    return _model


def simple_model_mix(x, tphsp, tdz,
           md, mpi, mkz, 
           mkst, wkst,
           mrho, wrho,
           rd, rr):
    """A simple model in which amplitudes parameters are fixed

    Args:
        x (tf.tensor): the input data
        tphsp (CombinedPhaseSpace): Dalitz phase space x Decay times class to calculate invariant masses and angles
        tdz (tf.constant): the lifetime of the D0
        md (tf.constant): the invariant mass of the D0 candidate
        mpi (tf.constant): the mass of the pion
        mkz (tf.constant): mass of the K0
        mkst (tf.constant): mass of the K*(892) resonance
        wkst (tf.constant): width of the K*(892) resonance
        mrho (tf.constant): mass of the rho(770) resonance
        wrho (tf.constant): width of the rho(770) resonance
        rd (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (D0)
        rr (tf.constant): Blatt-Weisskopf radius for Breit-Wigner lineshape (resonance)

    Returns:
        tf.tensor: the value(s) of the amplitude
    """

    # DZ - MIXING MODEL
    
    # CACHED VARIABLES
    m2ab = tphsp.phsp1.m2ab(x)
    m2bc = tphsp.phsp1.m2bc(x)
    m2ac = tphsp.phsp1.m2ac(x)

    hel_ab = atfd.helicity_amplitude(tphsp.phsp1.cos_helicity_ab(x), 1)
    hel_bc = atfd.helicity_amplitude(tphsp.phsp1.cos_helicity_bc(x), 1)
    hel_ac = atfd.helicity_amplitude(tphsp.phsp1.cos_helicity_ac(x), 1)

    bw1 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw2 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, mpi, mkz, mpi, md, rd, rr, 1, 1)
    bw3 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, mpi, mpi, mkz, md, rd, rr, 1, 1)

    def _model(a1r, a1i, a2r, a2i, a3r, a3i,
               x_mix_par, y_mix_par, qop_mag, qop_pha, 
               switches=4 * [1]):

        tep = atfm.psip(tphsp.phsp2.t(tphsp.data2(x)), y_mix_par, tdz)
        tem = atfm.psim(tphsp.phsp2.t(tphsp.data2(x)), y_mix_par, tdz)
        tei = atfm.psii(tphsp.phsp2.t(tphsp.data2(x)), x_mix_par, tdz)
        qoverp = atfi.complex(qop_mag * atfi.cos(qop_pha), 
                              qop_mag * atfi.sin(qop_pha))

        #a1r, a1i, a2r, a2i, a3r, a3i = atfi.const(1.0), atfi.const(0.0), atfi.const(0.5), atfi.const(0.0), atfi.const(2.0), atfi.const(0.0)
        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)

        # D0 AMPLITUDE
        ampl_dz = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl_dz += a1 * bw1 * hel_ab
        if switches[1]:
            ampl_dz += a2 * bw2 * hel_bc
        if switches[2]:
            ampl_dz += a3 * bw3 * hel_ac
        if switches[3]:
            ampl_dz += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        # D0bar AMPLITUDE: exchange a <--> c
        ampl_dzb = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl_dzb += a1 * bw2 * hel_bc
        if switches[1]:
            ampl_dzb += a2 * bw1 * hel_ab
        if switches[2]:
            ampl_dzb += -a3 * bw3 * hel_ac
        if switches[3]:
            ampl_dzb += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        # MIXING
        dens = atfm.mixing_density(ampl_dz, ampl_dzb, qoverp, tep, tem, tei)
        return dens

    return _model

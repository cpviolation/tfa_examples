import os, time
# import NumPy
import numpy as np
# Import Tensorflow
import tensorflow as tf
# Import AmpliTF modules
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace
#from amplitf.mixing import psip, psim, psii, mixing_density

# Import TFA modules
import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo

# Masses of final state particles
from particle.particle import literals as lp

# Import plotting module
import matplotlib.pyplot as plt

from tfaex import paths
from tfaex.io import load_belle_model
from tfaex.models import belle2018
from tfaex.models.d02kspipi import belle_cache_amp

# Import argparse for command line arguments
import argparse
# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Run toy MC for D0 -> Kspipi")
parser.add_argument("--ntoys", type=int, default=100000, help="Number of toys to generate")
parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("-x", type=float, default=0.004, help="Value of x for the mixing model")
parser.add_argument("-y", type=float, default=0.0064, help="Value of y for the mixing model")
parser.add_argument("-qop", type=float, default=1.0, help="Absolute value of q/p for CP violation")
parser.add_argument("-qop_phase", type=float, default=0.0, help="Phase of q/p for CP violation")
parser.add_argument("--output", type=str, default='d02kspipi_toy', help="Output file name (without extension) for the toy MC sample, written to $TFAEX_OUTPUT or <repo>/output")
parser.add_argument("--dryrun", action="store_true", help="Run without executing the main function")
args = parser.parse_args()

# define some global variables 
mkz = atfi.const(lp.K_S_0.mass/1000)
mpi = atfi.const(lp.pi_plus.mass/1000)
md = atfi.const(lp.D_0.mass/1000)
meta = atfi.const(lp.eta.mass/1000.)
metap = atfi.const(lp.etap_958.mass/1000.)
belle_model = load_belle_model()

#Define the phase space for the decay D0 -> Kspipi.
# Create Dalitz Phase Space
phsp = DalitzPhaseSpace(mpi, mkz, mpi, md)
# Decay Time Phase Space
tdz = atfi.const(1.)
tphsp = DecayTimePhaseSpace(tdz)
# Combined Phase Space
c_phsp = CombinedPhaseSpace(phsp, tphsp)

def belle_cache_model(x):
    return belle_cache_amp(x, phsp, paths.BELLE_CACHE)

def Af(x, switches=belle2018.DEFAULT_SWITCHES):
    return belle2018.amplitude(x, phsp, belle_model, switches)

def Afbar(x):
    # the conjugate of the amplitude
    # is the same as the amplitude with the masses swapped
    return Af(x[:,::-1])


def mixing_model(x):
    # Calculate the amplitudes - cached
    #ampl_dz = Af(c_phsp.data1(x))
    #ampl_dzb = Afbar(c_phsp.data1(x))
    #
    ampl_dz = belle_cache_model(c_phsp.data1(x))
    # swap m2(KS pi+) <-> m2(KS pi-) only; reversing x would also move the decay time column
    ampl_dzb = belle_cache_model(c_phsp.data1(x)[:, ::-1])
    # Calculate the model from the mixing parameters and time evolution operators
    def _model(x_mix_par, y_mix_par, qoverp_re, qoverp_im):
        t = c_phsp.phsp2.t(c_phsp.data2(x))
        tep = atfm.psip(t, y_mix_par, atfi.const(1.0))
        tem = atfm.psim(t, y_mix_par, atfi.const(1.0))
        tei = atfm.psii(t, x_mix_par, atfi.const(1.0))
        # calculate the amplitude density
        dens = atfm.mixing_density(ampl_dz, ampl_dzb, atfi.complex(qoverp_re, qoverp_im), tep, tem, tei)
        return dens
    return _model


def toymc_mixing_model(x):
    return mixing_model(x)(
        x_mix_par=atfi.const(args.x),
        y_mix_par=atfi.const(args.y),
        qoverp_re=atfi.const(args.qop) * atfi.cos(atfi.const(args.qop_phase)),
        qoverp_im=atfi.const(args.qop) * atfi.sin(atfi.const(args.qop_phase))
        )

def apply_acceptance(data):
    
    return 


def main():
    start_time = time.time()
    toy_sample = tft.run_toymc(
        toymc_mixing_model, c_phsp, args.ntoys, maximum=1.0e-20, chunk=1000000, components=False
    )
    end_time = time.time()
    print(f"Generated {args.ntoys} toys in {end_time - start_time:.2f} seconds.")
    # save data to file
    file_name = paths.output_dir() / (args.output + '.npy')
    np.save(file_name, toy_sample.numpy())
    print(f"Saved toy sample to {file_name}")
    
    return

if __name__ == "__main__":
    if not args.dryrun:
        main()


### TODO:
# Add a function to apply reconstruction efficiencies to the toy MC sample
# Add a function to generate generator-level data on-demand
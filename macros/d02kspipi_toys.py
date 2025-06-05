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

#from context import models
from models.d02kspipi import babar2008_model_amp
from models.helpers import decode_model, plot_data, plot_data_mix, plot_data_comparison_mix

# Import argparse for command line arguments
import argparse
# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Run toy MC for D0 -> Kspipi")
parser.add_argument("--ntoys", type=int, default=100000, help="Number of toys to generate")
parser.add_argument("-x", type=float, default=0.004, help="Value of x for the mixing model")
parser.add_argument("-y", type=float, default=0.0064, help="Value of y for the mixing model")
parser.add_argument("-qop", type=float, default=1.0, help="Absolute value of q/p for CP violation")
parser.add_argument("-qop_phase", type=float, default=0.0, help="Phase of q/p for CP violation")
parser.add_argument("--dryrun", action="store_true", help="Run without executing the main function")
args = parser.parse_args()


def define_phasespace():
    """
    Define the phase space for the decay D0 -> Kspipi.
    """
    # Dalitz Particles
    mkz = atfi.const(lp.K_S_0.mass/1000)
    mpi = atfi.const(lp.pi_plus.mass/1000)
    md = atfi.const(lp.D_0.mass/1000)
    # Create Dalitz Phase Space
    phsp = DalitzPhaseSpace(mpi, mkz, mpi, md)
    # Decay Time Phase Space
    tdz = atfi.const(1.)
    tphsp = DecayTimePhaseSpace(tdz)
    # Combined Phase Space
    c_phsp = CombinedPhaseSpace(phsp, tphsp)
    return c_phsp

def get_mixing_model():
    meta = atfi.const(lp.eta.mass/1000.)
    metap = atfi.const(lp.etap_958.mass/1000.)
    belle_model = decode_model('../../d02kspipi_toys/generator/inputs/belle_model.txt')

def main():
    # Define the phase space
    c_phsp = define_phasespace()
    # Define the mixing parameters
    x_mix = atfi.const(args.x)
    y_mix = atfi.const(args.y)
    qop_mix = atfi.const(args.qop)
    phi_mix = atfi.const(args.qop_phase)  # Phase of q/p for CP violation
    qoverp = atfi.complex( qop_mix * atfi.cos(phi_mix), 
                           qop_mix * atfi.sin(phi_mix) )
    
    return

if __name__ == "__main__":
    if not args.dryrun:
        main()
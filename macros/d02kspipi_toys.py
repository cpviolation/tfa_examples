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

#from context import models
from context import models
from models.d02kspipi import babar2008_model_amp
from models.helpers import decode_model, plot_data, plot_data_mix, plot_data_comparison_mix

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
parser.add_argument("--output", type=str, default='d02kspipi_toy', help="Output file name for the toy MC sample")
parser.add_argument("--dryrun", action="store_true", help="Run without executing the main function")
args = parser.parse_args()

# define some global variables 
mkz = atfi.const(lp.K_S_0.mass/1000)
mpi = atfi.const(lp.pi_plus.mass/1000)
md = atfi.const(lp.D_0.mass/1000)
meta = atfi.const(lp.eta.mass/1000.)
metap = atfi.const(lp.etap_958.mass/1000.)
belle_model = decode_model(os.environ['TFAEX_ROOT']+'/../d02kspipi_toys/generator/inputs/belle_model.txt')

#Define the phase space for the decay D0 -> Kspipi.
# Create Dalitz Phase Space
phsp = DalitzPhaseSpace(mpi, mkz, mpi, md)
# Decay Time Phase Space
tdz = atfi.const(1.)
tphsp = DecayTimePhaseSpace(tdz)
# Combined Phase Space
c_phsp = CombinedPhaseSpace(phsp, tphsp)

def babar_model_amp(x):
    return babar2008_model_amp(x, phsp,
        atfi.const(belle_model['rho770_Mass'][0]),
        atfi.const(belle_model['rho770_Width'][0]),
        atfi.const(belle_model['Kstar892_Mass'][0]),
        atfi.const(belle_model['Kstar892_Width'][0]),
        atfi.const(belle_model['Kstartwo1430_Mass'][0]),
        atfi.const(belle_model['Kstartwo1430_Width'][0]),
        atfi.const(belle_model['Kstar1410_Mass'][0]),
        atfi.const(belle_model['Kstar1410_Width'][0]),
        atfi.const(belle_model['Kstar1680_Mass'][0]),
        atfi.const(belle_model['Kstar1680_Width'][0]),
        atfi.const(belle_model['omega_Mass'][0]),
        atfi.const(belle_model['omega_Width'][0]),
        atfi.const(belle_model['ftwo1270_Mass'][0]),
        atfi.const(belle_model['ftwo1270_Width'][0]),
        atfi.const(belle_model['rho1450_Mass'][0]),
        atfi.const(belle_model['rho1450_Width'][0]),
        # LASS
        atfi.const(belle_model['LASS_a'][0]),
        atfi.const(belle_model['LASS_r'][0]),
        atfi.const(1.4617),
        atfi.const(0.2683),
        atfi.const(belle_model['LASS_R'][0]),
        atfi.const(belle_model['LASS_phi_R'][0]),
        atfi.const(belle_model['LASS_F'][0]),
        atfi.const(belle_model['LASS_phi_F'][0]),
        # K matrix model parameters
        atfi.const( [0.651, 1.2036, 1.55817, 1.21, 1.82206] ),
        atfi.const( [ [0.22889, -0.55377, 0, -0.39899, -0.34639],
                        [0.94128, 0.55095, 0, 0.39065, 0.31503],
                        [0.36856, 0.23888, 0.55639, 0.18340, 0.18681],
                        [0.33650, 0.40907, 0.85679, 0.19906, -0.00984],
                        [0.18171, -0.17558, -0.79658, -0.00355, 0.22358]] ),
        atfi.const(-3.92637),
        atfi.const([ [  0.23399,  0.15044, -0.20545,  0.32825,  0.35412],
                   [  0.15044, 0, 0, 0, 0],
                   [ -0.20545, 0, 0, 0, 0],
                   [  0.32825, 0, 0, 0, 0],
                   [  0.35412, 0, 0, 0, 0]]),
        atfi.stack([belle_model[f'Kmatrix_beta{i}_realpart'][0]+\
                    1.j*belle_model[f'Kmatrix_beta{i}_imaginarypart'][0] for i in range(1, 6)]),
        atfi.const(-0.070000000000000),
        atfi.stack([belle_model[f'Kmatrix_f_prod_1{i}_realpart'][0]+\
                    1.j*belle_model[f'Kmatrix_f_prod_1{i}_imaginarypart'][0] for i in range(1, 6)]),
        [[mpi,mpi], [mkz, mkz], [mpi], [meta, meta], [meta, metap]])

def Af(x, switches=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]):#15 * [1]):
    return babar_model_amp(x)(
        switches=switches,
        a1r=atfi.const(1.0), 
        a1i=atfi.const(0.0),
        a2r=atfi.const(belle_model['Kstar892minus_realpart'][0]),
        a2i=atfi.const(belle_model['Kstar892minus_imaginarypart'][0]),
        a3r=atfi.const(belle_model['Kstarzero1430minus_realpart'][0]),
        a3i=atfi.const(belle_model['Kstarzero1430minus_imaginarypart'][0]),
        a4r=atfi.const(belle_model['Kstartwo1430minus_realpart'][0]),
        a4i=atfi.const(belle_model['Kstartwo1430minus_imaginarypart'][0]),
        a5r=atfi.const(belle_model['Kstar1410minus_realpart'][0]),
        a5i=atfi.const(belle_model['Kstar1410minus_imaginarypart'][0]),
        a6r=atfi.const(belle_model['Kstar1680minus_realpart'][0]),
        a6i=atfi.const(belle_model['Kstar1680minus_imaginarypart'][0]),
        a7r=atfi.const(belle_model['Kstar892plus_realpart'][0]),
        a7i=atfi.const(belle_model['Kstar892plus_imaginarypart'][0]),
        a8r=atfi.const(belle_model['Kstarzero1430plus_realpart'][0]),
        a8i=atfi.const(belle_model['Kstarzero1430plus_imaginarypart'][0]),
        a9r=atfi.const(belle_model['Kstartwo1430plus_realpart'][0]),
        a9i=atfi.const(belle_model['Kstartwo1430plus_imaginarypart'][0]),
        a10r=atfi.const(belle_model['Kstar1410plus_realpart'][0]),
        a10i=atfi.const(belle_model['Kstar1410plus_imaginarypart'][0]),
        a11r=atfi.const(belle_model['Kstar1680plus_realpart'][0]),
        a11i=atfi.const(belle_model['Kstar1680plus_imaginarypart'][0]),
        a12r=atfi.const(belle_model['omega_realpart'][0]),
        a12i=atfi.const(belle_model['omega_imaginarypart'][0]),
        a13r=atfi.const(belle_model['ftwo1270_realpart'][0]),
        a13i=atfi.const(belle_model['ftwo1270_imaginarypart'][0]),
        a14r=atfi.const(belle_model['rho1450_realpart'][0]),
        a14i=atfi.const(belle_model['rho1450_imaginarypart'][0]),
    )

def Afbar(x):
    # the conjugate of the amplitude
    # is the same as the amplitude with the masses swapped
    return Af(x[:,::-1])


def mixing_model(x):
    # Calculate the amplitudes - cached
    ampl_dz = Af(c_phsp.data1(x))
    ampl_dzb = Afbar(c_phsp.data1(x))
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
    file_name = os.environ['TFAEX_ROOT']+'/../output/'+args.output+'.npy'
    np.save(file_name,toy_sample.numpy())
    
    return

if __name__ == "__main__":
    if not args.dryrun:
        main()


### TODO:
# Add a function to apply reconstruction efficiencies to the toy MC sample
# Add a function to generate generator-level data on-demand
import os
import numpy as np
import argparse
import uproot
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Run bin flip fitter for D0 -> Kspipi")
parser.add_argument("--nbins", type=int, default=10, help="Number of decay time bins")
parser.add_argument("--dryrun", action="store_true", help="Run without executing the main function")
args = parser.parse_args()


def order_dataset(data):
    # Order the dataset by decay time
    return data[data[:, 2].argsort()]


def add_binflip_binid(data):
    hf = uproot.open(os.environ['TFAEX_ROOT']+'/notebooks/files/BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD.root')
    hnp = hf['h_BinningLUT_K0Spipi_BABAR2008_EqualDeltadeltaD'].to_numpy()
    ix = np.digitize(data[:,0], hnp[1]) - 1
    iy = np.digitize(data[:,1], hnp[2]) - 1
    binid = hnp[0][ix, iy] * (2 * (ix>iy) - 1)  # Flip sign if ix > iy
    data = np.column_stack((data, binid))
    return data


def calculate_ratios(data, ndtbins=10):
    # Split the data into decay time bins
    data_dt = np.array_split(data, ndtbins)
    Np = [[ len(data_dt_sam[data_dt_sam[:,3]==binid]) for data_dt_sam in data_dt ] for binid in range(1,9)]
    Nm = [[ len(data_dt_sam[data_dt_sam[:,3]==-binid]) for data_dt_sam in data_dt ] for binid in range(1,9)]
    rat = np.divide(Np, Nm, out=np.zeros_like(Np, dtype=float), where=Nm!=0)
    rat_unc = np.divide(Np, np.power(Nm,2), out=np.zeros_like(Np, dtype=float), where=Nm!=0)
    rat_unc+= np.divide(np.power(Np,2), np.power(Nm,3), out=np.zeros_like(Np, dtype=float), where=Nm!=0)
    rat_unc = np.sqrt(rat_unc)
    return rat, rat_unc


def calculate_decay_time_centers(data, ndtbins=10):
    # Calculate the centers of the decay time bins
    data_dt = np.array_split(data, ndtbins)
    dt_centers = [np.mean(data_dt_bin[:,2] ) for data_dt_bin in data_dt]
    return dt_centers


def plot_ratios(data, ndtbins=10):
    ratios = calculate_ratios(data, ndtbins=ndtbins)
    dt_centers = calculate_decay_time_centers(data, ndtbins=ndtbins)
    fig, ax = plt.subplots(4,2, figsize=(10, 8), sharex=True, sharey=True)
    for i in range(8):
        ax[i//2, i%2].errorbar(dt_centers, ratios[0][i], yerr=ratios[1][i], fmt='o', label=f'Bin {i}')
        ax[i//2, i%2].set_title(f'Bin {i}')
        ax[i//2, i%2].set_xlabel('Decay Time (ps)')
        ax[i//2, i%2].set_ylabel('Ratio N+/N-')
        ax[i//2, i%2].legend()
    plt.tight_layout()
    return fig, ax


def main():
    data = np.load(os.environ['TFAEX_ROOT']+'/../output/d02kspipi_toy.npy')
    data = order_dataset(data)
    data = np.array_split(data, args.nbins)
    data = add_binflip_binid(data)
    return


if __name__ == "__main__":
    if not args.dryrun:
        main()
    else:
        print("Dry run mode: main function will not be executed.")
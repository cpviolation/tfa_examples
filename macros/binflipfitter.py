"""Bin-flip ratios R_b(t) = N(+b) / N(-b) versus decay time for a D0 -> KS pi pi toy sample."""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from tfaex import paths
from tfaex.io import add_binflip_binid


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None,
                        help="Toy sample (.npy); defaults to <output dir>/d02kspipi_toy.npy")
    parser.add_argument("--nevs", type=int, default=-1, help="Number of events to process")
    parser.add_argument("--nbins", type=int, default=10, help="Number of decay time bins")
    parser.add_argument("--dryrun", action="store_true", help="Parse the arguments and exit")
    return parser.parse_args(argv)


def order_dataset(data):
    # Order the dataset by decay time
    return data[data[:, 2].argsort()]


def calculate_ratios(data, nbins):
    # Split the data into decay time bins
    data_dt = np.array_split(data, nbins)
    Np = [[ len(data_dt_sam[data_dt_sam[:,3]==binid]) for data_dt_sam in data_dt ] for binid in range(1,9)]
    Nm = [[ len(data_dt_sam[data_dt_sam[:,3]==-binid]) for data_dt_sam in data_dt ] for binid in range(1,9)]
    rat = np.divide(Np, Nm, out=np.zeros_like(Np, dtype=float), where=Nm!=0)
    rat_unc = np.divide(Np, np.power(Nm,2), out=np.zeros_like(Np, dtype=float), where=Nm!=0)
    rat_unc+= np.divide(np.power(Np,2), np.power(Nm,3), out=np.zeros_like(Np, dtype=float), where=Nm!=0)
    rat_unc = np.sqrt(rat_unc)
    return rat, rat_unc


def calculate_decay_time_centers(data, nbins):
    # Calculate the centers of the decay time bins
    data_dt = np.array_split(data, nbins)
    dt_centers = [np.mean(data_dt_bin[:,2] ) for data_dt_bin in data_dt]
    return dt_centers


def plot_ratios(data, nbins):
    ratios = calculate_ratios(data, nbins)
    dt_centers = calculate_decay_time_centers(data, nbins)
    fig, ax = plt.subplots(4,2, figsize=(10, 8), sharex=True)
    for i in range(8):
        ax[i//2, i%2].errorbar(dt_centers, ratios[0][i], yerr=ratios[1][i], fmt='o', label=f'Bin {i+1}')
        #ax[i//2, i%2].set_title(f'Bin {i+1}')
        if i >= 6:  # Solo per l'ultima riga (i=6,7)
            ax[i//2, i%2].set_xlabel('Decay Time (ps)')
        ax[i//2, i%2].set_ylabel(f'$R_{i+1}$')
        if i % 2 == 1:  # Pannelli con indice dispari (colonna destra, i%2 == 1)
            ax[i//2, i%2].yaxis.set_label_position("right")
            ax[i//2, i%2].yaxis.tick_right()
        #ax[i//2, i%2].set_ylim(np.min(ratios[0][i] - 3.*ratios[1][i]), np.max(ratios[0][i] + 3.*ratios[1][i]))
        ax[i//2, i%2].legend()
    ax[0, 0].set_xlim(-0.5,8.5)
    plt.subplots_adjust(hspace=0, wspace=0.)
    #plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    return fig, ax


def plot_data(data):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].hist(data[:,2], bins=50, alpha=0.7, label='Decay Time Distribution')
    ax[0].set_xlabel('Decay Time (ps)')
    ax[0].set_ylabel('Counts')
    ax[0].set_yscale('log')
    #ax[0].set_title('Decay Time Distribution for D0 -> Kspipi')
    #ax[0].legend()
    ax[1].hist2d(data[:,0], data[:,1], bins=50, cmin=1)
    ax[1].set_xlabel(r'$m_- (GeV/c^2)$')
    ax[1].set_ylabel(r'$m_+ (GeV/c^2)$')
    #ax[1].set_title('2D Distribution')
    plt.tight_layout()
    return fig, ax


def main(argv=None):
    args = parse_args(argv)
    if args.dryrun:
        print(args)
        return
    data = np.load(args.input or paths.output_dir(create=False) / "d02kspipi_toy.npy")
    if 0 < args.nevs < len(data):
        data = data[: args.nevs]
    data = order_dataset(data)
    data = add_binflip_binid(data)
    plot_ratios(data, args.nbins)
    plt.show()


if __name__ == "__main__":
    main()

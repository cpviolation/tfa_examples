import matplotlib.pyplot as plt
import tfa.plotting as tfp

def decode_model(fname):
    parameter_names = {}
    with open('files/belle_model_definitions.txt', 'r') as f:
        for l in f:
            values = l.split()
            if len(values) == 0:
                continue
            parameter_names[values[1]] = values[0]    
    model = {}
    with open(fname, 'r') as f:
        for l in f:
            values = l.split()
            if len(values) == 0:
                continue
            model[parameter_names[values[0]]] = [float(v) for v in values[1:]]
    return model



def plot_data(data, phsp):
    tfp.set_lhcb_style(size=12, usetex=False)  # Adjust plotting style for LHCb papers
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))  # Single subplot on the figure

    # Plot 1D histogram from the toy MC sample
    tfp.plot_distr2d(
        data[:, 0],
        data[:, 1],
        bins=(50, 50),
        ranges=((0.3, 3.1), (0.3, 3.1)),
        fig=fig,
        ax=ax[0, 0],
        labels=(r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^-)$"),
        units=("MeV$^2$", "MeV$^2$"),
        log=True,
    )

    tfp.plot_distr1d(
        data[:, 0],
        bins=50,
        range=(0.3, 3.1),
        ax=ax[0, 1],
        label=r"$m^2(K_S^0\pi^+)$",
        units="MeV$^2$",
    )

    tfp.plot_distr1d(
        data[:, 1],
        bins=50,
        range=(0.3, 3.1),
        ax=ax[1, 0],
        label=r"$m^2(K_S^0\pi^-)$",
        units="MeV$^2$",
    )

    tfp.plot_distr1d(
        phsp.m2ac(data),
        bins=50,
        range=(0.05, 1.9),
        ax=ax[1, 1],
        label=r"$m^2(\pi^+\pi^-)$",
        units="MeV$^2$",
    )

    # Show the plot
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    return fig, ax

def plot_data_mix(data, c_phsp):
    tfp.set_lhcb_style(size=12, usetex=False)  # Adjust plotting style for LHCb papers
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 9))  # Single subplot on the figure

    # Plot 1D histogram from the toy MC sample
    tfp.plot_distr2d(
        c_phsp.data1(data)[:, 0],
        c_phsp.data1(data)[:, 1],
        bins=(50, 50),
        ranges=((0.3, 3.1), (0.3, 3.1)),
        fig=fig,
        ax=ax[0, 0],
        labels=(r"$m^2(K_S^0\pi^-)$", r"$m^2(K_S^0\pi^+)$"),
        units=("MeV$^2$", "MeV$^2$"),
        log=True,
    )

    tfp.plot_distr1d(
        c_phsp.data1(data)[:, 0],
        bins=50,
        range=(0.3, 3.1),
        ax=ax[0, 1],
        label=r"$m^2(K_S^0\pi^-)$",
        units="MeV$^2$",
    )

    tfp.plot_distr1d(
        c_phsp.data1(data)[:, 1],
        bins=50,
        range=(0.3, 3.1),
        ax=ax[1, 0],
        label=r"$m^2(K_S^0\pi^+)$",
        units="MeV$^2$",
    )

    tfp.plot_distr1d(
        c_phsp.phsp1.m2ac(c_phsp.data1(data)),
        bins=50,
        range=(0.05, 1.9),
        ax=ax[1, 1],
        label=r"$m^2(\pi^+\pi^-)$",
        units="MeV$^2$",
    )

    tfp.plot_distr1d(
        c_phsp.phsp2.t(c_phsp.data2(data)),
        bins=50,
        range=(0.05, 10),
        ax=ax[2, 0],
        log=True,
        label=r"$t/\tau$",
        units=None,
    )
    ax[2, 1].axis('off')  # Hide the unused subplot

    # Show the plot
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    return fig, ax

def plot_data_comparison_mix(data, c_phsp):
    tfp.set_lhcb_style(size=12, usetex=False)  # Adjust plotting style for LHCb papers
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))  # Single subplot on the figure

    tfp.plot_distr1d_comparison(
        c_phsp.data1(data[0])[:, 0],
        c_phsp.data1(data[1])[:, 0],
        bins=50,
        range=(0.3, 3.1),
        ax=ax[0, 0],
        label=r"$m^2(K_S^0\pi^-)$",
        units="MeV$^2$",
        legend=False
    )

    tfp.plot_distr1d_comparison(
        c_phsp.data1(data[0])[:, 1],
        c_phsp.data1(data[1])[:, 1],
        bins=50,
        range=(0.3, 3.1),
        ax=ax[0, 1],
        label=r"$m^2(K_S^0\pi^+)$",
        units="MeV$^2$",
        legend=False
    )

    tfp.plot_distr1d_comparison(
        c_phsp.phsp1.m2ac(c_phsp.data1(data[0])),
        c_phsp.phsp1.m2ac(c_phsp.data1(data[1])),
        bins=50,
        range=(0.05, 1.9),
        ax=ax[1, 0],
        label=r"$m^2(\pi^+\pi^-)$",
        units="MeV$^2$",
        legend=False
    )

    tfp.plot_distr1d_comparison(
        c_phsp.phsp2.t(c_phsp.data2(data[0])),
        c_phsp.phsp2.t(c_phsp.data2(data[1])),
        bins=50,
        range=(0.05, 10),
        ax=ax[1, 1],
        log=True,
        label=r"$t/\tau$",
        units=None,
    )

    # Show the plot
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    return fig, ax
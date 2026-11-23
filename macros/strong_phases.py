"""Compare the Belle model amplitude with the cached Belle amplitude, and compute the
strong phases c_b, s_b in the bin-flip binning for a D0 -> KS pi pi toy sample.
"""

import argparse
import functools

import numpy as np
import uproot
import matplotlib.pyplot as plt

import amplitf.interface as atfi

from tfaex import paths
from tfaex.io import add_binflip_binid, load_belle_model
from tfaex.models import belle2018
from tfaex.models.d02kspipi import dalitz_phase_space

phsp = dalitz_phase_space()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None,
                        help="Toy sample (.npy) for the strong phase calculation")
    parser.add_argument("--dryrun", action="store_true", help="Parse the arguments and exit")
    return parser.parse_args(argv)


@functools.cache
def _belle_model():
    return load_belle_model()


def Af(x, switches=belle2018.DEFAULT_SWITCHES):
    return belle2018.amplitude(x, phsp, _belle_model(), switches)


def Afbar(x):
    # the D0bar amplitude is the D0 one with m2(KS pi+) <-> m2(KS pi-)
    return Af(x[:, ::-1])


def get_model_cache(model, xbins, ybins):
    # Create a grid of points at the center of each bin
    x_centers = (xbins[:-1] + xbins[1:]) / 2
    y_centers = (ybins[:-1] + ybins[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    grid_points = np.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
    # Filter points inside the phase space
    inside_mask = phsp.inside(grid_points)
    inside_indices = np.where(inside_mask)[0]
    
    # Initialize output arrays with zeros
    amp_full = np.zeros(len(grid_points))
    amp2_full = np.zeros(len(grid_points))
    phase_full = np.zeros(len(grid_points))
    
    # Calculate the amplitude and phase only for points inside the phase space
    if len(inside_indices) > 0:
        grid_points_inside = grid_points[inside_indices]
        amp_inside, amp2_inside, phase_inside = get_amplitude_and_phase_from_model(grid_points_inside, model)
        amp_full[inside_indices] = amp_inside
        amp2_full[inside_indices] = amp2_inside
        phase_full[inside_indices] = phase_inside
    
    return amp_full.reshape(x_grid.shape), amp2_full.reshape(x_grid.shape), phase_full.reshape(x_grid.shape), xbins, ybins


def get_belle_cache():
    belle_cache = uproot.open(paths.BELLE_CACHE)
    amp, xbins, ybins = belle_cache['histo_amplitude'].to_numpy()
    amp2, _, _ = belle_cache['histo_amplitude_squared'].to_numpy()
    phase, _, _ = belle_cache['histo_phase'].to_numpy()
    return amp, amp2, phase, xbins, ybins


def get_amplitude_and_phase_from_model(x, model):
    # Calculate the amplitude on  data
    phsp_data = x[:, :2]
    a = model(phsp_data)
    amp2 = atfi.real(a * atfi.conjugate(a)).numpy()
    amp = atfi.sqrt(amp2)
    phase = atfi.atan2(atfi.imaginary(a), atfi.real(a)).numpy()
    return amp, amp2, phase


def get_strong_phase_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    strong_phases = phase[y_bin_indices, x_bin_indices]
    return strong_phases


def get_delta_strong_phase_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    delta_strong_phases = phase[y_bin_indices, x_bin_indices] - phase[x_bin_indices, y_bin_indices]  # Subtract the phase of the same bin (this is a placeholder for actual logic)
    return delta_strong_phases


def get_amp_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    amp_values = amp[y_bin_indices, x_bin_indices]
    return amp_values


def get_ampsq_from_cache(x, belle_cache=None):
    if belle_cache is None:
        belle_cache = get_belle_cache()
    amp, amp2, phase, xbins, ybins = belle_cache
    # Find the bin for each event
    x_bin_indices = np.digitize(x[:, 0], xbins) - 1
    y_bin_indices = np.digitize(x[:, 1], ybins) - 1
    # Clip indices to be within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, len(xbins)-2)
    y_bin_indices = np.clip(y_bin_indices, 0, len(ybins)-2)
    # Get the strong phase for each event from the cache
    amp2_values = amp2[y_bin_indices, x_bin_indices]
    return amp2_values


def plot_2d_binned_array(array_2d, xbins, ybins, xlabel="X", ylabel="Y", title="2D Binned Data", cmap='RdBu_r', vcenter=0):
    """Plot a 2D array with bin edges as axes.
    
    Args:
        array_2d: 2D numpy array with shape (len(ybins)-1, len(xbins)-1)
        xbins: bin edges for x-axis
        ybins: bin edges for y-axis
        xlabel: label for x-axis
        ylabel: label for y-axis
        title: plot title
        cmap: colormap name (default 'RdBu_r' for diverging colormap)
        vcenter: value mapped to center of colormap (white) - default 0
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    from matplotlib.colors import TwoSlopeNorm, Normalize
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin, vmax = np.nanmin(array_2d), np.nanmax(array_2d)
    
    # Check if data contains both positive and negative values
    has_positive = vmax > vcenter
    has_negative = vmin < vcenter
    
    if has_positive and has_negative:
        # Use diverging colormap with white at vcenter (mixed signs)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        used_cmap = cmap
    elif has_positive:
        # All positive: white at 0, gradient to blue
        norm = Normalize(vmin=vcenter, vmax=vmax)
        used_cmap = 'Blues'
    else:
        # All negative: gradient from blue to white at 0
        norm = Normalize(vmin=vmin, vmax=vcenter)
        used_cmap = 'Blues_r'
    
    mesh = ax.pcolormesh(xbins, ybins, array_2d, cmap=used_cmap, norm=norm, shading='auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = plt.colorbar(mesh, ax=ax)
    return fig, ax


def plot_strong_phase_by_bin(x, delta_strong):
    sort_idx = np.argsort(x[:, 2])
    x_sorted = x[sort_idx]
    ds_sorted = delta_strong[sort_idx]
    bin_ids = x_sorted[:, 2].astype(int)
    unique_bins, bin_starts = np.unique(bin_ids, return_index=True)
    bin_starts = np.append(bin_starts, len(bin_ids))
    fig, ax = plt.subplots()
    for i in range(8):
        ax.hist(ds_sorted[bin_starts[i]:bin_starts[i+1]], bins=100, range=(-np.pi,np.pi), alpha=0.5)
    ax.set_xlabel("Strong Phase Difference")
    ax.set_ylabel("Events")
    ax.set_title("Strong Phase Differences by Bin")
    #plt.savefig("strong_phase_by_bin.png")
    return fig, ax


def calculate_strong_phases(x):
    r"""Calculate the :math:`c_b` and :math:`s_b` strong phases average difference in the binflip bins by using the formulas

    ..math::

        \begin{align}
        c_i &= \frac{1}{F_i F_{-i}} \int_i |A_{D^0}(m^2_+, m^2_-)| |A_{D^0}(m^2_-, m^2_+)| \cos\left(\Delta\delta_D(m^2_+, m^2_-)\right) dm^2_+ dm^2_-
        s_i &= \frac{1}{F_i F_{-i}} \int_i |A_{D^0}(m^2_+, m^2_-)| |A_{D^0}(m^2_-, m^2_+)| \sin\left(\Delta\delta_D(m^2_+, m^2_-)\right) dm^2_+ dm^2_-
        \end{align}

    Args:
        x (float, float, int): data point with (m^2_+, m^2_-, bin_id)

    Returns:
        array: array with the strong phase differences for each bin
    """    
    # Sort data by bin_id for faster processing
    sort_idx = np.argsort(x[:, 3])
    x_sorted = x[sort_idx]
    
    # Calculate the amplitude on sorted data
    phsp_data = x_sorted[:, :2]
    ampl_dz = Af(phsp_data)
    ampl_dzb = Afbar(phsp_data)
    
    # Calculate phase difference: delta = arg(A_D0) - arg(A_D0bar)
    # This is equivalent to arg(A_D0 / A_D0bar) or arg(A_D0 * conj(A_D0bar))
    #delta_strong = atfi.atan2(atfi.imaginary(ampl_dz * atfi.conjugate(ampl_dzb)), 
    #                          atfi.real(ampl_dz * atfi.conjugate(ampl_dzb)))
    delta_strong = atfi.atan2(atfi.imaginary(ampl_dz), atfi.real(ampl_dz)) - atfi.atan2(atfi.imaginary(ampl_dzb), atfi.real(ampl_dzb))
    
    # Convert to numpy for faster bin operations
    delta_strong_np = delta_strong.numpy()
    bin_ids = x_sorted[:, 3].astype(int)
    
    # Calculate cb and sb for the binning scheme using vectorized operations
    unique_bins, bin_starts = np.unique(bin_ids, return_index=True)
    bin_starts = np.append(bin_starts, len(bin_ids))
    
    N = x.shape[0]
    for i in range(1, 9):
        Fpi = (bin_starts[8+i+1]-bin_starts[8+i])/N
        Fmi = (bin_starts[8-i+1]-bin_starts[8-i])/N
        Ddd = delta_strong_np[bin_starts[8+i]:bin_starts[8+i+1]]
        Adz = np.abs(ampl_dz.numpy()[bin_starts[8+i]:bin_starts[8+i+1]])/(Fpi+Fmi)/N
        Adzb = np.abs(ampl_dzb.numpy()[bin_starts[8+i]:bin_starts[8+i+1]])/(Fpi+Fmi)/N
        ci = np.sum(Adz * Adzb * np.cos(Ddd)) / np.sqrt(Fpi * Fmi)
        si = np.sum(Adz * Adzb * np.sin(Ddd)) / np.sqrt(Fpi * Fmi)
        print(f"Bin {i}: c = {ci:.6f}, s = {si:.6f}, N+ = {Fpi}, N- = {Fmi}") # NORMALISATION ISSUE
    
    # Return in original order
    delta_strong_original_order = np.empty_like(delta_strong_np)
    delta_strong_original_order[sort_idx] = delta_strong_np
    
    return delta_strong_original_order


def load_data(file_name):
    data = np.load(file_name)
    data = add_binflip_binid(data)
    return data


def get_uniform_phase_space_sample(n_events):
    data = phsp.uniform_sample(n_events)
    time = np.random.exponential(scale=1.0, size=len(data))
    data = np.insert(data, 2, time, axis=1)
    data = add_binflip_binid(data)
    return data


def main(argv=None):
    args = parse_args(argv)
    if args.dryrun:
        print(args)
        return

    # Compare the strong phases calculated from the cache with those calculated from the model
    belle_cache = get_belle_cache()
    xbins = belle_cache[3]
    ybins = belle_cache[4]
    model_cache = get_model_cache(Afbar, xbins, ybins) # need to swap x and y bins for the model cache to match the belle cache
    figs, axs = [],[]
    titles = ["Amplitude Difference (Model - Belle Cache)", "Amplitude Squared Difference (Model - Belle Cache)", "Phase Difference (Model - Belle Cache)"]
    for i in range(3):
        fig, ax = plot_2d_binned_array(model_cache[i]-belle_cache[i], xbins, ybins, title=titles[i], xlabel="$m^2_+$", ylabel="$m^2_-$")
        figs.append(fig)
        axs.append(ax)
    plt.show()

    # data = load_data(args.input)
    
    # # Calculate strong phase differences
    # print(f"Calculating strong phases for {len(data)} events...")
    # delta_strong_np = calculate_strong_phases(data)
    
    # # Print statistics
    # print(f"\nStrong phase difference statistics:")
    # print(f"  Mean: {np.mean(delta_strong_np):.4f} rad ({np.degrees(np.mean(delta_strong_np)):.2f}°)")
    # print(f"  Std Dev: {np.std(delta_strong_np):.4f} rad ({np.degrees(np.std(delta_strong_np)):.2f}°)")
    # print(f"  Min: {np.min(delta_strong_np):.4f} rad ({np.degrees(np.min(delta_strong_np)):.2f}°)")
    # print(f"  Max: {np.max(delta_strong_np):.4f} rad ({np.degrees(np.max(delta_strong_np)):.2f}°)")
    
    # # Optionally save results
    # if args.binning:
    #     output_file = f"strong_phases_{args.binning}.npy"
    #     np.save(output_file, delta_strong_np)
    #     print(f"\nSaved strong phases to {output_file}")
    
    return 

if __name__ == "__main__":
    main()

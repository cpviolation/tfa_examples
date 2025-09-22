import numpy as np
import matplotlib.pyplot as plt

from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
import amplitf.interface as atfi
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

@tf.function
def generalized_gamma(shape, d, p, a, dtype=tf.float32, seed=None):
    """
    Samples from the generalized gamma distribution.
    
    Parameters:
    - shape: output tensor shape
    - d: shape parameter (float or tensor)
    - p: power parameter (float or tensor)
    - a: scale parameter (float or tensor)
    - dtype: tf.float32 or tf.float64
    - seed: optional random seed
    
    Returns:
    - samples: Tensor of shape `shape`, dtype `dtype`
    """
    d = tf.convert_to_tensor(d, dtype=dtype)
    p = tf.convert_to_tensor(p, dtype=dtype)
    a = tf.convert_to_tensor(a, dtype=dtype)

    gamma_dist = tfd.Gamma(concentration=d, rate=1.0)
    u = gamma_dist.sample(shape, seed=seed)
    
    # Transform: X = a * U^(1/p)  
    x = a * tf.pow(u, 1.0 / p)
    return x

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gengamma
from scipy.integrate import simps

# Define your PDF (up to normalization)
def target_pdf(u, a, b, n, m, beta):
    return (u**a / (b + u**n))**m * np.exp(-beta * u)


def fit_gengamma_to_acceptance():
    # Parameters
    a = 2
    b = 1
    n = 3
    m = 1.5
    beta = 1.0
    t0 = 0.5

    # Step 1: Create a grid over u = t - t0
    u_grid = np.linspace(1e-5, 10, 1000)
    pdf_vals = target_pdf(u_grid, a, b, n, m, beta)

    # Step 2: Normalize PDF
    Z = simps(pdf_vals, u_grid)  # numerical integral
    pdf_vals /= Z

    # Step 3: Sample from the PDF (approximate inverse transform)
    # We'll fit the PDF to a generalized gamma
    from scipy.optimize import minimize

    # Generate a sample histogram as data
    rng = np.random.default_rng(42)
    cdf = np.cumsum(pdf_vals)
    cdf /= cdf[-1]
    inverse_cdf = lambda y: np.interp(y, cdf, u_grid)
    samples_from_pdf = np.array([inverse_cdf(y) for y in rng.uniform(0, 1, 10000)])

    # Step 4: Fit a generalized gamma to the samples
    shape_a, shape_c, loc, scale = gengamma.fit(samples_from_pdf, floc=0)

    print(f"Fitted generalized gamma parameters:")
    print(f"  shape c:     {shape_c}")
    print(f"  shape a:     {shape_a}")
    print(f"  loc:         {loc}")
    print(f"  scale:       {scale}")

    # Step 5: Sample from the fitted distribution
    samples = gengamma.rvs(c=shape_c, a=shape_a, loc=loc, scale=scale, size=10000)
    t_samples = t0 + samples  # shift back to original domain

    # Step 6: Plot comparison
    plt.figure(figsize=(8, 5))
    plt.hist(t_samples - t0, bins=100, density=True, alpha=0.5, label="Fitted gengamma samples")
    plt.plot(u_grid, pdf_vals, 'r-', label="Target PDF (normalized)")
    plt.xlabel("u = t - t0")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Comparison: Fitted Generalized Gamma vs Target PDF")
    plt.grid()
    plt.show()

    return shape_c, shape_a, loc, scale



def dt_acceptance(t, t0, a, b, beta, m, n):
    dt = t-t0
    return np.where( dt>=0, (dt**a / (b + dt**n))**m * np.exp(-beta * dt), 0)

def plot_dt_acceptance():

    t0 = 0.335
    a, b, beta, m, n = 1.14, 6.73, 1.27, 2.97, 0.17
    t = np.linspace(0, 10, 1000)
    y = dt_acceptance(t, t0, a, b, beta, m, n)

    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('Acceptance')
    plt.yscale('log')
    plt.title('Decay Time Acceptance Function')
    plt.grid()
    plt.show()
    return

def plot_dt_acceptance_tf():
    shape_c, shape_a, loc, scale = fit_gengamma_to_acceptance()
    t0 = 0.335
    size = 10000
    d, p, a = shape_a, shape_c, scale
    sample = generate_dt_acceptance_samples(size, t0, d, p, a)
    plt.hist(sample.numpy()[:, 0] + t0, bins=100, density=True, alpha=0.5, label="TF Samples")
    plt.xlabel("t")
    plt.ylabel("Density")
    plt.show()
    return


def generate_dt_acceptance_samples(size, t0, d, p, a):
    tdz = atfi.const(1.)
    tphsp = DecayTimePhaseSpace(tdz)
    # Generate samples from the generalized gamma distribution
    pars = (d, p, a)
    sample = tphsp.acceptance_sample(generalized_gamma, pars, size)
    return sample

def main():
    plot_dt_acceptance_tf()
    return

if __name__ == "__main__":
    main()
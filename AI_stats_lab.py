"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():

    analytic_gt5 = np.exp(-5)
    analytic_lt5 = 1 - np.exp(-5)
    analytic_interval = np.exp(-3) - np.exp(-7)

    n = 100000
    samples = np.random.exponential(scale=1, size=n)

    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():

    f = lambda x: 2 * x * np.exp(-x**2)

    integral_value, _ = quad(f, 0, np.inf)

    is_valid_pdf = bool(np.isclose(integral_value, 1.0))

    x = np.linspace(0, 3, 400)
    y = f(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("PDF: f(x) = 2x e^{-x^2}")
    # plt.show()

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():

    analytic_gt5 = np.exp(-5)
    analytic_interval = np.exp(-1) - np.exp(-3)

    n = 100000
    samples = np.random.exponential(scale=1, size=n)

    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)

    STEP 1
    Standardize variable

        Z = (X - 10)/2

    STEP 2
    Compute analytically

        P(X ≤ 12)
        P(8 < X < 12)

    STEP 3
    Simulate 100000 samples

    STEP 4
    Estimate probabilities

    RETURN

        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """

    # Analytical probabilities
    analytic_le12 = norm.cdf(12, loc=10, scale=2)
    analytic_interval = norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2)

    # Simulation
    samples = np.random.normal(loc=10, scale=2, size=100000)

    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval

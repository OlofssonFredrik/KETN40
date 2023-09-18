import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from functions import (
    obj_segments,
    const_segments,
    tubular_reactor_model,
    sim_segments,
    const,
)

params = {
    "v": 0.1,  # m/s
    "L": 1,  # m
    "alpha": 0.058,  # 1/s
    "beta": 0.2,  # 1/s
    "gamma": 16.7,  #
    "delta": 0.25,  #
    "cin": 0.02,  # mol/L
    "Tin": 340,  # K
    "c_lb": 0,
    "c_ub": 0.02,
    "T_lb": 280,
    "T_ub": 400,
    "Tw_lb": 280,
    "Tw_ub": 400,
}


# Bounds, constraints and initial guess
x_lb = (params["Tw_lb"] - params["Tin"]) / params["Tin"]
x_ub = (params["Tw_ub"] - params["Tin"]) / params["Tin"]
bounds = [[x_lb, x_ub]]
bounds_segments = [[x_lb, x_ub]] * 10
init_guess = x_ub * np.ones(10)
con_segments = {"type": "ineq", "fun": lambda u: const_segments(u, params)}


# Optimization
res = minimize(
    lambda u: obj_segments(u, params, print_vals=False),
    x0=init_guess,
    bounds=bounds_segments,
    method="SLSQP",
    constraints=con_segments,
)

# Print results
conversion = 1 - res.fun
Tjackets = params["Tin"] * (1 + res.x)
combined = np.vstack((Tjackets, res.x)).T
np.set_printoptions(suppress=True)
print(f"combined = {combined}")
print(f"conversion = {conversion}")

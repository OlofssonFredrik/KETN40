import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def tubular_reactor_model(t, y, u, params):
    x1 = y[0]
    x2 = y[1]
    dx1dz = (
        params["alpha"]
        / params["v"]
        * (1 - x1)
        * np.exp(params["gamma"] * x2 / (1 + x2))
    )
    dx2dz = params["alpha"] * params["delta"] / params["v"] * (1 - x1) * np.exp(
        params["gamma"] * x2 / (1 + x2)
    ) + params["beta"] / params["v"] * (u - x2)

    dx3dz = x2**2
    return np.array([dx1dz, dx2dz, dx3dz])


def sim_segments(u, params):
    y0 = np.zeros(3)
    segment_length = params["L"] / 10
    6
    ysol = np.zeros(
        (3, 11)
    )  # An array for the outlet concentrations and␣temperatures of all segments + the first inlet concentration
    zsol = np.zeros(
        11
    )  # An array for the z-coordinates of all segment outlets␣+ the inlet
    for i in range(10):
        zspan = [i * segment_length, (i + 1) * segment_length]
        sol = solve_ivp(
            lambda t, y: tubular_reactor_model(t, y, u[i], params), zspan, y0
        )

        ysol[:, i + 1] = sol.y[:, -1]
        y0 = sol.y[:, -1]
        zsol[i + 1] = (i + 1) * segment_length
    x1 = ysol[0, :]
    x2 = ysol[1, :]
    x3 = ysol[2, -1]
    T = params["Tin"] * (1 + x2)
    c = params["cin"] * (1 - x1)
    Tw = params["Tin"] * (1 + u)
    return zsol, c, T, Tw, x1, x2, x3


def obj_segments(u, params, print_vals=False, full_output=False):
    z, c, T, Tw, x1, x2, x3 = sim_segments(u, params)
    # print(f"x1: {x1}" + "\n")
    J = 1 - x1
    # print(f"J: {J}" + "\n")
    Q = J[-1]
    # print(f"Q: {Q}" + "\n")
    if print_vals == True:
        pass
        # print(u, Q)
    if full_output:
        return Q, x1
    return Q


def obj_segments_2(u, w1, params, print_vals=False, full_output=False):
    z, c, T, Tw, x1, x2, x3 = sim_segments(u, params)
    # print(f"x1: {x1}" + "\n")
    K1 = 1  # Can change later
    J = (1 - w1) * (1 - x1) + w1 * K1 * x2**2
    # print(f"J: {J}" + "\n")
    Q = J[-1]
    # print(f"Q: {Q}" + "\n")
    if print_vals == True:
        pass
        # print(u, Q)
    if full_output:
        # print(f"z: {z}" + "\n")
        return Q, x1, x2
    else:
        return Q


def obj_segments_3(u, w2, params, print_vals=False, full_output=False):
    z, c, T, Tw, x1, x2, x3 = sim_segments(u, params)
    # print(f"x1: {x1}" + "\n")
    K2 = 1
    J = (1 - w2) * (1 - x1) + w2 * K2 * x3

    # print(f"J: {J}" + "\n")
    Q = J[-1]
    # print(f"Q: {Q}" + "\n")
    if print_vals == True:
        pass
        # print(u, Q)
    if full_output:
        return Q, x1, x3
    else:
        return Q


def const_segments(u, params):
    z, c, T, Tw, x1, x2, x3 = sim_segments(u, params)
    con = np.hstack(([params["T_ub"] - T, T - params["T_lb"]]))
    return con


def sim(u, params):
    u = u[0]
    y0 = np.zeros(2)
    zspan = [0, params["L"]]
    sol = solve_ivp(lambda t, y: tubular_reactor_model(t, y, u, params), zspan, y0)
    z = sol.t
    x1 = sol.y[0, :]
    x2 = sol.y[1, :]
    T = params["Tin"] * (1 + x2)
    c = params["cin"] * (1 - x1)
    Tw = params["Tin"] * (1 + u)
    return z, c, T, Tw, x1, x2


def const(u, params):
    z, c, T, Tw, x1, x2 = sim(u, params)
    con = [
        params["T_ub"]
        - np.max(T),  # No temperature in the reactor may␣,→exceed the maximum
        np.min(T) - params["T_lb"],
    ]  # No temperature in the reactor may go␣below the minimum
    return con

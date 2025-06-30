# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# List of several dimensional analysis examples 

import numpy as np

def remove_dependent(example):
    name, (var_names, A, dims, dep) = example
    if dep not in var_names:
        raise ValueError(f"dependent var {dep!r} not in {var_names}")
    idx = var_names.index(dep)
    # drop the dep var from the list
    new_vars = var_names[:idx] + var_names[idx+1:]
    # drop the corresponding column from A
    new_A    = np.delete(A, idx, axis=1)
    return name, (new_vars, new_A, dims, dep)


def drop_zero_columns(example):
    """
    Remove any variable whose entire column in A is zero.
    Returns (new_var_names, new_A, dims, target).
    """
    name, (var_names, A, dims, dep) = example
    keep = ~np.all(A == 0, axis=0)
    new_vars = [v for v, k in zip(var_names, keep) if k]
    new_A = A[:, keep]
    return name, (new_vars, new_A, dims, dep)

# --- Problem definitions ---

examples = {
    "Pressure Drop in Pipe": (
        ['Δp','R','d','μ','Q'],
        np.array([
            [ 1,  0,  0,  1,  0],   # M
            [-1,  1,  1, -1,  3],   # L
            [-2,  0,  0, -1, -1],   # T
        ]),
        ['M','L','T'],
        'Δp'
    ),
    "Speed of Virus Infection": (
        ['V_p','P_r','θ','C_a','C_e','E_fs','H'],
        np.array([
            [ 0,  0,  0,  0,  0,  0, 1],  # M
            [ 1,  1,  0,  3,  0, -2,-3],  # L
            [-1,  0,  0, -1,  1,  0, 0],  # T
            [ 0,  0,  1,  0,  0,  0, 0],  # Θ
        ]),
        ['M','L','T','Θ'],
        'V_p'
    ),
    "Economic Growth": (
        ['P','L','ω_L','Y','r','δ'],
        np.array([
            [ 1,  0,  1,  1,  0,  0],  # K (capital)
            [ 0,  1, -1,  0,  0,  0],  # Q (labour)
            [ 0, -1,  0, -1, -1, -1],  # T (time)
        ]),
        ['K','Q','T'],
        'Y'
    ),
    "Pressure Inside a Bubble (M,L,T)": (
        ['Δp','R','σ'],
        np.array([
            [ 1,  0,  1],   # M
            [-1,  1,  0],   # L
            [-2,  0, -2],   # T
        ]),
        ['M','L','T'],
        'Δp'
    ),
    "Pressure Inside a Bubble (F,L)": (
        ['Δp','R','σ'],
        np.array([
            [ 1,  0,  1],   # F (force)
            [-2,  1, -1],   # L (length)
        ]),
        ['F','L'],
        'Δp'
    ),
    "Hydrogen Knudsen Compressor": (
        ['u','H','DeltaT','L','T0','lambda'],
        np.array([
            [ 0,  1,  0,  1,  0,  1],  # M
            [-1,  0,  1,  0,  1,  0],  # L
            [ 0,  0,  0,  0,  0,  0],  # Θ (temperature enters via exponents on deltaT and T0)
        ]),
        ['M','L','Θ'],
        'u'
    ),
    "Centrifugal Pump": (
        ['ΔP','R','V','Q','E','G'],
        np.array([
            [ 1,  1,  1,  0,  0,  0],  # M
            [-3,  2, -1,  3,  1,  0],  # L
            [ 0, -3, -1, -1,  0, -1],  # T
        ]),
        ['M','L','T'],
        'ΔP'
    ),
    "System I (Umströmung)": (
        ['ΔF','rho','v','D','eta'],
        np.array([
            [ 1,  1,  0,  0,  1],  # M (force F has dimension M·L·T⁻², but since F is dependent we only care about the others)
            [ 1, -3,  1,  1, -1],  # L
            [-2,  0, -1,  0, -1],  # T
        ]),
        ['M','L','T'],
        'ΔF'
    ),
    "System II (Auftrieb)": (
        ['ΔF','rho_F','drho','D','eta','g'],
        np.array([
            [ 1,  1,  1,  0,  1,  0],  # M
            [ 1, -3, -3,  1, -1,  1],  # L
            [-2,  0,  0,  0, -1, -2],  # T
        ]),
        ['M','L','T'],
        'ΔF'
    ),
    "System IIIa (Rauhigkeit)": (
        ['Delta_p','rho','v','D','L','k_s'],
        np.array([
            [ 1,  1,  0,  0,  0,  0],  # M
            [-1, -3,  1,  1,  1,  1],  # L
            [-2,  0, -1,  0,  0,  0],  # T
        ]),
        ['M','L','T'],
        'Delta_p'
    ),
    "System IIIb (Einbauten)": (
        ['Delta_p','rho','eta','v','D'],
        np.array([
            [ 1,  1,  1,  0,  0],  # M
            [-1, -3, -1,  1,  1],  # L
            [-2,  0, -1,  0,  0],  # T
        ]),
        ['M','L','T'],
        'Delta_p'
    ),
    
    "Forced Convection over a Cylinder": (
        ['Nu', 'Re', 'Pr', 'L/D', 'k', 'rho', 'c_p', 'mu'],
        np.array([
            #   Nu Re Pr L/D   k     rho  c_p   mu
            [   0,  0,  0,   0,   1,    1,   0,    1],   # M
            [   0,  0,  0,   0,   1,   -3,   2,   -1],   # L
            [   0,  0,  0,   0,  -3,    0,  -2,   -1],   # T
            [   0,  0,  0,   0,  -1,    0,  -1,    0],   # Θ
        ]),
        ['M','L','T','Θ'],
        'Nu'
    ),
    "Natural Convection from a Horizontal Plate": (
        ['Nu', 'Gr', 'Pr', 'L', 'beta', 'DeltaT', 'rho', 'mu', 'k', 'g'],
        np.array([
            #   Nu Gr Pr   L  beta ΔT  rho  mu   k   g
            [   0,  0,  0,   0,   0,   0,   1,    1,   1,   0],  # M
            [   0,  0,  0,   1,   0,   0,  -3,   -1,   1,   1],  # L
            [   0,  0,  0,   0,   0,   0,   0,   -1,  -3,  -2],  # T
            [   0,  0,  0,   0,  -1,   1,   0,    0,  -1,   0],  # Θ
        ]),
        ['M','L','T','Θ'],
        'Nu'
    ),
    "Packed‐Bed Pressure Drop (Ergun)": (
        ['DeltaP','rho','mu','U','D_p','epsilon','L'],
        np.array([
            #   ΔP rho mu  U  Dp eps L
            [    1,   1,  1,  0,  0,  0, 0],  # M
            [   -1,  -3, -1,  1,  1,  0, 1],  # L
            [   -2,   0, -1, -1,  0,  0, 0],  # T
        ]),
        ['M','L','T'],
        'DeltaP'
    ),
    "Stirred‐Tank Mixing (Power Number)": (
        ['P','rho','N','D','mu','sigma'],
        np.array([
            #   P rho N D mu sigma
            [   1,   1, 0, 0,  1,     1],  # M
            [   2,  -3, 0, 1, -1,     0],  # L
            [  -3,   0, -1,0, -1,    -2],  # T
            [   0,   0, 0, 0,  0,     0],  # Θ
        ]),
        ['M','L','T','Θ'],
        'P'
    ),
    "Rayleigh–Bénard Convection": (
        ['Nu','Ra','Pr','H','k','rho','c_p','mu','g','beta','DeltaT'],
        np.array([
            #   Nu Ra Pr H  k    rho  c_p   mu   g  beta ΔT
            [   0,  0, 0, 0,   1,    1,    0,    1,   0,   0,  0],  # M
            [   0,  0, 0, 1,   1,   -3,    2,   -1,   1,   0,  0],  # L
            [   0,  0, 0, 0,  -3,    0,   -2,   -1,  -2,   0,  0],  # T
            [   0,  0, 0, 0,  -1,    0,   -1,    0,   0,  -1,  1],  # Θ
        ]),
        ['M','L','T','Θ'],
        'Nu'
    ),

    "Drop Dynamics in Shear (Ca, Re)": (
        ['d','mu_c','mu_d','sigma','G','rho_c','rho_d'],
        np.array([
            #  d mu_c mu_d sigma G rho_c rho_d
            [  0,   1,   1,     1,   0,     1,     1],  # M
            [  1,  -1,  -1,     0,   0,    -3,    -3],  # L
            [  0,  -1,  -1,    -2,  -1,     0,     0],  # T
        ]),
        ['M','L','T'],
        'd'
    ),
    "MHD Duct Flow": (
        ['Ha','Re','sigma_e','B','L','rho','mu'],
        np.array([
            #  Ha Re σ_e  B   L rho mu
            [   0,  0, -1,  1,  0,  1, 1],  # M
            [   0,  0, -3,  0,  1, -3,-1],  # L
            [   0,  0,  3, -2,  0,  0,-1],  # T
            [   0,  0,  2, -1,  0,  0, 0],  # I
        ]),
        ['M','L','T','I'],
        'Ha'
    ),
    "Aeroacoustic Dipole Radiation": (
        ['p_prime','rho','U','L','c','omega','l'],
        np.array([
            #   p' rho  U  L   c omega l
            [    1,  1, 0,  0,  0,   0, 0],  # M
            [   -1, -3, 1,  1,  1,   0, 1],  # L
            [   -2,  0, -1, 0, -1,  -1, 0],  # T
        ]),
        ['M','L','T'],
        'p_prime'
    ),
    "Laminar Forced‐Convection over a Cylinder": (
        # variables: heat‐transfer coefficient h, diameter D, conductivity k,
        # velocity U, viscosity mu, density rho, specific heat c_p
        ['h','D','k','U','mu','rho','c_p'],
        np.array([
            #    h   D   k   U   mu  rho  c_p
            [   1,  0,  1,  0,   1,   1,   0],   # M
            [  -2,  1,  1,  1,  -1,  -3,   2],   # L
            [  -3,  0, -3, -1,  -1,   0,  -2],   # T
            [  -1,  0, -1,  0,   0,   0,  -1],   # Θ
        ]),
        ['M','L','T','Θ'],
        'h'
    ),
}


#!/usr/bin/env python3

# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Analyses several dimensional analysis examples using continuous evolutionary optimization
# pi groups are ranked using R2 and a mean CV penalty to ensure sufficient spread.
# Uses a generated sample dataframe, Generation trys to avoid a bias to a specific set of pi groups.
# Replace with real experimental data as soon as you have these

# To install fcmaes execute:
# pip install fcmaes --upgrade

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Toggle to integer exponents if you really want
use_int_exponents = False
# Remove the dependent variable from the input matrix 
# Sometimes needs to be set to False (you have to include the dependent) if you want to find any pi groups
use_remove_dependent = True

# --- Your evolutionary optimizer (minimizer) ---
def evolutionary_optimizer(fitness, dim, lb, ub, **kwargs):
    from scipy.optimize import Bounds
    from fcmaes.optimizer import wrapper, Bite_cpp
    from fcmaes import retry
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}", level="INFO")
    max_evals = 2000
    res = retry.minimize(
        # wrapper(fitness), # for debugging to monitor the optimization progress
        fitness,
        bounds=Bounds(lb, ub),
        num_retries=16,
        optimizer=Bite_cpp(max_evals),
        workers=16
    )
    return res.x

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
}

# --- Helpers ---------------------------------------------------------

def nullspace_basis(A, tol=1e-12):
    U, s, Vt = np.linalg.svd(A)
    rank = (s > tol).sum()
    return Vt.T[:, rank:]   # shape (N, N-rank)

def sample_dataframe(var_names, n=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    logs = rng.uniform(-3, 3, size=(n, len(var_names)))
    return pd.DataFrame(10**logs, columns=var_names)

def make_unbiased_surrogate(df_x, Ns, noise_scale=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    # build all k π-features from Ns
    logX = np.log(df_x.values)
    Theta = np.exp(logX @ Ns)           # shape (n, k)
    # random weights
    w = rng.standard_normal(Theta.shape[1])
    noise = noise_scale * rng.standard_normal(Theta.shape[0])
    return Theta @ w + noise

# read experimental data
def load_data(path, var_names, dep_var):
    return pd.read_excel(path)

def make_objective(df_x, y, Ns, m, cv_floor=0.1, cv_cap=10.0, cv_pen=100.0):
    """
    Build an objective function evaluating the data
    """
    X = df_x.values
    logX = np.log(X)
    n, N = X.shape
    k = Ns.shape[1]

    def objective(flat_c):
        # reshape to C
        C = flat_c.reshape((k, m))
        E = Ns @ C
        if use_int_exponents:
            E = E.astype('int')
        # build pi-matrix
        Pi = np.exp(logX @ E)  # shape (n, m)
        # fit with intercept
        model = LinearRegression(fit_intercept=True).fit(Pi, y)
        r2 = model.score(Pi, y)
        # mean CV / spread
        cvs = Pi.std(axis=0) / np.maximum(np.abs(Pi.mean(axis=0)), 1e-12)      
        # penalize too‐small CV
        low_viols = np.maximum(0.0, cv_floor - cvs)
        # penalize too‐large CV
        high_viols = np.maximum(0.0, cvs - cv_cap)
        cv_penalty = cv_pen*(low_viols.sum() + high_viols.sum())
        return (1.0 - r2) + cv_penalty

    return objective

def compute_r2_and_E(flat_c, df_x, y, Ns, m):
    logX = np.log(df_x.values)
    n, N  = df_x.shape
    k     = Ns.shape[1]
    C = flat_c.reshape((k, m))
    E = Ns @ C
    if use_int_exponents:
        E = E.astype(int)
    Pi = np.exp(logX @ E)
    r2 = LinearRegression(fit_intercept=True).fit(Pi, y).score(Pi, y)
    return r2, E


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


def optimize_example(rng, name, var_names, A, dims, dep_var, df=None):
    print(f"\n=== {name} ===")
    print("    Variables:", var_names)
    Ns = nullspace_basis(A) 
    if not df is None: # we have experimental data
        df_x = df[var_names]
        y = df[dep_var]
    else:
        # sample X
        df_x = sample_dataframe(var_names, n=300, rng=rng)
        # build unbiased surrogate y
        y  = make_unbiased_surrogate(df_x, Ns, noise_scale=0.1, rng=rng)
    # optimize for m=1..min(3,k)
    k = Ns.shape[1]
    for m in range(1, k+1):
        print(f"\n  → m = {m} π-group{'s' if m>1 else ''}")
        obj = make_objective(df_x, y, Ns, m)
        lb = [-9.0] * (k * m)
        ub = [ 9.0] * (k * m)
        best = evolutionary_optimizer(obj, dim=k*m, lb=lb, ub=ub)            
        best_r2, E_opt = compute_r2_and_E(best, df_x, y, Ns, m)
        # --- compute mean CV and complexity ---
        logX = np.log(df_x.values)
        Pi = np.exp(logX @ E_opt)   # shape (n, m)
        cvs = Pi.std(axis=0) / np.maximum(np.abs(Pi.mean(axis=0)), 1e-12)
        mean_cv = cvs.mean()
        complexity = np.abs(E_opt).sum()
        
        print(f"    R² = {best_r2:.6f}  mean_CV = {mean_cv:.6f}  complexity = {complexity:.1f}")
        for j in range(m):
            exps = E_opt[:, j]
            print(f"    π_{j+1} exponents: {np.array2string(exps,precision=4)}")

def optimize_Knudsen(rng):
    name = "Hydrogen Knudsen Compressor"
    example = name, (examples[name])
    example = remove_dependent(example) 
    name, (var_names, A, dims, dep_var) = example
    df = None
    #df = load_data("knudsen_data.xlsx", var_names, dep_var)
    optimize_example(rng, name, var_names, A, dims, dep_var, df) 
                
def main():
    rng = np.random.default_rng(0)
    # optimize_Knudsen(rng)
    # sys.exit()
    
    for example in examples.items():
        if use_remove_dependent:
            example = remove_dependent(example)
        name, (var_names, A, dims, dep_var) = example
        optimize_example(rng, name, var_names, A, dims, dep_var)

if __name__ == "__main__":
    main()

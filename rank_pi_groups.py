#!/usr/bin/env python3

# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Analyses several dimensional analysis examples similar to https://github.com/saadgroup/BuckinghamPy
# pi groups are ranked using the weighted sum of complexity, CV and R2.
# Adapt the weights to your requirements. 
# Uses a generated sample dataframe, Generation trys to avoid a bias to a specific set of pi groups.
# Replace with real experimental data if you have these

import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 1) Helpers ---

def find_repeating_sets(A, var_names, dep_var=None):
    """
    Return all subsets of size r = rank(A) whose columns are full-rank,
    but if dep_var is given, exclude any set that contains dep_var.
    """
    r = np.linalg.matrix_rank(A)
    N = A.shape[1]
    valid = []
    for combo in itertools.combinations(range(N), r):
        reps = [var_names[i] for i in combo]
        if dep_var and dep_var in reps:
            continue
        if np.linalg.matrix_rank(A[:, combo]) == r:
            valid.append(reps)
    return valid

def compute_pi_groups(A, var_names, reps):
    """
    Solve for each non-repeater j: A[:,reps] · e = -A[:, j], returning
    dict π_name → {var: exponent, ...}.
    """
    rep_idx = [var_names.index(v) for v in reps]
    A_r = A[:, rep_idx]
    pi = {}
    for j, name in enumerate(var_names):
        if name in reps:
            continue
        e, *_ = np.linalg.lstsq(A_r, -A[:, j], rcond=None)
        # build exponent dict
        exps = {name: 1.0}
        for k, idx in enumerate(rep_idx):
            exps[var_names[idx]] = float(e[k])
        pi[f"π_{name}"] = exps
    return pi

def sample_dataframe(var_names, n=3000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    logs = rng.uniform(-3, 3, size=(n, len(var_names)))
    return pd.DataFrame(10**logs, columns=var_names)

# --- 2) Define examples, now with known dependent var as 4th element ---

examples = {
    "Pressure Drop in Pipe": (
        ['Δp','R','d','μ','Q'],
        np.array([[ 1,  0,  0,  1,  0],
                  [-1,  1,  1, -1,  3],
                  [-2,  0,  0, -1, -1]]),
        'Δp'
    ),
    "Speed of Virus Infection": (
        ['V_p','P_r','θ','C_a','C_e','E_fs','H'],
        np.array([[ 0, 0, 0,  0,  0,  0, 1],
                  [ 1, 1, 0,  3,  0, -2,-3],
                  [-1, 0, 0, -1,  1,  0, 0],
                  [ 0, 0, 1,  0,  0,  0, 0]]),
        'V_p'
    ),
    "Economic Growth": (
        ['P','L','ω_L','Y','r','δ'],
        np.array([[ 1,  0,  1,  1,  0,  0],
                  [ 0,  1, -1,  0,  0,  0],
                  [ 0, -1,  0, -1, -1, -1]]),
        'Y'
    ),
    "Pressure Inside a Bubble (M,L,T)": (
        ['Δp','R','σ'],
        np.array([[ 1,  0,  1],
                  [-1,  1,  0],
                  [-2,  0, -2]]),
        'Δp'
    ),
    "Pressure Inside a Bubble (F,L)": (
        ['Δp','R','σ'],
        np.array([[ 1,  0,  1],
                  [-2,  1, -1]]),
        'Δp'
    ),
    "Hydrogen Knudsen Compressor": (
        ['u','H','DeltaT','L','T0','lambda'],
        np.array([[0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 0, 1, 0]]),
        'u'
    ),
    "Centrifugal Pump": (
        ['ΔP','R','V','Q','E','G'],
        np.array([[ 1,  1,  1,  0,  0,  0],
                  [-3,  2, -1,  3,  1,  0],
                  [ 0, -3, -1, -1,  0, -1]]),
        'ΔP'
    ),
    "System I (Umströmung)": (
        ['F_W','rho','v','D','eta'],
        np.array([[ 1,  1,  0, 0,  1],
                  [ 1, -3,  1, 1, -1],
                  [-2,  0, -1, 0, -1]]),
        'F_W'
    ),
    "System II (Auftrieb)": (
        ['F_A','rho_F','drho','D','eta','g'],
        np.array([[ 1,  1,  1, 0,  1, 0],
                  [ 1, -3, -3, 1, -1, 1],
                  [-2,  0,  0, 0, -1,-2]]),
        'F_A'
    ),
    "System IIIa (Rauhigkeit)": (
        ['Delta_p','rho','v','D','L','k_s'],
        np.array([[ 1,  1,  0, 0, 0, 0],
                  [-1, -3,  1, 1, 1, 1],
                  [-2,  0, -1, 0, 0, 0]]),
        'Delta_p'
    ),
    "System IIIb (Einbauten)": (
        ['Delta_p','rho','v','D'],
        np.array([[ 1,  1,  0, 0],
                  [-1, -3,  1, 1],
                  [-2,  0, -1, 0]]),
        'Delta_p'
    ),
}

rng = np.random.default_rng(123)

# --- 3) Ranking pipeline ---

for name, (var_names, A, dep_var) in examples.items():
    print(f"\n=== Example: {name} ===")

    # 3a) enumerate only those repeater‐sets that exclude the dependent var
    all_reps = find_repeating_sets(A, var_names, dep_var=dep_var)
    if not all_reps:
        print("  ↳ No valid repeating sets (nullity = 0).")
        continue

    # 3b) sample original X
    df_x = sample_dataframe(var_names, n=3000, rng=rng)

    # 3c) build a global surrogate P_global covering all π‐blocks
    pi_blocks = []
    for i, reps in enumerate(all_reps):
        pi_groups = compute_pi_groups(A, var_names, reps)
        if not pi_groups:
            continue
        df_pi = pd.DataFrame({
            f"S{i}_{pi_name}": np.prod([df_x[var]**exp for var,exp in exps.items()], axis=0)
            for pi_name, exps in pi_groups.items()
        })
        pi_blocks.append(df_pi)
    P_global = pd.concat(pi_blocks, axis=1)

    # 3d) unbiased surrogate y = P_global·w + noise
    w = rng.standard_normal(P_global.shape[1])
    noise = 0.1*rng.standard_normal(P_global.shape[0])
    y = P_global.values @ w + noise
    df_x['y'] = y

    # 3e) rank each candidate set by its own πs predicting y
    records = []
    for reps in all_reps:
        pi_groups = compute_pi_groups(A, var_names, reps)
        if not pi_groups:
            continue
        df_pi = pd.DataFrame({
            pi_name: np.prod([df_x[var]**exp for var,exp in exps.items()], axis=0)
            for pi_name, exps in pi_groups.items()
        })
        complexity = sum(abs(e) for exps in pi_groups.values() for e in exps.values())
        mean_cv    = (df_pi.std()/df_pi.mean()).mean()
        r2         = LinearRegression().fit(df_pi.values, y).score(df_pi.values, y)
        records.append({
            'repeating_vars': reps,
            'π_names':        list(pi_groups.keys()),
            'complexity':     complexity,
            'mean_CV':        mean_cv,
            'R2':             r2
        })

    df_m = pd.DataFrame(records)
    # normalize metrics
    for col in ('complexity','mean_CV','R2'):
        mn, mx = df_m[col].min(), df_m[col].max()
        df_m[f'norm_{col}'] = (df_m[col]-mn)/(mx-mn) if mx>mn else 0.0
    df_m['norm_complexity'] = 1-df_m['norm_complexity']
    w1, w2, w3 = 0.33, 0.33, 0.34
    df_m['score'] = w1*df_m['norm_complexity'] + w2*df_m['norm_mean_CV'] + w3*df_m['norm_R2']

    # 3f) print ranking sorted (lowest score first or as you prefer)
    print(df_m.sort_values('score', ascending=True)[[
        'repeating_vars','π_names','complexity','mean_CV','R2','score'
    ]].to_string(index=False))

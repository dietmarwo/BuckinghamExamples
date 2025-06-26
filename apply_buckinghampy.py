#!/usr/bin/env python3

# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Applies https://github.com/saadgroup/BuckinghamPy to several dimensional analysis examples 

# To install buckinghampy:

# Clone the git repo:
#   git clone https://github.com/saadgroup/BuckinghamPy.git . 
# Use pip tool to install the package in the active python evironment:
#   pip install .

from buckinghampy import BuckinghamPi
import numpy as np

# --- Helper to format dimension exponents into strings ---

def exponents_to_dimension_string(exps, dims):
    num, den = [], []
    for d, e in zip(dims, exps):
        if e > 0:
            num.append(f"{d}" + (f"^{int(e)}" if e!=1 else ""))
        elif e < 0:
            den.append(f"{d}" + (f"^{int(-e)}" if e!=-1 else ""))
    if not num:
        num = ["1"]
    s = "*".join(num)
    if den:
        s += "/" + "*".join(den)
    return s

# --- All examples: (var names, A-matrix, fund-dims, dependent var) ---

examples = {
    "Pressure Drop in Pipe": (
        ['Δp','R','d','μ','Q'],
        [[ 1, 0, 0, 1, 0],
         [-1, 1, 1,-1, 3],
         [-2, 0, 0,-1,-1]],
        ['M','L','T'],
        'Δp'
    ),
    "Speed of Virus Infection": (
        ['V_p','P_r','θ','C_a','C_e','E_fs','H'],
        [[ 0, 0, 0,  0,  0,  0, 1],
         [ 1, 1, 0,  3,  0, -2,-3],
         [-1, 0, 0, -1,  1,  0, 0],
         [ 0, 0, 1,  0,  0,  0, 0]],
        ['M','L','T','Θ'],
        'V_p'
    ),
    "Economic Growth": (
        ['P','L','ω_L','Y','r','δ'],
        [[ 1,  0,  1,  1,  0,  0],
         [ 0,  1, -1,  0,  0,  0],
         [ 0, -1,  0, -1, -1, -1]],
        ['K','Q','T'],
        'Y'
    ),
    "Pressure Inside a Bubble (M,L,T)": (
        ['Δp','R','σ'],
        [[ 1,  0,  1],
         [-1,  1,  0],
         [-2,  0, -2]],
        ['M','L'],       # ← only the independent dims (rank=2)
        'Δp'
    ),
    "Pressure Inside a Bubble (F,L)": (
        ['Δp','R','σ'],
        [[ 1,  0,  1],
         [-2,  1, -1]],
        ['F','L'],
        'Δp'
    ),
    "Hydrogen Knudsen Compressor": (
        ['u','H','DeltaT','L','T0','lambda'],
        # here 'u' is the dependent (e.g. flow velocity or pressure rise)
        [[0, 1, 0, 1, 0, 1],
         [0, 0, 1, 0, 1, 0]],
        ['M','L'],
        'u'
    ),
    "Centrifugal Pump": (
        ['ΔP','R','V','Q','E','G'],
        [[ 1,  1,  1,  0,  0,  0],
         [-3,  2, -1,  3,  1,  0],
         [ 0, -3, -1, -1,  0, -1]],
        ['M','L','T'],
        'ΔP'
    ),
    "System I (Umströmung)": (
        ['F_W','rho','v','D','eta'],
        [[ 1,  1,  0, 0,  1],
         [ 1, -3,  1, 1, -1],
         [-2,  0, -1, 0, -1]],
        ['M','L','T'],
        'F_W'
    ),
    "System II (Auftrieb)": (
        ['F_A','rho_F','drho','D','eta','g'],
        [[ 1,  1,  1, 0,  1, 0],
         [ 1, -3, -3, 1, -1, 1],
         [-2,  0,  0, 0, -1,-2]],
        ['M','L','T'],
        'F_A'
    ),
    "System IIIa (Rauhigkeit)": (
        ['Delta_p','rho','v','D','L','k_s'],
        [[ 1,  1,  0, 0, 0, 0],
         [-1, -3,  1, 1, 1, 1],
         [-2,  0, -1, 0, 0, 0]],
        ['M','L','T'],
        'Delta_p'
    ),
    "System IIIb (Einbauten)": (
        ['Delta_p','rho','v','D'],
        [[ 1,  1,  0, 0],
         [-1, -3,  1, 1],
         [-2,  0, -1, 0]],
        ['M','L','T'],
        'Delta_p'
    ),
}

if __name__ == "__main__":
    for name, (vars_, A, dims, dep) in examples.items():
        print(f"\n=== {name} ===")
        N = len(vars_)
        D = len(dims)
        if N <= D:
            print(f"  ↳ Skipped: only {N} vars vs. {D} dims.")
            continue

        bp = BuckinghamPi()
        # add each variable, marking the dependent as non_repeating
        for var, col in zip(vars_, zip(*A)):
            dim_str = exponents_to_dimension_string(col, dims)
            bp.add_variable(
                name=var,
                dimensions=dim_str,
                non_repeating=(var == dep)
            )

        # generate & print all π-term sets
        try:
            bp.generate_pi_terms()
        except Exception as e:
            print(f"  ↳ Error: {e}")
            continue

        # output plain-text (not IPython Math/Markdown)
        if not bp.pi_terms:
            print("  ↳ No π-terms produced.")
        else:
            for pi in bp.pi_terms:
                print("  ", pi)

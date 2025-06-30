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
from examples import examples, drop_zero_columns

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

def apply_buckingham_py(name, var_names, A, dims, dep_var):
    print(f"\n=== {name} ===")
    N = len(var_names)
    D = len(dims)
    if N <= D:
        print(f"  ↳ Skipped: only {N} vars vs. {D} dims.")
        return

    bp = BuckinghamPi()
    # add each variable, marking the dependent as non_repeating
    for var, col in zip(var_names, zip(*A)):
        dim_str = exponents_to_dimension_string(col, dims)
        bp.add_variable(
            name=var,
            dimensions=dim_str,
            non_repeating=(var == dep_var)
        )

    # generate & print all π-term sets
    try:
        bp.generate_pi_terms()
    except Exception as e:
        print(f"  ↳ Error: {e}")
        return

    # output plain-text (not IPython Math/Markdown)
    if not bp.pi_terms:
        print("  ↳ No π-terms produced.")
    else:
        for pi in bp.pi_terms:
            print("  ", pi)
                

def main():    
    for example in examples.items():
        example = name, (var_names, A, dims, dep_var) = drop_zero_columns(example)
        name, (var_names, A, dims, dep_var) = example
        # if not name.startswith('Laminar Forced‐Convection'): continue # execute single
        apply_buckingham_py(name, var_names, A, dims, dep_var)
        
if __name__ == "__main__":
    main()


# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

'''
    Uses approximated data / results of experiments to perform continuous optimization for 

    "Hydrogen Knudsen Compressor": (
        ['u','H','DeltaT','L','T0','lambda'],
        np.array([
            [ 0,  1,  0,  1,  0,  1],  # M
            [-1,  0,  1,  0,  1,  0],  # L
            [ 0,  0,  0,  0,  0,  0],  # Θ (temperature enters via exponents on deltaT and T0)
        ]),
        ['M','L','Θ'],
        'u'
    )
    - data taken from from https://github.com/xqb-python/Dimensional-Analysis/tree/main/%E4%B8%AD%E5%BF%83%E5%9E%82%E7%9B%B4%E7%BA%BF%E4%B8%8A%E7%9A%84%E9%80%9F%E5%BA%A6%E5%88%86%E5%B8%83
    
    and

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
    )
    - data generated using cylinder_gen_params_1.py and cylinder_approximation.py 
'''

import numpy as np
import pandas as pd
from examples import examples, drop_zero_columns, remove_dependent
from optimize_pi_groups import optimize_example
# use model data

def optimize_Knudsen(rng):
    name = "Hydrogen Knudsen Compressor"
    example = name, (examples[name])
    example = remove_dependent(example) 
    name, (var_names, A, dims, dep_var) = example
    df = pd.read_excel("knudsen_data.xlsx")
    optimize_example(rng, name, var_names, A, dims, dep_var, df) 
    
def optimize_cylinder(rng):
    name = "Laminar Forced‐Convection over a Cylinder"
    example = name, (examples[name])
    example = drop_zero_columns(example)
    name, (var_names, A, dims, dep_var) = example
    df = pd.read_csv("n1_pi_cylinder_design_h.csv")
    optimize_example(rng, name, var_names, A, dims, dep_var, df) 
        
def main():
    rng = np.random.default_rng(42)    
    optimize_cylinder(rng)
    optimize_Knudsen(rng)
    
if __name__ == "__main__":
    main()
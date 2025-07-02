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
    df = pd.read_csv("n1_pi_cylinder_design_eff.csv")
    optimize_example(rng, name, var_names, A, dims, dep_var, df) 
        
def main():
    rng = np.random.default_rng(42)    
    optimize_cylinder(rng)
    #optimize_Knudsen(rng)
    
if __name__ == "__main__":
    main()
# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
#
# To install fcmaes execute:
# pip install fcmaes --upgrade
#
# Analyses several dimensional analysis examples using continuous evolutionary optimization
# pi groups are ranked using R2 and a mean CV penalty to ensure sufficient spread.
# Uses a generated sample dataframe, Generation trys to avoid a bias to a specific set of pi groups.
# Replace with real experimental data as soon as you have these
#
# Compare with https://github.com/xqb-python/Dimensional-Analysis/blob/main/%E4%B8%AD%E5%BF%83%E5%9E%82%E7%9B%B4%E7%BA%BF%E4%B8%8A%E7%9A%84%E9%80%9F%E5%BA%A6%E5%88%86%E5%B8%83/%E6%9C%80%E5%A4%A7%E6%BB%91%E7%A7%BB%E9%80%9F%E5%BA%A6.py
# using GA instead of an evolutionary approach. 
# Even more similar is https://www.ijche.com/article_10200_e5d7175834c141c6c71c4fe626ec5cb4.pdf
# which applies CMA-ES. But it is focused on a specific problem, where our approach is more general, applied
# here to about 20 distinct problems
#
# Usually π-group determination requires as a first step to determine which variables should go into which π-group₂ 
# (or whether to include a third group at all) which is a discrete, combinatorial choice.
#
# The continuous‐exponent pipeline implemented below never explicitly “chooses” which 
# variables go into which π-group the way the combinatorial subset‐enumeration code does. 
#
# Instead, it relies on the fact that:
#
#   1) Dimensional homogeneity ⇔ Nullspace:
#   Any exponent vector E ∈ ℝⁿ that makes
#       ∏_{i=1}^N x_i^{E_i}
#   dimensionless must satisfy A E = 0.
#
#   Computing Ns = nullspace(A) gives you an orthonormal basis of that nullspace (shape N×k),
#   so every valid E can be written as
#       E = Ns · c
#   for some coefficient vector c ∈ ℝᵏ.
#
#   2) Stacking into m π-groups:
#   If you want m π-groups, you choose a k×m matrix C whose columns are the c vectors 
#   for each π-group, compute
#       E = Ns · C   (an N×m matrix),
#   and then your π-features are
#       Π_{:,j} = exp(log(X) · E_{:,j})
#   These Πs are guaranteed to be dimensionless by construction.
#
#   3) Continuous optimization over C:
#   You then frame a single continuous optimization problem over the entries of C ∈ ℝ^{k×m},
#   maximizing R² (minus your CV penalty). The evolutionary optimizer wanders around ℝ^{k·m},
#   implicitly exploring all ways of mixing the k basis-vectors into m groups. There is no need
#   to discretely enumerate “which subset of variables” – the continuous weights in C merely
#   dial in each variable’s exponent.
#
#   Because E = Ns·C enforces A E = 0, every continuous trial C yields a valid set of π-groups.
#   The combinatorial question “which variables in π₁ vs π₂” is therefore solved implicitly
#   by the optimizer finding the best continuous weights, not by manual subset enumeration.
#
# Note that the optimization library used https://github.com/dietmarwo/fast-cma-es supports CMA-ES 
# and several other established algorithms. 
#
# Replacing the one used is a one-liner, but BiteOpt was chosen because of its superior flexibility:  
# - Tracks success statistics (e.g. which mutation scales actually produce improvements)
# - Re-weights its proposal distributions “on the fly” based on those stats
# - Balances exploration vs. exploitation automatically as the landscape changes

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from examples import examples, drop_zero_columns, remove_dependent

# Toggle to integer exponents if you really want
use_int_exponents = False
# Remove the dependent variable from the input matrix 
# Sometimes needs to be set to False (you have to include the dependent) if you want to find any pi groups
use_remove_dependent = True
use_drop_zero_columns = True
num_retries=32 # number of restarta
max_evals=2000 # number of evals per restart
workers=16 # parallelisation of the restarts

# --- Your evolutionary optimizer (minimizer) ---
def evolutionary_optimizer(fitness, dim, lb, ub, **kwargs):
    from scipy.optimize import Bounds
    from fcmaes.optimizer import wrapper, Bite_cpp
    from fcmaes import retry
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}", level="INFO")
    res = retry.minimize(
        # wrapper(fitness), # for debugging to monitor the optimization progress
        fitness,
        bounds=Bounds(lb, ub),
        num_retries=num_retries,
        optimizer=Bite_cpp(max_evals),
        workers=workers
    )
    return res.x

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
    df = load_data("knudsen_data.xlsx", var_names, dep_var)
    optimize_example(rng, name, var_names, A, dims, dep_var, df) 
                
def main():
    rng = np.random.default_rng(42)
    # optimize_Knudsen(rng)
    # sys.exit()
    
    for example in examples.items():
        if use_remove_dependent:
            example = remove_dependent(example)
        if use_drop_zero_columns:
            example = drop_zero_columns(example)
        name, (var_names, A, dims, dep_var) = example
        # if not name.startswith('Laminar Forced‐Convection'): continue # execute single
        optimize_example(rng, name, var_names, A, dims, dep_var)

    print(f'\nremove_dependent = {use_remove_dependent}, drop_zero_columns = {use_drop_zero_columns}')
    print(f'{num_retries} parallel restarts, {max_evals} evaluations each, {workers} workers')

if __name__ == "__main__":
    main()

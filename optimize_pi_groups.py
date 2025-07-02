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
# Typically, π-group determination necessitates an initial discrete, combinatorial decision: 
# selecting which variables belong to which π-group (or determining whether to incorporate 
# an additional group). The continuous-exponent pipeline described below circumvents explicit 
# variable assignment to π-groups, unlike subset-enumeration approaches. Instead, it 
# leverages the following principles:
#
# 1. Dimensional Homogeneity and Nullspace Equivalence
#
# Any exponent vector E ∈ ℝⁿ that renders ∏ᵢ₌₁ᴺ xᵢᴱⁱ dimensionless must satisfy A·E = 0.
#
# Computing Nₛ = nullspace(A) yields an orthonormal basis of that nullspace (with shape N×k), 
# enabling any valid E to be expressed as E = Nₛ · c for some coefficient vector c ∈ ℝᵏ.
#
# 2. Construction of m π-Groups
#
# To construct m π-groups, select a k×m matrix C whose columns represent the c vectors 
# for each π-group. Compute E = Nₛ · C (an N×m matrix) and construct the π-features: 
# Π₍₊,ⱼ₎ = exp(log(X) · E₍₊,ⱼ₎).
#
# These π-terms are guaranteed to be dimensionless by construction.
#
# 3. Continuous Optimization Framework
#
# The approach formulates a single continuous optimization problem over the entries of 
# C ∈ ℝᵏˣᵐ, maximizing R² (adjusted for cross-validation penalty). The evolutionary 
# optimizer explores ℝᵏ·ᵐ, implicitly investigating all possible combinations of the 
# k basis vectors into m groups.
#
# This eliminates the need for discrete enumeration of variable subsets—the continuous 
# weights in C determine each variable's exponent coefficient.
#
# Since E = Nₛ·C enforces A·E = 0, every continuous trial C produces a valid π-group 
# configuration. The combinatorial challenge of "which variables belong to π₁ versus π₂" 
# is resolved implicitly through the optimizer's identification of optimal continuous 
# weights, rather than through manual subset enumeration.
#
# NOTE: The optimization library employed (https://github.com/dietmarwo/fast-cma-es) 
# supports CMA-ES and various other established algorithms. Substituting the current 
# implementation requires only a single line change. BiteOpt was selected for its 
# superior flexibility:
#
# - Maintains success statistics (tracking which mutation scales generate actual improvements)
# - Dynamically re-weights proposal distributions based on performance statistics  
# - Automatically balances exploration versus exploitation as the optimization landscape evolves

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
                
def main():
    rng = np.random.default_rng(42)
    
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

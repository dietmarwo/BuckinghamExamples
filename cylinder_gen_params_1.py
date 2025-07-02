# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

'''
    Generate design points satisfying the single continuous π‑group for cylinder h,
    with final filtering by Re_D to target Re_D ∈ [30,300].
    
        π₁ = D^0.5171 · k^-0.1687 · U^0.5171 · μ^-0.3485 · ρ^0.5171 · c_p^0.1687
    
    D is swept 5 mm … 30 mm; π₁ spans ±2 decades. For each (D, π₁) solve for U, compute
    Re_D, and keep rows only if 30 ≤ Re_D ≤ 300. This ensures the full design lies
    within the desired Reynolds-number window. Output → n1_pi_cylinder_design.csv
'''

import numpy as np
import pandas as pd
from itertools import product

# 1) π‑group exponents for [D, k, U, mu, rho, c_p]
exp = np.array([ 0.5171, -0.1687,  0.5171, -0.3485,  0.5171,  0.1687])
vars_ = ['D','k','U','mu','rho','c_p']
idx_D, idx_U = vars_.index('D'), vars_.index('U')

# 2) fixed fluid properties (air at 25 °C)
piv = dict(
    k   = 0.0260,     # W/m/K
    mu  = 1.85e-5,    # Pa·s
    rho = 1.18,       # kg/m³
    c_p = 1005.0,     # J/kg/K
)

# 3) pre‑compute constant log-sum for k, mu, rho, c_p
ln_fixed = sum(exp[vars_.index(k)] * np.log(v) for k, v in piv.items())
e_D, e_U = exp[idx_D], exp[idx_U]

# 4) design grids
D_vals  = np.logspace(np.log10(0.005), np.log10(0.03), 10)  # 5–30 mm, 10 pts
pi_vals = np.logspace(-2.0, 2.0, 41)                        # ±2 decades, 41 pts

# 5) solve for U and compute Re_D
rows = []
for D, p1 in product(D_vals, pi_vals):
    ln_U = (np.log(p1) - ln_fixed - e_D*np.log(D)) / e_U
    U    = np.exp(ln_U)
    Re_D = piv['rho'] * U * D / piv['mu']
    if 30.0 <= Re_D <= 300.0:
        rows.append(dict(piv, D=D, U=U, Re_D=Re_D))

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No rows found – relax π span or D range.")

# 6) save and summary
pd.set_option('display.max_columns', None,
              'display.expand_frame_repr', False)
print(df[['D','U','Re_D']].describe())
df.to_csv("n1_pi_cylinder_design.csv", index=False)

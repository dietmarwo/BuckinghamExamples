# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

'''
    Approximates the heat‐transfer coefficient h for given parameters 
    D, k, U, mu. rho. c_p using a well known analytical method for

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
    You may replace this using either simulation or do real physical experiments.
'''

import numpy as np
import pandas as pd
import sys, pathlib

# Hilpert + Zukauskas (cross-flow cylinder)  | Re window |   C    |   m
_HILPERT_ZUK = [
    (0.4,        4.0,          0.989, 0.330),
    (4.0,       40.0,          0.911, 0.385),
    (40.0,     400.0,          0.683, 0.466),
    (400.0,   4000.0,          0.193, 0.618),
    (4000.0, 40000.0,          0.027, 0.805),
]

def _churchill_bernstein(Re, Pr):
    """Nusselt for external cross-flow cylinder (valid 0.2<Re<1e7)."""
    term1 = (0.4 / Pr)**(2/3)
    A = 0.62 * Re**0.5 * Pr**(1/3)
    B = (1 + term1)**0.25
    C = (1 + (Re / 282000)**(5/8))**(4/5)
    return 0.3 + A / B * C

def h_cylinder(D, k, U, mu, rho, c_p):
    """
    Heat-transfer coefficient h [W m⁻² K⁻¹] for air/water over a cylinder.
    Uses Hilpert/Zukauskas when 0.4<=Re<=4e4, otherwise Churchill–Bernstein.
    Arrays broadcast like NumPy.
    """
    Re = rho * U * D / mu
    Pr = mu * c_p / k

    Nu = np.empty_like(Re, dtype=float)

    # 1) try to map via Hilpert/Zukauskas piecewise power laws
    assigned = np.zeros_like(Re, dtype=bool)
    for Re_lo, Re_hi, C, m in _HILPERT_ZUK:
        mask = (Re >= Re_lo) & (Re < Re_hi)
        Nu[mask] = C * Re[mask]**m * Pr**(1/3)
        assigned |= mask

    # 2) everything else → Churchill–Bernstein
    if (~assigned).any():
        mask = ~assigned
        Nu[mask] = _churchill_bernstein(Re[mask], Pr)

    return Nu * k / D

def test():
    # air at 25 °C
    k   = 0.026   # W/m/K
    mu  = 1.85e-5 # Pa·s
    rho = 1.18    # kg/m³
    c_p = 1005.0  # J/kg/K

    D = 0.01
    U = np.linspace(0.05, 6.0, 8)

    h = h_cylinder(D, k, U, mu, rho, c_p)
    df = pd.DataFrame({"U [m/s]": U, "Re_D": rho*U*D/mu, "h [W/m²K]": h})
    print(df.to_string(index=False, float_format="%.3f"))
    

def compute_h(csv_path: str):
    in_path  = pathlib.Path(csv_path)
    out_path = in_path.with_name(in_path.stem + "_h.csv")

    df = pd.read_csv(in_path, comment='#')
    df['h'] = [h_cylinder(D, k, U, mu, rho, c_p) for k, mu, rho, c_p, D, U, Re_D in df.values]
    df.to_csv(out_path, index=False)
    print(f"✓ wrote {len(df)} rows to {out_path}")

# ---------- quick demo ----------
if __name__ == "__main__":
    compute_h("n1_pi_cylinder_design.csv")


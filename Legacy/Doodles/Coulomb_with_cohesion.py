# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:02:07 2025

@author: ibouckaert
"""

import numpy as np
import sympy as sp

sp.init_printing(use_unicode=True)

# Parameters
t, s, m, c, p = sp.symbols('tau sigma mu c psi')
kn, ks = sp.symbols('k_n, k_s')
vars = sp.Matrix([s, t])

# Yield surfaces
F1 = sp.Matrix([t + m * s - c])
F2 = sp.Matrix([-t + m * s - c])

G1 = sp.Matrix([t - c + p * s])
G2 = sp.Matrix([-t - c + p * s])
# G1 = sp.Matrix([t - c +])
# G2 = sp.Matrix([-t - c])

dF1 = F1.jacobian(vars)
dF2 = F2.jacobian(vars)
dG1 = G1.jacobian(vars)
dG2 = G2.jacobian(vars)

D_el = sp.Matrix([[kn, 0], [0, ks]])

# print(dF1, dF2, dG1, dG2)


de, dg = sp.symbols('d_e d_g')
ds, dt = sp.symbols('d_s d_t')
dl = sp.symbols('dl')
s = sp.Matrix([s, t])
deps = sp.Matrix([de, dg])
dsig = sp.Matrix([ds, dt])

# Projection on F1
eqs1 = dsig - D_el * (deps - dl * dG1.T)
eq2 = F1[0] + dt + m * ds

eqs = list(eqs1) + [eq2]

sol1 = sp.solve(eqs, [ds, dt, dl])
# sp.pprint(sol1[ds])
sp.latex(sol1[ds])

# Tangent stiffness when F1 is active
sp.pprint(D_el - (((D_el * dG1.T) * (D_el * dF1.T).T) / (dF1 * D_el * dG1.T)[0]))

# Projection on F2
eqs1 = dsig - D_el * (deps - dl * dG2.T)
eq2 = F2[0] - dt + m * ds

eqs = list(eqs1) + [eq2]

sol2 = sp.solve(eqs, [ds, dt, dl])
sp.pprint(sol2)

# Tangent stiffness when F1 is active
sp.pprint(D_el - (((D_el * dG2.T) * (D_el * dF2.T).T) / (dF2 * D_el * dG2.T)[0]))

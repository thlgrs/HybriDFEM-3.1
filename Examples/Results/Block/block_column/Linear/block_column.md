# Analysis Results: block_column


## Model Information
- **Structure Type**: Rigid Block Grid (DFEM)
- **Grid**: 20 x 80 blocks (1600 total)
- **Block Size**: 0.050 x 0.050 x 1.0 m
- **Total Size**: 1.0 x 4.0 x 1.0 m
- **Nodes**: 1600, **DOFs**: 4800
- **Material**: E = 30.0 GPa, nu = 0.2
- **Contact**: kn = 1e+12 N/m, ks = 1e+11 N/m
- **Contact Faces**: 3100
- **CP per CF**: 20
- **Analysis**: Linear static
- **Loading**: Fx = 500.0 kN, Fy = -500.0 kN, Mz = 0.0 kNm

## Displacements

**Control Node 1590:**
  - ux = 4.641246e-03 m  (4.6412 mm)
  - uy = -1.058285e-04 m  (-0.1058 mm)
  - rz = -1.615942e-03 rad  (-0.0926 deg)

**Maximum Displacements:**
  - Max |ux| = 4.653883e-03 m
  - Max |uy| = 8.809682e-04 m
  - Max |rz| = 1.822414e-03 rad

## Reaction Forces

**Total Base Reactions:**
  - Rx (horizontal) = -500000.00 N (-500.000 kN)
  - Ry (vertical)   = +500000.00 N (+500.000 kN)
  - Mz (moment)     = +2250000.00 Nm (+2250.000 kNm)

**Equilibrium Check:**
  - Applied Fx = +500000.00 N
  - Applied Fy = -500000.00 N
  - Applied Mz = +0.00 Nm
  - Reaction Rx = -500000.00 N
  - Reaction Ry = +500000.00 N
  - Reaction Mz = +2250000.00 Nm
  - Rel. error Fx = -8.83127e-12 [OK]
  - Rel. error Fy = -2.34880e-12 [OK]
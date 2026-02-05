# Analysis Report: Shear_BlockSquare

*Generated: 2026-01-22 15:16:05*


## Configuration
- **Structure Type**: Block

### Geometry
- **Dimensions**: 1.0 x 1.0 x 0.25 m (W x H x t)
- **Mesh**: 40 x 40 elements/blocks

### Material
- **E** = 10.0 GPa
- **nu** = 0.0

### Contact
- **kn** = 1.00e+10 N/m
- **ks** = 1.00e+10 N/m
- **Linear Geometry**: True
- **Contact Points/Face**: 20

### Loading
- **Fx** = 1000.0 kN

### Solver
- **Type**: linear

## Model Statistics
- **Nodes**: 1,600
- **Total DOFs**: 4,800
- **Fixed DOFs**: 120
- **Free DOFs**: 4,680
- **Blocks**: 1,600
- **Contact Faces**: 3,120

## Displacements

### Control Node 1599
- **ux** = 3.636295e-03 m (3.6363 mm)
- **uy** = -1.466550e-03 m (-1.4666 mm)
- **rz** = -6.816332e-03 rad (-0.3905 deg)

### Maximum Values
- **Max |ux|** = 3.636295e-03 m
- **Max |uy|** = 1.466550e-03 m
- **Max |rz|** = 6.816332e-03 rad
- **Max |U|** = 6.816332e-03 m

## Reaction Forces
**Total Base Reactions:**
- **Rx** = -1000000.00 N (-1000.000 kN)
- **Ry** = -0.00 N (-0.000 kN)
- **Mz** = +1000000.00 Nm (+1000.000 kNm)

## Equilibrium Check
| Direction | Applied | Reaction | Abs Error | Rel Error | Status |
|-----------|---------|----------|-----------|-----------|--------|
| Fx | +1000.00 kN | -1000.00 kN | 2.02e-07 N | 2.02e-13 | OK |
| Fy | +0.00 kN | -0.00 kN | 1.63e-07 N | 1.63e-07 | OK |

## Performance
- **Total Time**: 55.81 s

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| Model Creation | 15.861 | 28.4% |
| Boundary Conditions | 0.009 | 0.0% |
| Solver | 39.940 | 71.6% |
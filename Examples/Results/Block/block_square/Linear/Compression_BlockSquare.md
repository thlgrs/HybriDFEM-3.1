# Analysis Report: Compression_BlockSquare

*Generated: 2026-01-22 15:17:28*


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
- **Fy** = -1000.0 kN

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

### Control Node 1576
- **ux** = 1.545961e-04 m (0.1546 mm)
- **uy** = -7.231629e-04 m (-0.7232 mm)
- **rz** = -3.726537e-03 rad (-0.2135 deg)

### Maximum Values
- **Max |ux|** = 1.734858e-04 m
- **Max |uy|** = 8.693188e-04 m
- **Max |rz|** = 3.897738e-03 rad
- **Max |U|** = 3.897738e-03 m

## Reaction Forces
**Total Base Reactions:**
- **Rx** = +0.00 N (+0.000 kN)
- **Ry** = +1000000.00 N (+1000.000 kN)
- **Mz** = +500000.00 Nm (+500.000 kNm)

## Equilibrium Check
| Direction | Applied | Reaction | Abs Error | Rel Error | Status |
|-----------|---------|----------|-----------|-----------|--------|
| Fx | +0.00 kN | +0.00 kN | 4.71e-09 N | 4.71e-09 | OK |
| Fy | -1000.00 kN | +1000.00 kN | 8.31e-08 N | 8.31e-14 | OK |

## Performance
- **Total Time**: 72.08 s

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| Model Creation | 26.592 | 36.9% |
| Boundary Conditions | 0.021 | 0.0% |
| Solver | 45.470 | 63.1% |
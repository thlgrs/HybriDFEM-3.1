# Refactored Legacy Examples

This directory contains examples migrated from `Legacy/Examples/` to use the new Core API.

## API Mapping: Legacy → Core

### Structure Creation

| Legacy | Core |
|--------|------|
| `st.Structure_2D()` | `Structure_Block()` for pure blocks |
| `St.add_block(vertices, rho, b=B)` | `St.add_block_from_vertices(vertices, b=B, material=Material(rho=rho))` |
| `St.add_beam(N1, N2, n_blocks, h, rho, b=B, material=mat)` | `BeamBlock(N1, N2, n_blocks, h, rho, b=B, material=mat)` |
| `St.add_tapered_beam(N1, N2, n_blocks, h1, h2, rho, ...)` | `TaperedBeamBlock(N1, N2, n_blocks, h1, h2, rho, ...)` |
| `St.add_arch(c, a1, a2, R, n_blocks, h, rho, ...)` | `ArchBlock(c, a1, a2, R, n_blocks, h, rho, ...)` |
| `St.add_wall(c1, l_block, h_block, pattern, rho, ...)` | `WallBlock(c1, l_block, h_block, pattern, rho, ...)` |

### Node Management

| Legacy | Core |
|--------|------|
| `St.make_nodes()` | `St.make_nodes()` |
| `St.make_cfs(lin_geom, nb_cps, offset, contact, surface)` | `St.make_cfs(lin_geom, nb_cps, offset, contact, surface)` |

### Boundary Conditions

| Legacy | Core |
|--------|------|
| `St.fixNode(node_id, dofs)` | `St.fix_node(node_id, dofs)` |
| `St.loadNode(node_id, dofs, force, fixed=False)` | `St.load_node(node_id, dofs, force, fixed=False)` |
| `St.reset_loading()` | `St.reset_loading()` |
| `St.set_damping_properties(xsi, damp_type, stiff_type)` | `St.set_damping_properties(xsi, damp_type, stiff_type)` |

### Solvers

| Legacy | Core |
|--------|------|
| `St.solve_linear()` | `Static.solve(St)` or `StaticLinear.solve(St)` |
| `St.solve_forcecontrol(steps, ...)` | `StaticNonLinear.solve_forcecontrol(St, steps, ...)` |
| `St.solve_dispcontrol(steps, disp, node, dof, ...)` | `StaticNonLinear.solve_dispcontrol(St, steps, ...)` |
| `St.solve_modal(modes, no_inertia, ...)` | `Modal.solve_modal(St, modes, no_inertia, ...)` |
| `St.solve_dyn_linear(T, dt, Meth, ...)` | `Dynamic(T, dt, Meth=Meth, ...).linear(St)` |
| `St.solve_dyn_nonlinear(T, dt, Meth, lmbda, ...)` | `Dynamic(T, dt, Meth=Meth, lmbda=lmbda, ...).nonlinear(St)` |

### Materials

| Legacy | Core |
|--------|------|
| `mat.Material(E, nu, corr_fact, shear_def)` | `Material(E, nu, rho, corr_fact, shear_def)` from `Core.Objects.ConstitutiveLaw.Material` |
| `cont.Coulomb(kn, ks, mu)` | `Coulomb(kn, ks, mu, c, psi)` from `Core.Objects.DFEM` |
| `cont.NoTension_EP(kn, ks)` | `NoTension_EP(kn, ks)` from `Core.Objects.DFEM` |
| `cont.NoTension_CD(kn, ks)` | `NoTension_CD(kn, ks)` from `Core.Objects.DFEM` |
| `cont.Bilinear(kn, ks, fy, a)` | `Bilinear(kn, ks, fy, a)` from `Core.Objects.DFEM` |
| `surf.Surface(kn, ks)` | `Surface(kn, ks)` from `Core.Objects.DFEM` |

### Visualization

| Legacy | Core |
|--------|------|
| `St.plot_structure(scale, ...)` | `St.plot(show_deformed=True, deformation_scale=scale)` |
| `St.plot_def_structure(scale, ...)` | `St.plot(show_deformed=True, deformation_scale=scale)` |
| `St.plot_modes(n, scale, ...)` | Not yet implemented - use custom plotting |
| `St.save_structure(filename)` | `St.save_structure(filename)` |

### Results Access

| Legacy | Core |
|--------|------|
| `St.eig_vals` | `St.eig_vals` (after `Modal.solve_modal()`) |
| `St.eig_modes` | `St.eig_modes` (after `Modal.solve_modal()`) |
| `St.U` | `St.U` |
| `St.P` | `St.P` |
| `St.list_blocks[i].m` | `St.list_blocks[i].m` |
| `St.list_blocks[i].I` | `St.list_blocks[i].I` |

## Import Changes

### Legacy Imports
```python
from Legacy.Objects import Structure as st
from Legacy.Objects import Material as mat
import Legacy.Objects.Contact as cont
from Legacy.Objects import Surface as surf
```

### Core Imports
```python
from Core import Structure_Block, Static, StaticNonLinear, Dynamic, Modal, Visualizer
from Core.Structures import BeamBlock, TaperedBeamBlock, ArchBlock, WallBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.DFEM import Coulomb, NoTension_EP, NoTension_CD, Bilinear, Surface
```

## Examples in This Directory

### Modal_Analysis/
- `EigVals_Tapered_Beam.py` - Eigenvalue analysis of a tapered beam

### Linear_Dynamic/
- `Cantilever_FreeVib.py` - Free vibration of a cantilever beam

### Nonlinear_Dynamic/
- `Oneblock_Rocking.py` - Rocking motion of a single block

## Running Examples

```bash
# From the HybriDFEM directory
python Examples/Refactored_Legacy/Modal_Analysis/EigVals_Tapered_Beam.py
python Examples/Refactored_Legacy/Linear_Dynamic/Cantilever_FreeVib.py
python Examples/Refactored_Legacy/Nonlinear_Dynamic/Oneblock_Rocking.py
```

## Key Differences to Note

1. **Structure classes are now separate**: Use `Structure_Block` for pure block structures, `Structure_FEM` for pure FEM, and `Hybrid` for combined.

2. **Solvers are now static methods**: Instead of `St.solve_linear()`, use `Static.solve(St)`.

3. **Materials require density parameter**: In Core, materials are created with `Material(E, nu, rho=...)` where `rho` is passed as a keyword argument.

4. **Contact laws import from `Core.Objects.ConstitutiveLaw.Contact`**: All contact laws (Coulomb, NoTension_EP, NoTension_CD, Bilinear) are available from this module.

5. **Method naming convention changed**: `fixNode` → `fix_node`, `loadNode` → `load_node`.

6. **Block generators are separate classes**: `BeamBlock`, `TaperedBeamBlock`, `ArchBlock`, `WallBlock` are standalone Structure_Block subclasses.

7. **`contact=` vs `surface=` parameter**: When using `make_cfs()` with list-based contact point positions (e.g., `nb_cps=[-1, 0, 1]`), use the `surface=` parameter instead of `contact=`:
   ```python
   # When nb_cps is an integer (number of contact pairs):
   St.make_cfs(lin_geom=True, nb_cps=2, contact=Coulomb(kn, ks, mu))

   # When nb_cps is a list (contact point positions):
   St.make_cfs(lin_geom=False, nb_cps=[-1, 0, 1], surface=NoTension_CD(kn, ks), offset=-1)
   ```

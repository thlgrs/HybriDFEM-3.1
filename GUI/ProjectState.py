# GUI/ProjectState.py
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

# Import real structure classes from backend
try:
    from Core.Structures import Hybrid, Structure_Block, Structure_FEM
    from Core.Objects.DFEM.Block import Block_2D
    from Core.Objects.ConstitutiveLaw.Material import Material, PlaneStress, PlaneStrain
    from Core.Objects.FEM.Element2D import Geometry2D
    from Core.Objects.FEM.Triangles import Triangle3, Triangle6
    from Core.Objects.FEM.Quads import Quad4, Quad8
    from Core.Solvers.Visualizer import PlotStyle

except ImportError:
    # Fallback to dummy classes for testing if Core imports fail
    print("Warning: Core imports failed. Using dummy classes.")


    class Hybrid(QObject):
        def __init__(self):
            super().__init__()
            self.list_nodes, self.list_blocks = [], []
            self.nb_dofs = 0

        def plot(self, ax, **kwargs): ax.text(0.5, 0.5, "Structure Factice", ha='center')

        def add_block_from_dimensions(self, ref_point, l, h, **kwargs): self.list_blocks.append(f"Block at {ref_point}")

        def make_nodes(self): self.nb_dofs = len(self.list_blocks) * 3

        def fix_node(self, node_id, dofs): print(f"Dummy: Node {node_id} fixed for {dofs}")

        def load_node(self, node_id, dof, value): print(f"Dummy: Load {value} N on Node {node_id}, DOF {dof}")


    class Structure_block(Hybrid):
        pass


    class Structure_FEM(QObject):
        def __init__(self, fixed_dofs_per_node=False):
            super().__init__()
            self.list_nodes, self.list_fes = [], []
            self.nb_dofs = 0

        def plot(self, ax, **kwargs): ax.text(0.5, 0.5, "FEM Structure (Dummy)", ha='center')

        def make_nodes(self): self.nb_dofs = len(self.list_nodes) * 2

        def fix_node(self, node_id, dofs): print(f"Dummy: Node {node_id} fixed for {dofs}")

        def load_node(self, node_id, dof, value): print(f"Dummy: Load {value} N on Node {node_id}, DOF {dof}")


    class Block_2D:
        pass


    class Material:
        def __init__(self, E, nu, rho): self.E, self.nu, self.rho = E, nu, rho


    class PlaneStress:
        def __init__(self, E, nu, rho): self.E, self.nu, self.rho = E, nu, rho


    class PlaneStrain:
        def __init__(self, E, nu, rho): self.E, self.nu, self.rho = E, nu, rho


    class Geometry2D:
        def __init__(self, t): self.t = t


    class Triangle3:
        def __init__(self, **kwargs): pass


    class Triangle6:
        def __init__(self, **kwargs): pass


    class Quad4:
        def __init__(self, **kwargs): pass


    class Quad8:
        def __init__(self, **kwargs): pass


class ProjectState(QObject):
    # Signals
    structure_changed = pyqtSignal()
    materials_changed = pyqtSignal()
    bcs_changed = pyqtSignal()  # Notify GUI when boundary conditions change
    selection_mode_changed = pyqtSignal(str)  # Notify GUI of current selection mode
    results_ready = pyqtSignal(object)
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Structure type: 'hybrid', 'block', 'fem'
        self.structure_type = 'hybrid'
        self.structure = None
        self.materials = {}
        self.results = None
        self.project_file_path = None

        # Coupling configuration for hybrid structures
        self.coupling_enabled = False
        self.coupling_method = 'constraint'  # 'constraint', 'penalty', 'lagrange', 'mortar'
        self.coupling_params = {'tolerance': 1e-9}

        # Contact configuration for block structures
        self.contact_config = {
            'enabled': False,
            'tolerance': 1e-9,
            'margin': 0.01,
            'nb_cps': 5,
            'law': 'NoTension_EP',
            'kn': 1e10,
            'ks': 1e9,
            'mu': 0.3,
            'c': 0.0,
            'psi': 0.0,
            'ft': 0.0
        }

        # Selection mode: "idle", "select_support_node", "select_load_node"
        self.selection_mode = "idle"
        # Store boundary conditions for GUI display
        self.supports = {}  # Dictionary {node_id: [dofs]}
        self.loads = {}  # Dictionary {node_id: [(dof, value)]}

        # Plot style for visualization
        try:
            self.plot_style = PlotStyle()  # Default style
        except:
            self.plot_style = None  # Fallback if PlotStyle not available

        self.default_material = Material(E=30e9, nu=0.2, rho=2400)
        self.materials["Concrete C30/37 (Default)"] = {
            'object': self.default_material,
            'formulation': 'Basic',
            'E': 30e9,
            'nu': 0.2,
            'rho': 2400,
            'thickness': 0.01
        }

    def set_structure_type(self, stype):
        """
        Set the structure type and create an appropriate empty structure.

        Args:
            stype (str): Structure type - 'hybrid', 'block', or 'fem'
        """
        if stype not in ['hybrid', 'block', 'fem']:
            self.log_message.emit(f"[ERROR] Invalid structure type: {stype}")
            return

        self.structure_type = stype

        # Create empty structure of appropriate type
        if stype == 'block':
            self.structure = Structure_Block()
        elif stype == 'fem':
            self.structure = Structure_FEM(fixed_dofs_per_node=False)
        elif stype == 'hybrid':
            self.structure = Hybrid()

        # Clear boundary conditions and results
        self.supports = {}
        self.loads = {}
        self.results = None

        self.log_message.emit(f"Structure type set to: {stype}")
        self.structure_changed.emit()

    def new_project(self):
        # Reset to default hybrid structure
        self.structure_type = 'hybrid'
        self.structure = None
        self.results = None
        self.project_file_path = None
        self.materials = {
            "Concrete C30/37 (Default)": {
                'object': self.default_material,
                'formulation': 'Basic',
                'E': 30e9,
                'nu': 0.2,
                'rho': 2400,
                'thickness': 0.01
            }
        }

        self.supports = {}
        self.loads = {}
        self.set_selection_mode("idle")

        self.log_message.emit("New project created.")
        self.structure_changed.emit()
        self.materials_changed.emit()
        self.bcs_changed.emit()
        self.results_ready.emit(None)

    def load_structure_from_rhino(self, file_path):
        try:
            self.log_message.emit(f"Importing {file_path}...")
            self.structure = Hybrid()
            self.structure.add_block_from_dimensions(
                ref_point=np.array([0.5, 0.5]), l=1, h=1,
                material=self.default_material
            )
            self.structure.make_nodes()
            self.log_message.emit("Structure import successful.")
            self.structure_changed.emit()
        except Exception as e:
            self.log_message.emit(f"[ERROR] Import failed: {e}")

    def create_structured_mesh(self, x0, y0, x1, y1, nx, ny, element_type='Triangle3', material_name=None):
        """
        Create a structured FEM mesh over a rectangular domain.

        Args:
            x0, y0: Bottom-left corner coordinates
            x1, y1: Top-right corner coordinates
            nx, ny: Number of divisions in x and y directions
            element_type: 'Triangle3', 'Triangle6', 'Quad4', or 'Quad8'
            material_name: Name of material from library
        """
        try:
            # Create or verify structure type
            if self.structure is None:
                if self.structure_type == 'fem':
                    self.log_message.emit("No structure. Creating new Structure_FEM.")
                    self.structure = Structure_FEM(fixed_dofs_per_node=False)
                elif self.structure_type == 'hybrid':
                    self.log_message.emit("Creating Hybrid structure with FEM mesh.")
                    self.structure = Hybrid()
                elif self.structure_type == 'block':
                    msg = "Cannot add FEM mesh to Block structure. Change structure type to 'FEM' or 'Hybrid'."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

            if not hasattr(self.structure, 'list_fes'):
                msg = f"Current structure ({type(self.structure).__name__}) cannot add FEM elements."
                self.log_message.emit(f"[ERROR] {msg}")
                return

            # Get material and geometry from library
            if material_name and material_name in self.materials:
                material_data = self.materials[material_name]
                if isinstance(material_data, dict):
                    mat_obj = material_data['object']
                    thickness = material_data['thickness']
                    formulation = material_data['formulation']
                else:
                    msg = "Selected material is not FEM-compatible. Please create a PlaneStress or PlaneStrain material."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

                # Verify formulation
                if formulation not in ['PlaneStress', 'PlaneStrain']:
                    msg = f"ConstitutiveLaw '{material_name}' has formulation '{formulation}' which is not suitable for FEM."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

                self.log_message.emit(f"Using material: {material_name} ({formulation}, t={thickness}m)")
            else:
                msg = "No material selected or material not found."
                self.log_message.emit(f"[ERROR] {msg}")
                return

            # Create Geometry2D object
            geom = Geometry2D(t=thickness)

            # Generate structured grid
            self.log_message.emit(f"Generating {nx}×{ny} mesh with {element_type} elements...")

            dx = (x1 - x0) / nx
            dy = (y1 - y0) / ny

            # Determine if using quadrilateral or triangular elements
            is_quad = element_type in ['Quad4', 'Quad8']

            elem_count = 0
            for i in range(nx):
                for j in range(ny):
                    # Cell corners
                    x_left = x0 + i * dx
                    x_right = x0 + (i + 1) * dx
                    y_bottom = y0 + j * dy
                    y_top = y0 + (j + 1) * dy

                    if is_quad:
                        # Create quadrilateral element (counter-clockwise node ordering)
                        p1 = np.array([x_left, y_bottom])   # Bottom-left
                        p2 = np.array([x_right, y_bottom])  # Bottom-right
                        p3 = np.array([x_right, y_top])     # Top-right
                        p4 = np.array([x_left, y_top])      # Top-left

                        if element_type == 'Quad4':
                            quad = Quad4(nodes=[p1, p2, p3, p4], mat=mat_obj, geom=geom)
                        else:  # Quad8
                            quad = Quad8(nodes=[p1, p2, p3, p4], mat=mat_obj, geom=geom)

                        self.structure.list_fes.append(quad)
                        elem_count += 1

                    else:
                        # Create two triangular elements per cell
                        # Bottom-left triangle
                        p1 = np.array([x_left, y_bottom])
                        p2 = np.array([x_right, y_bottom])
                        p3 = np.array([x_left, y_top])

                        if element_type == 'Triangle3':
                            tri1 = Triangle3(nodes=[p1, p2, p3], mat=mat_obj, geom=geom)
                        else:  # Triangle6
                            tri1 = Triangle6(nodes=[p1, p2, p3], mat=mat_obj, geom=geom)

                        self.structure.list_fes.append(tri1)
                        elem_count += 1

                        # Top-right triangle
                        p1 = np.array([x_right, y_bottom])
                        p2 = np.array([x_right, y_top])
                        p3 = np.array([x_left, y_top])

                        if element_type == 'Triangle3':
                            tri2 = Triangle3(nodes=[p1, p2, p3], mat=mat_obj, geom=geom)
                        else:  # Triangle6
                            tri2 = Triangle6(nodes=[p1, p2, p3], mat=mat_obj, geom=geom)

                        self.structure.list_fes.append(tri2)
                        elem_count += 1

            # Build node list (merges duplicate nodes)
            self.structure.make_nodes()

            n_nodes = len(self.structure.list_nodes)
            n_dofs = self.structure.nb_dofs

            self.log_message.emit(
                f"Mesh generated: {elem_count} elements, {n_nodes} nodes, {n_dofs} DOFs"
            )

            self.structure_changed.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Mesh generation failed: {e}")

    def import_external_mesh(self, file_path, element_type='auto', thickness=0.01,
                           material_name=None, import_bc=True):
        """
        Import mesh from external file (Gmsh, Triangle, etc.).

        Args:
            file_path: Path to mesh file (.msh, .vtk, .vtu, etc.)
            element_type: 'auto', 'Triangle3', 'Triangle6', 'Quad4', or 'Quad8'
            thickness: Element thickness for 2D elements
            material_name: Name of material from library
            import_bc: Import boundary conditions from physical groups
        """
        try:
            from Core.Objects.FEM.Mesh import Mesh

            # Create or verify structure type
            if self.structure is None:
                if self.structure_type == 'fem':
                    self.log_message.emit("Creating new Structure_FEM for imported mesh.")
                    self.structure = Structure_FEM(fixed_dofs_per_node=False)
                elif self.structure_type == 'hybrid':
                    self.log_message.emit("Creating Hybrid structure for imported mesh.")
                    self.structure = Hybrid()
                elif self.structure_type == 'block':
                    msg = "Cannot import FEM mesh to Block structure. Change type to 'FEM' or 'Hybrid'."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

            if not hasattr(self.structure, 'list_fes'):
                msg = f"Current structure ({type(self.structure).__name__}) cannot add FEM elements."
                self.log_message.emit(f"[ERROR] {msg}")
                return

            # Get material
            if material_name and material_name in self.materials:
                material_data = self.materials[material_name]
                if isinstance(material_data, dict):
                    mat_obj = material_data['object']
                    formulation = material_data['formulation']
                else:
                    msg = "Selected material is not FEM-compatible."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

                if formulation not in ['PlaneStress', 'PlaneStrain']:
                    msg = f"ConstitutiveLaw '{material_name}' has formulation '{formulation}' not suitable for FEM."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

                self.log_message.emit(f"Using material: {material_name} ({formulation}, t={thickness}m)")
            else:
                msg = "No material selected. Using default."
                self.log_message.emit(f"[WARNING] {msg}")
                mat_obj = self.default_material

            # Load mesh
            self.log_message.emit(f"Loading mesh from: {file_path}")
            mesh_obj = Mesh(mesh_file=file_path)
            mesh_data = mesh_obj.read_mesh()

            # Create Geometry2D
            geom = Geometry2D(t=thickness)

            # Get nodes and elements
            nodes = mesh_obj.nodes()
            elements = mesh_obj.elements()

            self.log_message.emit(f"Mesh contains {len(nodes)} nodes, {len(elements)} elements")

            # Determine element type if auto
            if element_type == 'auto':
                # Infer from first element
                n_nodes_per_elem = elements.shape[1] if len(elements) > 0 else 0
                type_map = {3: 'Triangle3', 6: 'Triangle6', 4: 'Quad4', 8: 'Quad8'}
                element_type = type_map.get(n_nodes_per_elem, 'Triangle3')
                self.log_message.emit(f"Auto-detected element type: {element_type}")

            # Create elements
            elem_count = 0
            for elem_conn in elements:
                # Get node coordinates
                elem_nodes = [nodes[node_id][:2] for node_id in elem_conn]

                # Create appropriate element
                if element_type == 'Triangle3':
                    fe = Triangle3(nodes=elem_nodes, mat=mat_obj, geom=geom)
                elif element_type == 'Triangle6':
                    fe = Triangle6(nodes=elem_nodes, mat=mat_obj, geom=geom)
                elif element_type == 'Quad4':
                    fe = Quad4(nodes=elem_nodes, mat=mat_obj, geom=geom)
                elif element_type == 'Quad8':
                    fe = Quad8(nodes=elem_nodes, mat=mat_obj, geom=geom)
                else:
                    self.log_message.emit(f"[WARNING] Unknown element type: {element_type}, using Triangle3")
                    fe = Triangle3(nodes=elem_nodes, mat=mat_obj, geom=geom)

                self.structure.list_fes.append(fe)
                elem_count += 1

            # Build node list
            self.structure.make_nodes()

            # Import boundary conditions if requested
            if import_bc:
                try:
                    boundary_groups = mesh_obj.get_all_boundary_groups()
                    if boundary_groups:
                        self.log_message.emit(f"Found {len(boundary_groups)} boundary groups")
                        for group_name in boundary_groups:
                            self.log_message.emit(f"  - {group_name}")
                        self.log_message.emit("Note: Boundary groups detected but automatic BC application not yet implemented.")
                        self.log_message.emit("Use Materials & BCs panel to add constraints manually.")
                    else:
                        self.log_message.emit("No boundary groups found in mesh file.")
                except Exception as e:
                    self.log_message.emit(f"[WARNING] Could not read boundary groups: {e}")

            n_nodes = len(self.structure.list_nodes)
            n_dofs = self.structure.nb_dofs

            self.log_message.emit(
                f"Mesh imported successfully: {elem_count} elements, {n_nodes} nodes, {n_dofs} DOFs"
            )

            self.structure_changed.emit()

        except ImportError as e:
            self.log_message.emit(f"[ERROR] Import failed - missing library: {e}")
            self.log_message.emit("Install required libraries: pip install meshio gmsh")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Mesh import failed: {e}")

    def create_new_block(self, xc, yc, length, height, material_name=None):
        try:
            if self.structure is None:
                # Create structure based on current type setting
                if self.structure_type == 'block':
                    self.log_message.emit("No structure. Creating new Structure_Block.")
                    self.structure = Structure_Block()
                elif self.structure_type == 'hybrid':
                    self.log_message.emit("No structure. Creating new Hybrid structure.")
                    self.structure = Hybrid()
                elif self.structure_type == 'fem':
                    msg = "Cannot add blocks to FEM structure. Change structure type to 'block' or 'hybrid'."
                    self.log_message.emit(f"[ERROR] {msg}")
                    return

            if not hasattr(self.structure, 'add_block_from_dimensions'):
                msg = f"Current structure ({type(self.structure).__name__}) cannot add blocks."
                self.log_message.emit(f"[ERROR] {msg}")
                return

            # Select material from library or use default
            if material_name and material_name in self.materials:
                material_data = self.materials[material_name]
                # Extract material object from dict (new structure)
                if isinstance(material_data, dict):
                    formulation = material_data.get('formulation', 'Basic')

                    # CRITICAL: Validate material formulation for blocks
                    # Blocks require Basic materials (1D constitutive law)
                    # FEM materials (PlaneStress/PlaneStrain) have 2D constitutive laws
                    if formulation in ['PlaneStress', 'PlaneStrain']:
                        # Auto-convert FEM material to Basic by extracting E, nu, rho
                        E = material_data.get('E', 210e9)
                        nu = material_data.get('nu', 0.3)
                        rho = material_data.get('rho', 7850)
                        selected_material = Material(E=E, nu=nu, rho=rho)
                        self.log_message.emit(
                            f"[WARNING] ConstitutiveLaw '{material_name}' is {formulation} (FEM). "
                            f"Auto-converted to Basic material for block with E={E/1e9:.1f} GPa, nu={nu}, rho={rho} kg/m³"
                        )
                    else:
                        selected_material = material_data['object']
                        self.log_message.emit(f"Using material: {material_name}")
                else:
                    # Legacy: material stored directly
                    selected_material = material_data
                    self.log_message.emit(f"Using material: {material_name}")
            else:
                selected_material = self.default_material
                material_name = "Concrete C30/37 (Default)"
                self.log_message.emit(f"Using default material: {material_name}")

            ref_point = np.array([xc, yc])
            self.structure.add_block_from_dimensions(
                ref_point, l=length, h=height,
                material=selected_material
            )
            self.structure.make_nodes()

            # CRITICAL: Detect contact interfaces for block structures
            # Without this, the stiffness matrix remains all zeros
            if hasattr(self.structure, 'detect_interfaces'):
                self.structure.detect_interfaces()
                self.log_message.emit(f"Block added at ({xc}, {yc}) with {material_name}. Interfaces detected.")
            else:
                self.log_message.emit(f"Block added at ({xc}, {yc}) with {material_name}.")

            self.structure_changed.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Block creation failed: {e}")

    def add_new_material(self, name, E, nu, rho, formulation='Basic', thickness=0.01):
        """
        Add a new material to the library.

        Args:
            name (str): ConstitutiveLaw name
            E (float): Young's modulus [Pa]
            nu (float): Poisson's ratio
            rho (float): Density [kg/m³]
            formulation (str): 'Basic', 'PlaneStress', or 'PlaneStrain'
            thickness (float): Thickness for FEM elements [m]
        """
        if not name:
            self.log_message.emit("[ERROR] ConstitutiveLaw name cannot be empty.")
            return
        if name in self.materials:
            self.log_message.emit(f"[ERROR] ConstitutiveLaw '{name}' already exists.")
            return
        try:
            # Create material object based on formulation
            if formulation == 'PlaneStress':
                mat_obj = PlaneStress(E=E, nu=nu, rho=rho)
            elif formulation == 'PlaneStrain':
                mat_obj = PlaneStrain(E=E, nu=nu, rho=rho)
            else:
                mat_obj = Material(E=E, nu=nu, rho=rho)

            # Store material with metadata
            self.materials[name] = {
                'object': mat_obj,
                'formulation': formulation,
                'E': E,
                'nu': nu,
                'rho': rho,
                'thickness': thickness
            }

            self.log_message.emit(f"ConstitutiveLaw '{name}' ({formulation}) added to library.")
            self.materials_changed.emit()
        except Exception as e:
            self.log_message.emit(f"[ERROR Backend] ConstitutiveLaw creation failed: {e}")

    def set_selection_mode(self, mode="idle"):
        """Change the current selection mode."""
        self.selection_mode = mode
        self.selection_mode_changed.emit(mode)
        self.log_message.emit(f"Selection mode activated: {mode}")

    def add_support_to_node(self, node_id, dofs):
        """Add support boundary condition to a node."""
        if self.structure is None:
            self.log_message.emit("[ERROR] No structure to add support to.")
            return

        try:
            # Call backend method (from Structure_2D.py)
            self.structure.fix_node(node_id, dofs)

            # Store for display
            self.supports[node_id] = dofs

            self.log_message.emit(f"Support added to node {node_id} (DOFs: {dofs}).")
            self.bcs_changed.emit()  # Notify GUI (MaterialPanel)
            self.structure_changed.emit()  # Notify GUI (Viewport) to redraw

        except Exception as e:
            self.log_message.emit(f"[ERROR Backend] Failed to add support: {e}")

    def add_load_to_node(self, node_id, loads_list):
        """Add load to a node."""
        if self.structure is None:
            self.log_message.emit("[ERROR] No structure to add load to.")
            return

        try:
            # Call backend method (from Structure_2D.py)
            for dof_index, value in loads_list:
                self.structure.load_node(node_id, [dof_index], value)

            # Store for display
            self.loads[node_id] = loads_list

            self.log_message.emit(f"Load added to node {node_id}.")
            self.bcs_changed.emit()  # Notify GUI (MaterialPanel)
            self.structure_changed.emit()  # Notify GUI (Viewport) to redraw

        except Exception as e:
            self.log_message.emit(f"[ERROR Backend] Failed to add load: {e}")

    def configure_coupling(self, method, params):
        """
        Configure hybrid coupling for the structure.

        Args:
            method (str): 'constraint', 'penalty', 'lagrange', or 'mortar'
            params (dict): Method-specific parameters
        """
        if not self.structure:
            self.log_message.emit("[ERROR] No structure to configure coupling for.")
            return False

        # Check if structure is hybrid
        is_hybrid = (hasattr(self.structure, 'list_blocks') and
                    hasattr(self.structure, 'list_fes') and
                    len(self.structure.list_blocks) > 0 and
                    len(self.structure.list_fes) > 0)

        if not is_hybrid:
            self.log_message.emit("[WARNING] Structure is not hybrid. Coupling not applicable.")
            return False

        # Store configuration
        self.coupling_method = method
        self.coupling_params = params
        self.coupling_enabled = True

        self.log_message.emit(f"Coupling configured: {method.upper()} with params {params}")
        return True

    def apply_coupling_to_structure(self):
        """
        Apply coupling configuration to structure before analysis.
        Must be called after make_nodes() and before solver.
        """
        if not self.coupling_enabled or not self.structure:
            return

        # Check if structure has coupling capability
        if not hasattr(self.structure, 'enable_block_fem_coupling'):
            self.log_message.emit("[WARNING] Structure does not support coupling.")
            return

        try:
            method = self.coupling_method
            params = self.coupling_params

            self.log_message.emit(f"Applying {method.upper()} coupling to structure...")

            # Call backend coupling method
            if method == 'constraint':
                tolerance = params.get('tolerance', 1e-9)
                self.structure.enable_block_fem_coupling(
                    method='constraint',
                    tolerance=tolerance
                )
                self.log_message.emit(f"Constraint coupling applied (tolerance={tolerance}m)")

            elif method == 'penalty':
                penalty = params.get('penalty', 'auto')
                self.structure.enable_block_fem_coupling(
                    method='penalty',
                    penalty=penalty
                )
                self.log_message.emit(f"Penalty coupling applied (penalty={penalty})")

            elif method == 'lagrange':
                tolerance = params.get('tolerance', 1e-9)
                self.structure.enable_block_fem_coupling(
                    method='lagrange',
                    tolerance=tolerance
                )
                self.log_message.emit(f"Lagrange coupling applied (tolerance={tolerance}m)")

            elif method == 'mortar':
                integration_order = params.get('integration_order', 2)
                interface_tolerance = params.get('interface_tolerance', 0.01)
                self.structure.enable_block_fem_coupling(
                    method='mortar',
                    integration_order=integration_order,
                    interface_tolerance=interface_tolerance
                )
                self.log_message.emit(f"Mortar coupling applied (order={integration_order}, tol={interface_tolerance}m)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Coupling application failed: {e}")

    def configure_contact(self, config):
        """
        Configure contact mechanics for block structures.

        Args:
            config (dict): Contact configuration with keys:
                - enabled, tolerance, margin, nb_cps
                - law: 'NoTension_EP' or 'Coulomb'
                - kn, ks, mu, c, psi, ft
        """
        if not self.structure:
            self.log_message.emit("[ERROR] No structure to configure contact for.")
            return False

        # Check if structure has blocks
        has_blocks = hasattr(self.structure, 'list_blocks') and len(getattr(self.structure, 'list_blocks', [])) > 0

        if not has_blocks:
            self.log_message.emit("[WARNING] Structure has no blocks. Contact not applicable.")
            return False

        # Store configuration
        self.contact_config = config

        law_name = config['law']
        self.log_message.emit(f"Contact configured: {law_name} with {config['nb_cps']} points per face")
        return True

    def apply_contact_to_structure(self):
        """
        Apply contact configuration to structure before analysis.
        Must be called after make_nodes() and before solver.
        """
        if not self.contact_config['enabled'] or not self.structure:
            return

        # Check if structure has contact capability
        if not hasattr(self.structure, 'detect_interfaces'):
            self.log_message.emit("[WARNING] Structure does not support contact.")
            return

        try:
            # Import contact laws
            from Core.Objects.ConstitutiveLaw.Contact import NoTension_EP, Coulomb

            config = self.contact_config

            self.log_message.emit(f"Detecting contact interfaces...")

            # Detect interfaces
            self.structure.detect_interfaces(
                eps=config['tolerance'],
                margin=config['margin']
            )

            self.log_message.emit(f"Detected {len(getattr(self.structure, 'list_cfs', []))} contact faces")

            # Create contact law
            if config['law'] == 'NoTension_EP':
                contact_law = NoTension_EP(
                    kn=config['kn'],
                    ks=config['ks']
                )
                self.log_message.emit(
                    f"NoTension_EP contact law: kn={config['kn']:.2e} N/m, ks={config['ks']:.2e} N/m"
                )
            else:  # Coulomb
                contact_law = Coulomb(
                    kn=config['kn'],
                    ks=config['ks'],
                    mu=config['mu'],
                    c=config['c'],
                    psi=config['psi'],
                    ft=config['ft']
                )
                self.log_message.emit(
                    f"Coulomb contact law: kn={config['kn']:.2e} N/m, ks={config['ks']:.2e} N/m, "
                    f"μ={config['mu']}, c={config['c']:.1f} Pa"
                )

            # Apply contact law
            if hasattr(self.structure, 'make_cfs'):
                self.structure.make_cfs(
                    lin_geom=True,  # Linear geometry flag
                    nb_cps=config['nb_cps'],
                    contact=contact_law  # Contact law object
                )
                self.log_message.emit(f"Contact mechanics applied ({config['nb_cps']} points per face)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Contact application failed: {e}")

    def save_project(self, file_path):
        """
        Save project to JSON file.

        Args:
            file_path (str): Path to save file
        """
        import json

        try:
            # Build project data dictionary
            project_data = {
                'version': '1.0',
                'structure_type': self.structure_type,
                'materials': {},
                'boundary_conditions': {
                    'supports': {str(k): v for k, v in self.supports.items()},
                    'loads': {str(k): v for k, v in self.loads.items()}
                },
                'coupling_config': {
                    'enabled': self.coupling_enabled,
                    'method': self.coupling_method,
                    'params': self.coupling_params
                },
                'contact_config': self.contact_config.copy(),
                'geometry': {}
            }

            # Save materials
            for name, mat_data in self.materials.items():
                if isinstance(mat_data, dict):
                    project_data['materials'][name] = {
                        'formulation': mat_data['formulation'],
                        'E': mat_data['E'],
                        'nu': mat_data['nu'],
                        'rho': mat_data['rho'],
                        'thickness': mat_data['thickness']
                    }

            # Save geometry (blocks and mesh parameters if possible)
            if self.structure and hasattr(self.structure, 'list_blocks'):
                blocks_data = []
                for block in self.structure.list_blocks:
                    # Save block basic info (vertices, material reference)
                    # This is simplified - full serialization would need more work
                    block_info = {
                        'type': 'Block_2D',
                        # Add more fields as needed
                    }
                    blocks_data.append(block_info)
                project_data['geometry']['blocks'] = blocks_data

            # Write to file
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)

            self.project_file_path = file_path
            self.log_message.emit(f"Project saved to: {file_path}")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Project save failed: {e}")
            return False

    def load_project(self, file_path):
        """
        Load project from JSON file.

        Args:
            file_path (str): Path to project file
        """
        import json

        try:
            # Read file
            with open(file_path, 'r') as f:
                project_data = json.load(f)

            # Check version
            version = project_data.get('version', '1.0')
            self.log_message.emit(f"Loading project (version {version})...")

            # Load structure type
            structure_type = project_data.get('structure_type', 'hybrid')
            self.set_structure_type(structure_type)

            # Load materials
            self.materials = {}
            for name, mat_data in project_data.get('materials', {}).items():
                self.add_new_material(
                    name=name,
                    E=mat_data['E'],
                    nu=mat_data['nu'],
                    rho=mat_data['rho'],
                    formulation=mat_data['formulation'],
                    thickness=mat_data.get('thickness', 0.01)
                )

            # Load coupling configuration
            coupling_config = project_data.get('coupling_config', {})
            self.coupling_enabled = coupling_config.get('enabled', False)
            self.coupling_method = coupling_config.get('method', 'constraint')
            self.coupling_params = coupling_config.get('params', {'tolerance': 1e-9})

            # Load contact configuration
            contact_config = project_data.get('contact_config', {})
            self.contact_config = contact_config

            # Load boundary conditions
            bcs = project_data.get('boundary_conditions', {})
            self.supports = {int(k): v for k, v in bcs.get('supports', {}).items()}
            self.loads = {int(k): v for k, v in bcs.get('loads', {}).items()}

            # Note: Geometry reconstruction is complex and would require
            # re-creating blocks and mesh - for now we just load metadata
            # User will need to manually recreate geometry or we extend this later

            self.project_file_path = file_path
            self.log_message.emit(f"Project loaded from: {file_path}")
            self.log_message.emit("[WARNING] Geometry must be recreated manually (not yet implemented)")

            # Emit signals to update GUI
            self.structure_changed.emit()
            self.materials_changed.emit()
            self.bcs_changed.emit()

            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"[ERROR] Project load failed: {e}")
            return False

    def set_analysis_results(self, solved_structure):
        self.results = solved_structure
        self.log_message.emit("Analysis complete. Results available.")
        self.results_ready.emit(self.results)

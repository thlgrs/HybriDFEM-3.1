from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class MortarMaster(ABC):
    """
    Abstract base class for the 'Master' side of a Mortar interface.
    
    Abstraction allows MortarCoupling to couple with either:
    1. Rigid Blocks (BlockMaster)
    2. Deformable Elements (FEMMaster)
    """

    @abstractmethod
    def get_dof_indices(self, structure) -> List[int]:
        """Return global DOF indices associated with this master entity."""
        pass

    @abstractmethod
    def evaluate_kinematics(self, x_phys: np.ndarray, structure) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate kinematic coupling terms at a physical point x_phys.
        
        Parameters
        ----------
        x_phys : np.ndarray
            Physical coordinates (x, y) of the integration point (on slave surface).
        structure : Hybrid
            Reference to structure for geometry access.
            
        Returns
        -------
        N_master : np.ndarray
            Shape function values (or rigid body equivalents) mapping Master DOFs 
            to displacement at x_phys. Shape: (2, n_master_dofs) usually.
            
            For 2D:
            u(x) = N_master @ u_master_dofs
            
        valid : bool
            True if x_phys can be projected onto this master entity.
        """
        pass


class BlockMaster(MortarMaster):
    """
    Wrapper for a Rigid Block acting as a Master surface.
    """

    def __init__(self, block_id: int):
        self.block_id = block_id

    def get_dof_indices(self, structure) -> List[int]:
        block = structure.list_blocks[self.block_id]
        block_node_id = block.connect
        offset = structure.node_dof_offsets[block_node_id]
        # u, v, theta
        return [offset, offset + 1, offset + 2]

    def evaluate_kinematics(self, x_phys: np.ndarray, structure) -> Tuple[np.ndarray, bool]:
        """
        Rigid body kinematics:
        u_x = u_ref - (y - y_ref) * theta
        u_y = v_ref + (x - x_ref) * theta
        """
        block = structure.list_blocks[self.block_id]
        block_node_id = block.connect
        ref_pos = np.array(structure.list_nodes[block_node_id])

        dx = x_phys[0] - ref_pos[0]
        dy = x_phys[1] - ref_pos[1]

        # Kinematic Matrix N [2 x 3]
        # [ u_ref, v_ref, theta ]
        # row 0 (u_x): [ 1, 0, -dy ]
        # row 1 (u_y): [ 0, 1,  dx ]

        N_master = np.array([
            [1.0, 0.0, -dy],
            [0.0, 1.0, dx]
        ])

        return N_master, True


from Core.Objects.Coupling.InterfaceDetection import point_to_segment_distance


class FEMMaster(MortarMaster):
    """
    Wrapper for a Finite Element Edge acting as a Master surface.
    """

    def __init__(self, element_id: int, edge_index: int):
        self.element_id = element_id
        self.edge_index = edge_index  # [0, 1, 2] for Triangle, [0..3] for Quad

    def get_dof_indices(self, structure) -> List[int]:
        element = structure.list_fes[self.element_id]
        dofs = []
        for node in element.connect:
            off = structure.node_dof_offsets[node]
            dofs.extend([off, off + 1])  # Assuming 2D u,v
        return dofs

    def evaluate_kinematics(self, x_phys: np.ndarray, structure) -> Tuple[np.ndarray, bool]:
        """
        Deformable kinematics:
        u(x) = sum N_i(xi) * u_i
        """
        element = structure.list_fes[self.element_id]

        # 1. Identify Edge Nodes
        # Hardcoded connectivity for standard 2D elements
        if element.nd == 3:  # T3
            edges = [(0, 1), (1, 2), (2, 0)]
        elif element.nd == 4:  # Q4
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif element.nd == 6:  # T6
            edges = [(0, 3, 1), (1, 4, 2), (2, 5, 0)]  # Corner-Mid-Corner
        elif element.nd == 8 or element.nd == 9:  # Q8/Q9
            edges = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
        else:
            raise NotImplementedError(f"Element type with {element.nd} nodes not supported as Master.")

        edge_local_ids = edges[self.edge_index]

        # Get physical coordinates of edge vertices (endpoints)
        # For quadratic edges, we define the "segment" by corners for projection direction
        n0_idx = edge_local_ids[0]
        n1_idx = edge_local_ids[-1]  # Last node is the other corner

        v0 = np.array(element.nodes[n0_idx])
        v1 = np.array(element.nodes[n1_idx])

        # 2. Project x_phys to Edge Segment
        # This gives us t in [0, 1] relative to the straight line between corners
        dist, t = point_to_segment_distance(x_phys, v0, v1)

        # 3. Evaluate 1D Shape Functions
        # N_edge: shape functions for the nodes ON THE EDGE
        n_edge_nodes = len(edge_local_ids)
        if n_edge_nodes == 2:  # Linear
            xi = t  # [0, 1]
            N_edge = np.array([1.0 - xi, xi])
        elif n_edge_nodes == 3:  # Quadratic
            xi = t  # [0, 1]
            # Standard quadratic shape functions on [0, 1]
            # 0(start), 1(mid), 2(end) -> mapped to N0, N1, N2
            # Using standard Lagrange polynomials
            N0 = (1.0 - xi) * (1.0 - 2.0 * xi)
            N1 = 4.0 * xi * (1.0 - xi)
            N2 = xi * (2.0 * xi - 1.0)
            N_edge = np.array([N0, N1, N2])

        # 4. Construct Full N Matrix
        # Size: 2 x (2 * n_element_nodes)
        n_elem_nodes = element.nd
        N_full = np.zeros((2, 2 * n_elem_nodes))

        # Fill in contributions
        for i, local_node_idx in enumerate(edge_local_ids):
            val = N_edge[i]

            u_idx = 2 * local_node_idx
            v_idx = 2 * local_node_idx + 1

            N_full[0, u_idx] = val
            N_full[1, v_idx] = val

        return N_full, True

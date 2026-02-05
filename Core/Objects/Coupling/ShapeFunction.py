from typing import Optional, Tuple, List

import numpy as np
from scipy.spatial import cKDTree


def find_element_containing_point(structure, point: np.ndarray,
                                  candidates: Optional[List[int]] = None,
                                  tolerance: float = 1e-6) -> Optional[int]:
    """
    Find which FEM element contains a given point.

    Uses a two-stage approach:
    1. Proximity search: Find elements whose nodes are near the point
    2. Inverse mapping: Check if point is actually inside element

    Parameters
    ----------
    structure : Hybrid
        Structure with FEM elements
    point : np.ndarray
        Physical coordinates (x, y)
    candidates : List[int], optional
        List of candidate element IDs to search (if None, search all)
    tolerance : float
        Tolerance for inverse mapping convergence

    Returns
    -------
    int or None
        Element ID if found, None if point not in any element

    Notes
    -----
    For Triangle3/Triangle6 elements, a point is inside if its natural
    coordinates (ξ, η) satisfy:
        ξ ≥ 0, η ≥ 0, ξ+η ≤ 1
    """
    if candidates is None:
        candidates = list(range(len(structure.list_fes)))

    # Try each candidate element
    for elem_id in candidates:
        element = structure.list_fes[elem_id]

        try:
            # Attempt inverse mapping
            xi, eta, converged = inverse_mapping_newton(element, point, tolerance)

            if converged:
                # Check if natural coordinates are within element bounds
                # Use same tolerance as inverse mapping for consistency
                if is_point_inside_element(xi, eta, element, tolerance=tolerance):
                    return elem_id

        except:
            # Inverse mapping failed, try next element
            continue

    return None


def is_point_inside_element(xi: float, eta: float, element,
                            tolerance: float = 1e-6) -> bool:
    """
    Check if natural coordinates are inside element bounds.

    Parameters
    ----------
    xi, eta : float
        Natural coordinates
    element : Element2D
        FEM element
    tolerance : float
        Tolerance for boundary checks (relaxed to 1e-6 to accommodate
        edge points for mortar coupling)

    Returns
    -------
    bool
        True if point is inside element

    Notes
    -----
    For triangular elements in standard parametrization:
        Inside if: ξ ≥ -tol, η ≥ -tol, ξ+η ≤ 1+tol
    """
    # Triangular element bounds
    inside = (xi >= -tolerance and
             eta >= -tolerance and
             xi + eta <= 1.0 + tolerance)

    return inside


def inverse_mapping_newton(element, point: np.ndarray,
                           tolerance: float = 1e-5,
                           max_iterations: int = 50) -> Tuple[float, float, bool]:
    """
    Find natural coordinates (ξ, η) for a physical point (x, y).

    Uses Newton-Raphson iteration:
        Given x_target, y_target
        Find ξ, η such that x(ξ,η) = x_target, y(ξ,η) = y_target

    Parameters
    ----------
    element : Element2D
        FEM element
    point : np.ndarray
        Target physical coordinates [x, y]
    tolerance : float
        Convergence tolerance for position error
    max_iterations : int
        Maximum Newton iterations

    Returns
    -------
    xi : float
        Natural coordinate ξ
    eta : float
        Natural coordinate η
    converged : bool
        True if iteration converged

    Algorithm
    ---------
    Newton iteration:
        [δξ  ]   [J]⁻¹ [x_target - x(ξ,η)]
        [δη  ] =      [y_target - y(ξ,η)]

    where J is the Jacobian matrix:
        J = [∂x/∂ξ   ∂y/∂ξ ]
            [∂x/∂η   ∂y/∂η ]
    """
    x_target, y_target = point

    # Initial guess: centroid of element
    xi = 1.0 / 3.0
    eta = 1.0 / 3.0

    for iteration in range(max_iterations):
        # Get shape functions and derivatives at current (ξ, η)
        N, dN_dxi, dN_deta = element.N_dN(xi, eta)

        # Current physical position
        x_current = sum(N[i] * element.nodes[i][0] for i in range(element.nd))
        y_current = sum(N[i] * element.nodes[i][1] for i in range(element.nd))

        # Residual
        rx = x_target - x_current
        ry = y_target - y_current

        # Check convergence
        residual_norm = np.sqrt(rx**2 + ry**2)
        if residual_norm < tolerance:
            return xi, eta, True

        # Compute Jacobian matrix J
        dx_dxi = sum(dN_dxi[i] * element.nodes[i][0] for i in range(element.nd))
        dy_dxi = sum(dN_dxi[i] * element.nodes[i][1] for i in range(element.nd))
        dx_deta = sum(dN_deta[i] * element.nodes[i][0] for i in range(element.nd))
        dy_deta = sum(dN_deta[i] * element.nodes[i][1] for i in range(element.nd))

        J = np.array([[dx_dxi, dy_dxi],
                     [dx_deta, dy_deta]])

        # Check for singular Jacobian
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-14:
            return xi, eta, False

        # Solve for increment: J * [δξ, δη]^T = [rx, ry]^T
        try:
            delta = np.linalg.solve(J, np.array([rx, ry]))
        except np.linalg.LinAlgError:
            return xi, eta, False

        # Update natural coordinates
        xi += delta[0]
        eta += delta[1]

        # Clamp to reasonable bounds (prevent divergence)
        xi = np.clip(xi, -0.5, 1.5)
        eta = np.clip(eta, -0.5, 1.5)

    # Max iterations reached without convergence
    return xi, eta, False


def evaluate_shape_functions_at_point(structure, point: np.ndarray,
                                      candidate_elements: Optional[List[int]] = None,
                                      tolerance: float = 1e-6) -> Optional[Tuple[int, np.ndarray, float, float]]:
    """
    Evaluate shape functions at a physical point.

    This is the complete pipeline:
    1. Find element containing the point
    2. Compute natural coordinates via inverse mapping
    3. Evaluate shape functions at those coordinates

    Parameters
    ----------
    structure : Hybrid
        Structure with FEM elements
    point : np.ndarray
        Physical coordinates (x, y)
    candidate_elements : List[int], optional
        Candidate elements to search (None = search all)
    tolerance : float
        Convergence tolerance

    Returns
    -------
    element_id : int
        Element containing the point
    shape_functions : np.ndarray
        Shape function values [N₁, N₂, ...]
    xi : float
        Natural coordinate ξ
    eta : float
        Natural coordinate η

    Returns None if point is not inside any element.

    Example
    -------
    >>> result = evaluate_shape_functions_at_point(structure, np.array([0.5, 0.5]))
    >>> if result is not None:
    ...     elem_id, N, xi, eta = result
    ...     print(f"Point is in element {elem_id}")
    ...     print(f"Shape functions: {N}")
    """
    # Find containing element
    elem_id = find_element_containing_point(structure, point,
                                           candidates=candidate_elements,
                                           tolerance=tolerance)

    if elem_id is None:
        return None

    # Get element
    element = structure.list_fes[elem_id]

    # Inverse mapping to get natural coordinates
    xi, eta, converged = inverse_mapping_newton(element, point, tolerance)

    if not converged:
        return None

    # Evaluate shape functions
    N, dN_dxi, dN_deta = element.N_dN(xi, eta)

    return elem_id, N, xi, eta


def find_candidate_elements_near_point(structure, point: np.ndarray,
                                       search_radius: float) -> List[int]:
    """
    Find candidate elements near a point using spatial search.

    Uses a KD-tree built from element centroids for fast proximity search.

    Parameters
    ----------
    structure : Hybrid
        Structure with FEM elements
    point : np.ndarray
        Query point
    search_radius : float
        Search radius

    Returns
    -------
    List[int]
        Indices of elements within search radius

    Notes
    -----
    This is used to reduce the search space before attempting inverse mapping.
    Typical workflow:
        1. find_candidate_elements_near_point (fast, approximate)
        2. find_element_containing_point on candidates (slow, exact)
    """
    # Compute element centroids
    centroids = []
    for element in structure.list_fes:
        centroid = np.mean(element.nodes, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Build KD-tree
    tree = cKDTree(centroids)

    # Query for elements within radius
    indices = tree.query_ball_point(point, search_radius)

    return indices


def batch_evaluate_shape_functions(structure, points: List[np.ndarray],
                                   interface_ids: Optional[List[int]] = None) -> List[Optional[Tuple]]:
    """
    Evaluate shape functions at multiple points (batch operation).

    Parameters
    ----------
    structure : Hybrid
        Structure with FEM elements
    points : List[np.ndarray]
        List of physical points
    interface_ids : List[int], optional
        Interface IDs corresponding to points (for grouping)

    Returns
    -------
    List[Optional[Tuple]]
        List of results, one per point
        Each result is (elem_id, N, xi, eta) or None

    Notes
    -----
    This is more efficient than calling evaluate_shape_functions_at_point
    repeatedly because:
    1. Can build spatial data structures once
    2. Can group points by interface
    3. Can reuse candidates for nearby points
    """
    results = []

    for i, point in enumerate(points):
        # Find candidate elements (within reasonable distance)
        # Assume integration points are near mesh, so use small search radius
        candidates = find_candidate_elements_near_point(structure, point, search_radius=0.1)

        if not candidates:
            # Expand search if no candidates found
            candidates = None  # Search all elements

        # Evaluate shape functions
        result = evaluate_shape_functions_at_point(structure, point, candidates)

        results.append(result)

    return results

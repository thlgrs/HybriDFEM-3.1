from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class IntegrationPoint:
    """
    Integration point for mortar coupling.

    Attributes
    ----------
    position : np.ndarray
        Physical coordinates (x, y)
    weight : float
        Integration weight (includes Jacobian)
    normal : np.ndarray
        Interface normal at this point
    tangent : np.ndarray
        Interface tangent at this point
    parametric_coord : float
        Parametric coordinate along interface ξ ∈ [0, 1]
    interface_id : int
        Which interface this point belongs to
    """
    position: np.ndarray       # (x, y)
    weight: float              # w * |dx/dξ|
    normal: np.ndarray         # (nx, ny)
    tangent: np.ndarray        # (tx, ty)
    parametric_coord: float    # ξ ∈ [0, 1]
    interface_id: int = 0


def gauss_points_1d(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre quadrature points and weights for interval [0, 1].

    Parameters
    ----------
    order : int
        Integration order (1, 2, or 3)
        - order 1: 1-point rule (exact for linear)
        - order 2: 2-point rule (exact for cubic)
        - order 3: 3-point rule (exact for quintic)

    Returns
    -------
    points : np.ndarray
        Gauss points in [0, 1]
    weights : np.ndarray
        Corresponding weights (sum to 1.0)

    Notes
    -----
    Standard Gauss-Legendre rules are for [-1, 1], but we transform
    to [0, 1] for convenience with parametric coordinates.

    For mortar methods with linear elements, order=2 is typically sufficient.
    """
    if order == 1:
        # 1-point rule (midpoint)
        points = np.array([0.5])
        weights = np.array([1.0])

    elif order == 2:
        # 2-point Gauss rule
        # Points at (1 ± 1/√3)/2 in [0,1]
        alpha = 1.0 / np.sqrt(3.0)
        points = np.array([
            0.5 - alpha/2.0,  # ≈ 0.211
            0.5 + alpha/2.0   # ≈ 0.789
        ])
        weights = np.array([0.5, 0.5])

    elif order == 3:
        # 3-point Gauss rule
        # Points at (1 ± √(3/5))/2 and 1/2 in [0,1]
        alpha = np.sqrt(3.0 / 5.0)
        points = np.array([
            0.5 - alpha/2.0,  # ≈ 0.113
            0.5,               # = 0.5
            0.5 + alpha/2.0   # ≈ 0.887
        ])
        weights = np.array([5.0/18.0, 8.0/18.0, 5.0/18.0])

    else:
        raise ValueError(f"Unsupported integration order: {order}. "
                        f"Supported orders: 1, 2, 3")

    return points, weights


def gauss_points_2d_triangle(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss quadrature for triangular domains.

    Parameters
    ----------
    order : int
        Integration order (1, 2, or 3)

    Returns
    -------
    points : np.ndarray
        Gauss points in natural coordinates (ξ, η), shape (n_points, 2)
    weights : np.ndarray
        Corresponding weights (sum to 0.5 for unit triangle)

    Notes
    -----
    These are for reference only; mortar integration is primarily 1D along interfaces.
    Included for completeness if 2D integration over interface is needed.
    """
    if order == 1:
        # 1-point rule (centroid)
        points = np.array([[1.0/3.0, 1.0/3.0]])
        weights = np.array([0.5])

    elif order == 2:
        # 3-point rule (vertices of sub-triangle)
        points = np.array([
            [1.0/6.0, 1.0/6.0],
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0]
        ])
        weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])

    elif order == 3:
        # 4-point rule
        a = 1.0 / 3.0
        b = 0.2
        c = 0.6
        points = np.array([
            [a, a],
            [b, b],
            [c, b],
            [b, c]
        ])
        w1 = -27.0 / 96.0
        w2 = 25.0 / 96.0
        weights = np.array([w1, w2, w2, w2])

    else:
        raise ValueError(f"Unsupported 2D integration order: {order}")

    return points, weights


def generate_interface_integration_points(interface, integration_order: int = 2,
                                         interface_id: int = 0) -> List[IntegrationPoint]:
    """
    Generate integration points along a mortar interface.

    Parameters
    ----------
    interface : MortarInterface
        Interface object with geometric information
    integration_order : int
        Gauss quadrature order (1, 2, or 3)
    interface_id : int
        Interface identifier for tracking

    Returns
    -------
    List[IntegrationPoint]
        Integration points with weights and geometric data

    Notes
    -----
    For a 1D interface of length L, the integral becomes:
        ∫₀ᴸ f(x) dx = ∫₀¹ f(x(ξ)) |dx/dξ| dξ
                    ≈ Σᵢ wᵢ · f(x(ξᵢ)) · L

    Where L is the Jacobian (interface length).
    """
    # Get Gauss points and weights for [0, 1]
    xi_points, weights = gauss_points_1d(integration_order)

    integration_points = []

    for xi, w in zip(xi_points, weights):
        # Get physical position at this parametric coordinate
        position = interface.parametric_to_physical(xi)

        # Weight includes Jacobian (interface length)
        # For 1D integration: w_phys = w_ref * |dx/dξ|
        weighted = w * interface.interface_length

        # Create integration point
        int_point = IntegrationPoint(
            position=position,
            weight=weighted,
            normal=interface.normal.copy(),
            tangent=interface.tangent.copy(),
            parametric_coord=xi,
            interface_id=interface_id
        )

        integration_points.append(int_point)

    return integration_points


def verify_integration_accuracy(order: int, test_func, exact_value: float,
                                tolerance: float = 1e-12) -> bool:
    """
    Verify integration rule accuracy by integrating a test function.

    Parameters
    ----------
    order : int
        Integration order to test
    test_func : callable
        Function to integrate: f(x) for x ∈ [0, 1]
    exact_value : float
        Exact integral value
    tolerance : float
        Acceptable error

    Returns
    -------
    bool
        True if integration error is within tolerance

    Examples
    --------
    >>> # Test with constant function f(x) = 1, exact = 1
    >>> verify_integration_accuracy(2, lambda x: 1.0, 1.0)
    True

    >>> # Test with quadratic f(x) = x², exact = 1/3
    >>> verify_integration_accuracy(2, lambda x: x**2, 1.0/3.0)
    True
    """
    xi_points, weights = gauss_points_1d(order)

    # Compute integral via quadrature
    integral = 0.0
    for xi, w in zip(xi_points, weights):
        integral += w * test_func(xi)

    # Compute error
    error = abs(integral - exact_value)

    return error < tolerance


def test_integration_rules():
    """
    Test integration rules with known integrals.

    Returns
    -------
    bool
        True if all tests pass
    """
    print("\nTesting Gauss quadrature rules:")

    # Test 1: Integrate constant function
    test_passed = verify_integration_accuracy(
        order=1,
        test_func=lambda x: 1.0,
        exact_value=1.0
    )
    print(f"  Order 1, f(x)=1: {'✓' if test_passed else '✗'}")
    assert test_passed, "Order 1 should integrate constants exactly"

    # Test 2: Integrate linear function
    test_passed = verify_integration_accuracy(
        order=1,
        test_func=lambda x: x,
        exact_value=0.5
    )
    print(f"  Order 1, f(x)=x: {'✓' if test_passed else '✗'}")
    assert test_passed, "Order 1 should integrate linear exactly"

    # Test 3: Integrate quadratic function
    test_passed = verify_integration_accuracy(
        order=2,
        test_func=lambda x: x**2,
        exact_value=1.0/3.0
    )
    print(f"  Order 2, f(x)=x²: {'✓' if test_passed else '✗'}")
    assert test_passed, "Order 2 should integrate quadratic exactly"

    # Test 4: Integrate cubic function
    test_passed = verify_integration_accuracy(
        order=2,
        test_func=lambda x: x**3,
        exact_value=1.0/4.0
    )
    print(f"  Order 2, f(x)=x³: {'✓' if test_passed else '✗'}")
    assert test_passed, "Order 2 should integrate cubic exactly"

    # Test 5: Integrate quintic with order 3
    test_passed = verify_integration_accuracy(
        order=3,
        test_func=lambda x: x**5,
        exact_value=1.0/6.0
    )
    print(f"  Order 3, f(x)=x⁵: {'✓' if test_passed else '✗'}")
    assert test_passed, "Order 3 should integrate quintic exactly"

    return True


if __name__ == "__main__":
    # Run tests when module is executed
    print("="*70)
    print("INTEGRATION RULE TESTS")
    print("="*70)

    try:
        test_integration_rules()
        print("\n✓ All integration rule tests passed!")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise

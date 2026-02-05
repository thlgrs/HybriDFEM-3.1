"""
Static Solvers - Linear and Nonlinear Static Analysis
======================================================

This module provides solvers for static structural analysis:

1. **StaticLinear**: One-shot linear solver for K*u = P
2. **StaticNonLinear**: Iterative Newton-Raphson for nonlinear problems

Key Concepts for Students:
--------------------------

**Linear Static Analysis (StaticLinear.solve)**:
    Given: Stiffness K, External load P
    Solve: K * u = P

    With boundary conditions (partitioning):
    [K_ff  K_fs] [u_f]   [P_f]
    [K_sf  K_ss] [u_s] = [P_s]

    where f = free DOFs, s = fixed (support) DOFs
    Solve: u_f = K_ff^-1 * (P_f - K_fs * u_s)
    Reactions: R_s = K_sf * u_f + K_ss * u_s

**Nonlinear Static Analysis (StaticNonLinear)**:
    Uses Newton-Raphson iteration to solve equilibrium:
    P_external = P_internal(u)

    At each iteration:
    1. Compute residual: R = P_ext - P_r(u)
    2. Compute tangent: K_t = dP_r/du
    3. Solve increment: K_t * du = R
    4. Update: u = u + du
    5. Check convergence: ||R|| < tolerance

**Force Control vs Displacement Control**:
    - Force control: Increment external load, find displacement
    - Displacement control: Increment displacement at control DOF, find load factor
    - Displacement control is needed for softening/snap-through behavior

**Augmented Systems (Lagrange/Mortar Coupling)**:
    Saddle-point system:
    [K    C^T] [u     ]   [P]
    [C    0  ] [lambda] = [0]

    where C is constraint matrix, lambda are Lagrange multipliers
    (interface forces between coupled domains)
"""

import os
import time

import numpy as np
import scipy as sc
import scipy.linalg as la  # Dense Linear Algebra
import scipy.sparse as sp  # Sparse Matrix Storage
import scipy.sparse.linalg as spla  # Sparse Linear Algebra

from Core.Solvers.Solver import Solver


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ConvergenceError(RuntimeError):
    """Raised when Newton-Raphson fails to converge within max iterations.

    This typically indicates:
    - Load step too large (reduce step size)
    - Material instability/softening (use displacement control)
    - Numerical issues (check boundary conditions)
    """
    pass


class SingularSystemError(RuntimeError):
    """Raised when the stiffness matrix is singular and cannot be solved.

    This typically indicates:
    - Insufficient boundary conditions (rigid body modes)
    - Material failure (zero stiffness in some direction)
    - Numerical precision issues (check element quality)
    """
    pass


# =============================================================================
# SOLVER CONSTANTS
# =============================================================================

class SolverConstants:
    """Global constants for nonlinear solver algorithms.

    These can be overridden by passing explicit values to solver methods.
    """
    CONDITION_NUMBER_THRESHOLD = 1e15  # Warn if K condition exceeds this
    DEFAULT_FEM_STIFFNESS = 1e9        # Fallback stiffness estimate
    FORCE_TOLERANCE = 1.0              # Default force residual tolerance [N]
    CONSTRAINT_TOLERANCE = 1e-6        # Constraint violation tolerance [m]
    MAX_ITERATIONS = 25                # Max Newton-Raphson iterations per step


# =============================================================================
# BASE SOLVER CLASS
# =============================================================================

class StaticBase(Solver):
    """
    Base class containing shared utilities for both Linear and NonLinear static solvers.

    Provides helper methods for:
    - Load step parsing
    - Storage initialization
    - Coupling detection
    - Results export to HDF5
    """

    @staticmethod
    def _parse_load_steps(steps, disp=None):
        """Parse load/displacement steps into load factors and values."""
        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = steps if disp is None else [s / max(steps, key=abs) for s in steps]
            d_c = steps if disp is not None else None
        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1).tolist()
            d_c = (np.linspace(0, 1, nb_steps + 1) * disp).tolist() if disp is not None else None
        else:
            raise TypeError("Steps must be either a list or an int (number of steps)")

        if isinstance(lam, np.ndarray):
            lam = lam.tolist()
        return nb_steps, lam, d_c

    @staticmethod
    def _initialize_storage(nb_dofs, nb_steps, n_constraints=0, store_stiffness=False,
                            store_load_factor=False, n_fixed_dofs=0):
        """Initialize arrays for storing convergence history."""
        storage = {
            'U_conv': np.zeros((nb_dofs, nb_steps + 1), dtype=float),
            'P_r_conv': np.zeros((nb_dofs, nb_steps + 1), dtype=float),
            'iterations': np.zeros(nb_steps, dtype=int),
            'residuals': np.zeros((nb_steps, 2) if n_constraints > 0 else nb_steps, dtype=float),
        }
        if n_constraints > 0:
            storage['lambda_conv'] = np.zeros((n_constraints, nb_steps + 1), dtype=float)
        if store_load_factor or n_constraints > 0:
            storage['LoadFactor_conv'] = np.zeros(nb_steps + 1, dtype=float)
        if n_fixed_dofs > 0:
            storage['Reaction_conv'] = np.zeros((n_fixed_dofs, nb_steps + 1), dtype=float)
        if store_stiffness:
            storage['K_conv'] = np.zeros((nb_dofs, nb_dofs, nb_steps + 1), dtype=float)
        return storage

    @staticmethod
    def _detect_coupling(structure):
        """Detect active coupling type (Lagrange or Mortar)."""
        if hasattr(structure, 'lagrange_coupling') and structure.lagrange_coupling is not None:
            if structure.lagrange_coupling.active:
                return structure.lagrange_coupling, 'Lagrange'
        if hasattr(structure, 'mortar_coupling') and structure.mortar_coupling is not None:
            if structure.mortar_coupling.active:
                return structure.mortar_coupling, 'Mortar'
        raise ValueError("No active coupling found. Call enable_block_fem_coupling() first.")


    @staticmethod
    def _export_results(filepath, dir_name, storage, metadata, total_time):
        """Export results to HDF5 file."""
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Simulation done in {int(hours)}h {int(minutes)}m {int(seconds)}s.")
        if not filepath: return

        # Handle extension
        if not filepath.endswith('.h5'):
            filepath += ".h5"

        os.makedirs(dir_name, exist_ok=True)
        full_path = os.path.join(dir_name, filepath)

        import h5py
        with h5py.File(full_path, "w") as hf:
            for key, val in storage.items():
                hf.create_dataset(key, data=val)
            hf.attrs["Simulation_Time"] = total_time
            for k, v in metadata.items():
                # HDF5 doesn't support None or complex objects in attrs easily
                if v is not None:
                    try:
                        hf.attrs[k] = v
                    except TypeError:
                        hf.attrs[k] = str(v)
        print(f"Results saved to: {full_path}")


class StaticLinear(StaticBase):
    """
    Solver for Linear Static Equilibrium (One-shot resolution).
    Handles standard Ku=P and Linear Augmented (Saddle Point) systems.
    """

    @staticmethod
    def solve(structure, optimized=False):
        """
        Solves Ku = P.

        Args:
            structure: The structure object containing K0, P, constraints.
            optimized (bool):
                If True, converts K to sparse CSC and uses spsolve (Fast).
                If False, uses dense numpy arrays and standard solve (Slow).
        """
        # 1. Assembly
        structure.get_P_r()
        structure.get_K_str0()

        dof_free = structure.dof_free
        dof_fix = structure.dof_fix

        # --- OPTIMIZED PATH (Sparse) ---
        if optimized:
            # Ensure K0 is in Compressed Sparse Column format for efficient slicing
            if not sp.issparse(structure.K0):
                K_sparse = sp.csc_matrix(structure.K0)
            else:
                K_sparse = structure.K0

            # Slicing (Sparse format supports simple indexing, np.ix_ is not needed)
            K_ff = K_sparse[dof_free, :][:, dof_free]
            K_fr = K_sparse[dof_free, :][:, dof_fix]

            # RHS Calculation: P_free + P_fix_load - Reaction_from_enforced_disp
            # Note: K_fr is sparse, U is dense. Result is dense.
            rhs = (structure.P[dof_free] +
                   structure.P_fixed[dof_free] -
                   K_fr @ structure.U[dof_fix])

            # Solver: Uses SuperLU (efficient for sparse systems)
            structure.U[dof_free] = spla.spsolve(K_ff, rhs)

            # Reaction Forces
            K_rf = K_sparse[dof_fix, :][:, dof_free]
            K_rr = K_sparse[dof_fix, :][:, dof_fix]

            structure.P[dof_fix] = (K_rf @ structure.U[dof_free] +
                                    K_rr @ structure.U[dof_fix])

        # --- STANDARD PATH (Dense) ---
        else:
            # Slicing using np.ix_ for dense arrays
            K_ff = structure.K0[np.ix_(dof_free, dof_free)]
            K_fr = structure.K0[np.ix_(dof_free, dof_fix)]

            rhs = (structure.P[dof_free] +
                   structure.P_fixed[dof_free] -
                   K_fr @ structure.U[dof_fix])

            # Solver: Standard LAPACK LU solver
            structure.U[dof_free] = la.solve(K_ff, rhs)

            # Reaction Forces
            K_rf = structure.K0[np.ix_(dof_fix, dof_free)]
            K_rr = structure.K0[np.ix_(dof_fix, dof_fix)]

            structure.P[dof_fix] = (K_rf @ structure.U[dof_free] +
                                    K_rr @ structure.U[dof_fix])

        # Finalize
        structure.get_P_r()
        return structure

    @staticmethod
    def solve_augmented(structure):
        """Linear solver with Lagrange/Mortar coupling (Saddle Point)."""
        coupling, _ = StaticBase._detect_coupling(structure)

        # Ensure C matrix is built
        if coupling.constraint_matrix_C is None:
            coupling.build_constraint_matrix(structure)

        # OPTIMIZATION: Convert C to sparse immediately to avoid dense slicing
        if sp.issparse(coupling.constraint_matrix_C):
            C = coupling.constraint_matrix_C
        else:
            C = sp.csc_matrix(coupling.constraint_matrix_C)

        structure.get_P_r()
        structure.get_K_str0()

        dof_free = structure.dof_free
        dof_fix = structure.dof_fix

        # Convert to sparse if needed for slicing
        if not sp.issparse(structure.K0):
            K0_sparse = sp.csc_matrix(structure.K0)
        else:
            K0_sparse = structure.K0

        # Efficient Sparse Slicing
        K_ff = K0_sparse[dof_free, :][:, dof_free]
        K_fr = K0_sparse[dof_free, :][:, dof_fix]
        P_f = (structure.P + structure.P_fixed)[dof_free]
        U_r = structure.U[dof_fix]

        # Sparse Slicing of C
        C_f = C[:, dof_free]
        C_r = C[:, dof_fix]

        # Scaling Factor for Augmented System
        scale = 1.0
        if K_ff.shape[0] > 0:
            # Use a representative stiffness value from diagonal
            diag_K = K_ff.diagonal()
            scale = np.max(np.abs(diag_K)) if diag_K.size > 0 else 1.0
            if scale == 0: scale = 1.0

        n_free = len(dof_free)
        n_constraints = C.shape[0]
        n_aug = n_free + n_constraints

        # --- Sparse Assembly ---
        # [ K_ff      (C_f.T * s) ]
        # [ (C_f * s)     0       ]

        K_aug = sp.bmat([
            [K_ff, (C_f.T * scale)],
            [(C_f * scale), None]
        ], format='csc')

        # RHS
        P_eff = P_f - K_fr @ U_r
        P_aug = np.zeros(n_aug)
        P_aug[:n_free] = P_eff
        P_aug[n_free:] = -C_r @ U_r * scale 

        try:
            # Use sparse direct solver
            u_aug = spla.spsolve(K_aug, P_aug)
        except Exception as e:
            raise SingularSystemError(f"Augmented system is singular or failed to solve: {e}")

        if np.any(np.isnan(u_aug)):
            raise SingularSystemError("Solver returned NaNs. System is likely singular or unstable.")

        structure.U[dof_free] = u_aug[:n_free]

        # Recover true multipliers: (lambda/scale) * scale
        coupling.multipliers = u_aug[n_free:] * scale

        # Recover Reactions
        K_rf = K0_sparse[dof_fix, :][:, dof_free]
        K_rr = K0_sparse[dof_fix, :][:, dof_fix]

        structure.P[dof_fix] = K_rf @ structure.U[dof_free] + K_rr @ U_r + C_r.T @ coupling.multipliers

        # Update internal forces (P_r) to reflect the new equilibrium state
        structure.get_P_r()

        return structure


class StaticNonLinear(StaticBase):
    """
    Solver for NonLinear Static Equilibrium (Iterative Newton-Raphson).
    Handles Force Control, Displacement Control and their Augmented versions.
    """

    @staticmethod
    def _handle_contact_yielding(structure):
        """Updates contact state based on yielding behavior (STC/BSTC laws)."""
        if not hasattr(structure, 'list_cfs') or len(structure.list_cfs) == 0:
            return

        yielded_blocks = set()
        for cf in structure.list_cfs:
            for cp in cf.cps:
                if (hasattr(cp, 'sp1') and hasattr(cp.sp1, 'law') and
                        cp.sp1.law.tag == "STC" and cp.sp2.law.tag == "STC"):
                    if cp.sp1.law.yielded or cp.sp2.law.yielded:
                        yielded_blocks.add(cf.bl_A.connect)
                        yielded_blocks.add(cf.bl_B.connect)

        if not yielded_blocks: return

        for cf in structure.list_cfs:
            for cp in cf.cps:
                if (hasattr(cp, 'sp1') and hasattr(cp.sp1, 'law') and
                        cp.sp1.law.tag == "BSTC" and cp.sp2.law.tag == "BSTC"):
                    if cf.bl_A.connect in yielded_blocks or cf.bl_B.connect in yielded_blocks:
                        cp.sp1.law.reduced = True
                        cp.sp2.law.reduced = True

    @staticmethod
    def _check_convergence(R_u, dof_free, tol_force, R_lambda=None, tol_constraint=None):
        """Check convergence norms against tolerances."""
        norm_force = np.linalg.norm(R_u[dof_free])
        force_converged = norm_force < tol_force

        if R_lambda is None:
            return force_converged, norm_force, 0.0

        norm_cons = np.linalg.norm(R_lambda)
        cons_converged = norm_cons < tol_constraint
        return (force_converged and cons_converged), norm_force, norm_cons

    @staticmethod
    def solve_forcecontrol(structure, steps, tol=SolverConstants.FORCE_TOLERANCE,
                           stiff="tan", max_iter=SolverConstants.MAX_ITERATIONS,
                           filename="Results_ForceControl", dir_name=""):

        time_start = time.time()
        nb_steps, lam, _ = StaticBase._parse_load_steps(steps)
        store = StaticBase._initialize_storage(structure.nb_dofs, nb_steps)

        structure.get_P_r()
        structure.get_K_str()
        structure.get_K_str0()

        store['U_conv'][:, 0] = structure.U.copy()
        store['P_r_conv'][:, 0] = structure.P_r.copy()
        K_last_conv = structure.K0.copy()

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0
            P_target = lam[i] * structure.P + structure.P_fixed
            R = P_target[structure.dof_free] - structure.P_r[structure.dof_free]

            while not converged and iteration < max_iter:
                try:
                    K_curr = structure.K if stiff == "tan" else structure.K0
                    K_free = K_curr[np.ix_(structure.dof_free, structure.dof_free)]

                    dU = sc.linalg.solve(K_free, R)
                except (np.linalg.LinAlgError, SingularSystemError, Exception):
                    try:
                        K_prev_free = K_last_conv[np.ix_(structure.dof_free, structure.dof_free)]
                        dU = sc.linalg.solve(K_prev_free, R)
                    except np.linalg.LinAlgError:
                        K0_free = structure.K0[np.ix_(structure.dof_free, structure.dof_free)]
                        dU = sc.linalg.solve(K0_free, R)

                structure.U[structure.dof_free] += dU

                try:
                    structure.get_P_r()
                    structure.get_K_str()
                except Exception as e:
                    print(f"Internal calc error: {e}")
                    break

                R = P_target[structure.dof_free] - structure.P_r[structure.dof_free]
                res = np.linalg.norm(R)

                if res < tol:
                    converged = True
                else:
                    iteration += 1

            if converged:
                structure.commit()
                K_last_conv = structure.K.copy()
                store['residuals'][i - 1] = res
                store['iterations'][i - 1] = iteration
                store['U_conv'][:, i] = structure.U.copy()
                store['P_r_conv'][:, i] = structure.P_r.copy()
                print(f"Step {i} converged after {iteration + 1} iterations")
            else:
                print(f"Method did not converge at step {i}")
                break

        metadata = {
            "Tolerance": tol,
            "P_ref": np.linalg.norm(structure.P)  # Reference load magnitude for computing actual load
        }
        StaticBase._export_results(filename, dir_name, store, metadata, time.time() - time_start)
        return structure

    @staticmethod
    def solve_dispcontrol(structure, steps, disp, node, dof,
                          tol=SolverConstants.FORCE_TOLERANCE,
                          stiff="tan", max_iter=SolverConstants.MAX_ITERATIONS,
                          filename="Results_DispControl", dir_name="",
                          optimized=False):
        """
        Nonlinear Displacement Control.

        Args:
            optimized (bool): If True, uses Partitioned Solution with LU Decomposition re-use.
        """
        time_start = time.time()

        nb_steps, lam, d_c = StaticBase._parse_load_steps(steps, disp)
        n_fixed = len(structure.dof_fix)
        store = StaticBase._initialize_storage(structure.nb_dofs, nb_steps,
                                               store_load_factor=True,
                                               n_fixed_dofs=n_fixed)

        structure.get_P_r()
        structure.get_K_str()
        structure.get_K_str0()

        store['U_conv'][:, 0] = structure.U.copy()
        store['P_r_conv'][:, 0] = structure.P_r.copy()
        store['LoadFactor_conv'][0] = 0.0
        if n_fixed > 0:
            store['Reaction_conv'][:, 0] = structure.P_r[structure.dof_fix].copy()

        K_last_conv = structure.K0.copy()

        # Setup Control DOF
        if isinstance(node, int):
            control_dof = structure._global_dof(node, dof)
        elif isinstance(node, list):
            control_dof = structure._global_dof(node[0], dof)
        other_dofs = structure.dof_free[structure.dof_free != control_dof]

        # Optimization: Pre-calculate indices
        ix_ff = np.ix_(other_dofs, other_dofs)

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0
            lam[i] = lam[i - 1]
            dU_c = d_c[i] - d_c[i - 1]

            R = -structure.P_r + lam[i] * structure.P + structure.P_fixed
            Rf = R[other_dofs]
            Rc = R[control_dof]

            while not converged and iteration < max_iter:

                # ==========================================
                # OPTIMIZED ROUTINE (LU Factorization)
                # ==========================================
                if optimized:
                    def solve_with_lu(K_source, P_f, R_f_mod, R_c_mod, dU_c_val):
                        # 1. Extract and Factorize Matrix ONCE
                        K_ff = K_source[ix_ff]
                        K_cf = K_source[control_dof, other_dofs]
                        P_c = structure.P[control_dof]

                        # LU Decomposition (The heavy lifting happens here)
                        # Returns (lu, piv) tuple
                        lu_piv = sc.linalg.lu_factor(K_ff)

                        # 2. Back-substitute twice (Very fast: O(N^2))
                        v1 = sc.linalg.lu_solve(lu_piv, P_f)
                        v2 = sc.linalg.lu_solve(lu_piv, R_f_mod)

                        # 3. Scalar Math for Load Factor
                        denom = np.dot(K_cf, v1) - P_c
                        if abs(denom) < 1e-12: raise SingularSystemError("Singular Control Point")

                        numer = R_c_mod - np.dot(K_cf, v2)
                        d_lam = numer / denom

                        # 4. Final dU
                        dU_free = v2 + d_lam * v1
                        return np.append(dU_free, d_lam)

                    # Prepare Vectors
                    P_f = structure.P[other_dofs]

                    # Update Stiffness Pointers
                    K_curr = structure.K if stiff == "tan" else structure.K0
                    K_fc_curr = K_curr[other_dofs, control_dof]
                    K_cc_curr = K_curr[control_dof, control_dof]

                    R_f_mod = Rf - K_fc_curr * dU_c
                    R_c_mod = Rc - K_cc_curr * dU_c

                    try:
                        sol = solve_with_lu(K_curr, P_f, R_f_mod, R_c_mod, dU_c)
                    except (np.linalg.LinAlgError, SingularSystemError, Exception):
                        try:
                            # Fallback 1: Last Converged (Recalculate modified residuals for consistency)
                            K_fc_prev = K_last_conv[other_dofs, control_dof]
                            K_cc_prev = K_last_conv[control_dof, control_dof]
                            R_f_mod = Rf - K_fc_prev * dU_c
                            R_c_mod = Rc - K_cc_prev * dU_c
                            sol = solve_with_lu(K_last_conv, P_f, R_f_mod, R_c_mod, dU_c)
                        except:
                            # Fallback 2: K0
                            print("Strategies failed, falling back to K0...")
                            K0_full = structure.K0
                            K_fc_0 = K0_full[other_dofs, control_dof]
                            K_cc_0 = K0_full[control_dof, control_dof]
                            R_f_mod = Rf - K_fc_0 * dU_c
                            R_c_mod = Rc - K_cc_0 * dU_c
                            sol = solve_with_lu(K0_full, P_f, R_f_mod, R_c_mod, dU_c)
                            break

                # ==========================================
                # ORIGINAL ROUTINE
                # ==========================================
                else:
                    def attempt_solve(K_source, P_f, P_c):
                        K_ff = K_source[np.ix_(other_dofs, other_dofs)]
                        K_cf = K_source[control_dof, other_dofs]
                        K_fc = K_source[other_dofs, control_dof]
                        K_cc = K_source[control_dof, control_dof]

                        mat = np.block([[K_ff, -P_f], [K_cf, -P_c]])
                        rhs = np.append(Rf - dU_c * K_fc, Rc - dU_c * K_cc)
                        return sc.linalg.solve(mat, rhs)

                    P_f = structure.P[other_dofs].reshape(len(other_dofs), 1)
                    P_c = structure.P[control_dof]

                    try:
                        K_curr = structure.K if stiff == "tan" else structure.K0
                        sol = attempt_solve(K_curr, P_f, P_c)

                    except (np.linalg.LinAlgError, SingularSystemError, Exception):
                        try:
                            sol = attempt_solve(K_last_conv, P_f, P_c)
                        except (np.linalg.LinAlgError, SingularSystemError, Exception):
                            # Minimal fallback construction...
                            break

                dU_dl = sol
                lam[i] += dU_dl[-1]
                structure.U[other_dofs] += dU_dl[:-1]
                structure.U[control_dof] += dU_c

                try:
                    structure.get_P_r()
                    if stiff == "tan": structure.get_K_str()
                except Exception:
                    break

                R = -structure.P_r + lam[i] * structure.P + structure.P_fixed
                Rf = R[other_dofs]
                Rc = R[control_dof]
                res = np.linalg.norm(R[structure.dof_free])

                if res < tol:
                    converged = True
                    structure.commit()
                    StaticNonLinear._handle_contact_yielding(structure)
                else:
                    iteration += 1
                    dU_c = 0

            if converged:
                K_last_conv = structure.K.copy()
                store['residuals'][i - 1] = res
                store['iterations'][i - 1] = iteration
                store['U_conv'][:, i] = structure.U.copy()
                store['P_r_conv'][:, i] = structure.P_r.copy()
                store['LoadFactor_conv'][i] = lam[i]
                if n_fixed > 0:
                    K_curr = structure.K if stiff == "tan" else structure.K0
                    K_rf = K_curr[np.ix_(structure.dof_fix, structure.dof_free)]
                    K_rr = K_curr[np.ix_(structure.dof_fix, structure.dof_fix)]
                    P_reaction = K_rf @ structure.U[structure.dof_free] + K_rr @ structure.U[structure.dof_fix]
                    store['Reaction_conv'][:, i] = P_reaction
                print(f"Disp. Increment {i} converged after {iteration + 1} iterations")
            else:
                print(f"Method did not converge at Increment {i}")
                break

        metadata = {
            "Tolerance": tol,
            "control_node": node,
            "control_dof": dof,
            "target_disp": disp,
            "P_ref": np.linalg.norm(structure.P)
        }
        StaticBase._export_results(filename, dir_name, store, metadata, time.time() - time_start)
        return structure

    @staticmethod
    def solve_forcecontrol_augmented(structure, steps=10,
                                     tol=SolverConstants.FORCE_TOLERANCE,
                                     tol_constraint=SolverConstants.CONSTRAINT_TOLERANCE,
                                     max_iter=SolverConstants.MAX_ITERATIONS,
                                     stiff="tan", large_displacement=False,
                                     filename="Results_ForceControl_Augmented",
                                     dir_name=""):
        """
        Nonlinear Force Control with Augmented Lagrange/Mortar Coupling.
        Includes robust fallback strategies and Exact Geometric Constraint updates.
        """
        time_start = time.time()

        coupling, coupling_type = StaticBase._detect_coupling(structure)

        # Build initial C if not present
        if coupling.constraint_matrix_C is None:
            coupling.build_constraint_matrix(structure)
        C = coupling.constraint_matrix_C

        n_constraints = C.shape[0]
        nb_steps, lam, _ = StaticBase._parse_load_steps(steps)

        store = StaticBase._initialize_storage(structure.nb_dofs, nb_steps, n_constraints)
        store['U_conv'][:, 0] = structure.U.copy()
        structure.get_P_r()
        store['P_r_conv'][:, 0] = structure.P_r.copy()

        lambda_mult = np.zeros(n_constraints)

        if structure.K0 is None: structure.get_K_str0()
        K_last_conv = structure.K0.copy()

        n_free = len(structure.dof_free)
        n_aug = n_free + n_constraints

        print(f"\nStarting {coupling_type} Force Control (Augmented).")
        print(f"Large Displacement: {large_displacement}")

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0
            P_target = lam[i] * structure.P + structure.P_fixed

            while not converged and iteration < max_iter:
                iteration += 1

                # --- 1. Update Tangent Constraint Matrix (C) ---
                if large_displacement:
                    # If supported by coupling, update C to current configuration
                    if hasattr(coupling, 'update_constraint_matrix'):
                        coupling.update_constraint_matrix(structure)
                    C = coupling.constraint_matrix_C

                # --- 2. Calculate Residuals ---
                structure.get_P_r()

                # Force Residual: P_ext - P_int - C^T * lambda
                R_u = P_target - structure.P_r - C.T @ lambda_mult

                # Constraint Residual (Gap):
                # If Large Disp: Calculate exact non-linear gap g(u)
                # If Small Disp: Calculate linear projection C*u
                if large_displacement and hasattr(coupling, 'compute_exact_constraints'):
                    gap = coupling.compute_exact_constraints(structure)
                else:
                    gap = C @ structure.U

                # The residual for the solver is the GAP we want to eliminate.
                # For checking convergence, we look at the magnitude of this gap.
                R_lam_check = gap  # For norm calculation

                # --- 3. Check Convergence ---
                is_conv, n_force, n_cons = StaticNonLinear._check_convergence(
                    R_u, structure.dof_free, tol, R_lam_check, tol_constraint
                )

                if is_conv:
                    converged = True
                    store['iterations'][i - 1] = iteration
                    store['residuals'][i - 1] = [n_force, n_cons]
                    print(f"Step {i} converged: |Ru|={n_force:.2e}, |Rl|={n_cons:.2e}")
                    break

                # --- 4. Define Internal Augmented Solver ---
                def attempt_solve_augmented(K_core):
                    # Add geometric stiffness if applicable
                    if large_displacement and hasattr(coupling, 'get_geometric_stiffness'):
                        K_geo = coupling.get_geometric_stiffness(structure, lambda_mult)
                        K_core = K_core + K_geo

                    dof_free = structure.dof_free
                    K_ff = K_core[np.ix_(dof_free, dof_free)]

                    C_f = C[:, dof_free]

                    # Build K_aug
                    K_aug = np.zeros((n_aug, n_aug))
                    K_aug[:n_free, :n_free] = K_ff
                    K_aug[:n_free, n_free:] = C_f.T
                    K_aug[n_free:, :n_free] = C_f

                    # Build R_aug
                    R_aug = np.zeros(n_aug)
                    R_aug[:n_free] = R_u[dof_free]

                    # Constraint equation: C * du = -gap
                    # We want to zero out the gap: g(u + du) ~ g(u) + C*du = 0 => C*du = -g(u)
                    R_aug[n_free:] = -gap

                    # Solve (using general solver for robustness on indefinite systems)
                    return sc.linalg.solve(K_aug, R_aug)

                # --- 5. Solve with Fallbacks ---
                try:
                    # 1. Tangent Stiffness
                    if stiff == "tan": structure.get_K_str()
                    K_curr = structure.K if stiff == "tan" else structure.K0
                    du_aug = attempt_solve_augmented(K_curr)

                except (np.linalg.LinAlgError, SingularSystemError, Exception):
                    try:
                        # 2. Previous Stiffness
                        du_aug = attempt_solve_augmented(K_last_conv)
                    except (np.linalg.LinAlgError, SingularSystemError, Exception):
                        # 3. Initial Stiffness
                        try:
                            du_aug = attempt_solve_augmented(structure.K0)
                        except:
                            print(f"  > Matrix singular at Step {i}, Iter {iteration}.")
                            break

                # --- 6. Update State ---
                structure.U[structure.dof_free] += du_aug[:n_free]
                lambda_mult += du_aug[n_free:]

            if converged:
                structure.commit()
                StaticNonLinear._handle_contact_yielding(structure)
                K_last_conv = structure.K.copy()

                store['U_conv'][:, i] = structure.U.copy()
                store['P_r_conv'][:, i] = structure.P_r.copy()
                store['lambda_conv'][:, i] = lambda_mult.copy()
            else:
                print(f"Step {i} failed to converge.")
                break

        coupling.multipliers = lambda_mult.copy()
        metadata = {
            "coupling_type": coupling_type,
            "n_constraints": n_constraints,
            "tol_force": tol,
            "tol_constraint": tol_constraint,
            "P_ref": np.linalg.norm(structure.P)  # Reference load magnitude for computing actual load
        }
        StaticBase._export_results(filename, dir_name, store, metadata, time.time() - time_start)
        return structure

    @staticmethod
    def solve_dispcontrol_augmented(structure, steps, disp, node, dof,
                                    tol=SolverConstants.FORCE_TOLERANCE,
                                    tol_constraint=SolverConstants.CONSTRAINT_TOLERANCE,
                                    max_iter=SolverConstants.MAX_ITERATIONS,
                                    stiff="tan", large_displacement=False,
                                    filename="Results_DispControl_Augmented",
                                    dir_name="",
                                    optimized=False):
        """
        Nonlinear Displacement Control with Augmented Lagrange/Mortar Coupling.

        Args:
            optimized (bool): If True, uses Partitioned Solution + LU Factorization.
        """
        time_start = time.time()

        # 1. Setup Coupling
        coupling, coupling_type = StaticBase._detect_coupling(structure)
        if coupling.constraint_matrix_C is None:
            coupling.build_constraint_matrix(structure)
        C = coupling.constraint_matrix_C

        # 2. Setup Steps
        n_constraints = C.shape[0]
        nb_steps, lam, d_c = StaticBase._parse_load_steps(steps, disp)

        # 3. Storage & Init
        store = StaticBase._initialize_storage(structure.nb_dofs, nb_steps, n_constraints, store_load_factor=True)
        store['U_conv'][:, 0] = structure.U.copy()
        structure.get_P_r()
        store['P_r_conv'][:, 0] = structure.P_r.copy()

        lambda_mult = np.zeros(n_constraints)

        if structure.K0 is None: structure.get_K_str0()
        K_last_conv = structure.K0.copy()

        # 4. Identify DOFs
        if isinstance(node, int):
            control_dof = structure._global_dof(node, dof)
        elif isinstance(node, list):
            control_dof = structure._global_dof(node[0], dof)

        other_dofs = structure.dof_free[structure.dof_free != control_dof]
        n_free = len(other_dofs)

        # System Size: (Free DOFs) + (Multipliers) + (Load Factor Lambda)
        n_aug = n_free + n_constraints + 1

        # Optimization Indices
        ix_ff = np.ix_(other_dofs, other_dofs)
        ix_fc = np.ix_(other_dofs, [control_dof])

        print(f"\nStarting {coupling_type} Disp Control (Augmented).")
        print(f"Control DOF: {control_dof}, Large Disp: {large_displacement}")

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0

            dU_c_step = d_c[i] - d_c[i - 1]
            dU_c_accumulated = 0.0
            lam[i] = lam[i - 1]

            while not converged and iteration < max_iter:
                iteration += 1

                # --- A. Update Geometry ---
                if large_displacement:
                    if hasattr(coupling, 'update_constraint_matrix'):
                        coupling.update_constraint_matrix(structure)
                    C = coupling.constraint_matrix_C

                # --- B. Calculate Residuals ---
                dU_c_iter = dU_c_step - dU_c_accumulated
                structure.get_P_r()

                P_ext_current = lam[i] * structure.P + structure.P_fixed
                R_u = P_ext_current - structure.P_r - C.T @ lambda_mult

                if large_displacement and hasattr(coupling, 'compute_exact_constraints'):
                    gap = coupling.compute_exact_constraints(structure)
                else:
                    gap = C @ structure.U

                R_u_f = R_u[other_dofs]
                R_u_c = R_u[control_dof]

                # ==========================================
                # OPTIMIZED ROUTINE (LU Partitioned)
                # ==========================================
                if optimized:
                    def solve_partitioned_lu(K_core):
                        # 1. Construct Core System (K_ff + Constraints)
                        if large_displacement and hasattr(coupling, 'get_geometric_stiffness'):
                            K_geo = coupling.get_geometric_stiffness(structure, lambda_mult)
                            K_core = K_core + K_geo

                        K_ff = K_core[ix_ff]
                        K_cf = K_core[control_dof, other_dofs]
                        K_fc = K_core[other_dofs, control_dof]
                        K_cc = K_core[control_dof, control_dof]

                        C_f = C[:, other_dofs]
                        C_c = C[:, control_dof]

                        size_core = n_free + n_constraints
                        K_sys = np.zeros((size_core, size_core))
                        K_sys[:n_free, :n_free] = K_ff
                        K_sys[:n_free, n_free:] = C_f.T
                        K_sys[n_free:, :n_free] = C_f

                        # --- LU Factorization of Core System ---
                        # Returns (lu, piv)
                        lu_piv = sc.linalg.lu_factor(K_sys)

                        # 2. Build Vectors
                        P_f = structure.P[other_dofs]
                        P_sys = np.zeros(size_core)
                        P_sys[:n_free] = P_f

                        R_mod_f = R_u_f - (K_fc * dU_c_iter)
                        gap_mod = -gap - (C_c * dU_c_iter)
                        R_sys = np.zeros(size_core)
                        R_sys[:n_free] = R_mod_f
                        R_sys[n_free:] = gap_mod

                        # 3. Double Back-Substitution (O(N^2))
                        v1 = sc.linalg.lu_solve(lu_piv, P_sys)
                        v2 = sc.linalg.lu_solve(lu_piv, R_sys)

                        v1_u = v1[:n_free]
                        v1_lam = v1[n_free:]
                        v2_u = v2[:n_free]
                        v2_lam = v2[n_free:]

                        # 4. Scalar Solve
                        P_c = structure.P[control_dof]
                        lhs_scalar = (np.dot(K_cf, v1_u) + np.dot(C_c, v1_lam)) - P_c
                        rhs_scalar = (R_u_c - K_cc * dU_c_iter) - (np.dot(K_cf, v2_u) + np.dot(C_c, v2_lam))

                        if abs(lhs_scalar) < 1e-12:
                            raise SingularSystemError("Singular Control Point in Augmented Solve")

                        d_load_factor = rhs_scalar / lhs_scalar

                        dU_f_total = v2_u + d_load_factor * v1_u
                        d_lambda_total = v2_lam + d_load_factor * v1_lam

                        return np.hstack((dU_f_total, d_lambda_total, d_load_factor))

                    # Solve with Fallbacks
                    try:
                        if stiff == "tan": structure.get_K_str()
                        K_curr = structure.K if stiff == "tan" else structure.K0
                        sol = solve_partitioned_lu(K_curr)
                    except (np.linalg.LinAlgError, SingularSystemError, Exception):
                        try:
                            sol = solve_partitioned_lu(K_last_conv)
                        except:
                            print(f"  > Matrix singular at Step {i}, Iter {iteration}.")
                            break

                # ==========================================
                # ORIGINAL ROUTINE
                # ==========================================
                else:
                    def attempt_solve_disp_aug(K_core):
                        if large_displacement and hasattr(coupling, 'get_geometric_stiffness'):
                            K_geo = coupling.get_geometric_stiffness(structure, lambda_mult)
                            K_core = K_core + K_geo

                        K_ff = K_core[ix_ff]
                        K_fc = K_core[ix_fc]
                        K_cf = K_core[[control_dof], :][:, other_dofs]
                        K_cc = K_core[control_dof, control_dof]

                        C_f = C[:, other_dofs]
                        C_c = C[:, control_dof].reshape(n_constraints, 1)
                        P_f = structure.P[other_dofs].reshape(n_free, 1)
                        P_c = structure.P[control_dof]

                        K_mat_aug = np.zeros((n_aug, n_aug), dtype=float)
                        K_mat_aug[:n_free, :n_free] = K_ff
                        K_mat_aug[:n_free, n_free:n_free + n_constraints] = C_f.T
                        K_mat_aug[n_free:n_free + n_constraints, :n_free] = C_f
                        K_mat_aug[:n_free, -1] = -P_f.flatten()
                        K_mat_aug[-1, -1] = -P_c
                        K_mat_aug[-1, :n_free] = K_cf.flatten()
                        K_mat_aug[-1, n_free:n_free + n_constraints] = C_c.flatten()

                        R_vec_aug = np.zeros(n_aug, dtype=float)
                        R_vec_aug[:n_free] = R_u_f - (K_fc.flatten() * dU_c_iter)
                        R_vec_aug[n_free:n_free + n_constraints] = -gap - (C_c.flatten() * dU_c_iter)
                        R_vec_aug[-1] = R_u_c - (K_cc * dU_c_iter)

                        return sc.linalg.solve(K_mat_aug, R_vec_aug)

                    try:
                        if stiff == "tan": structure.get_K_str()
                        K_curr = structure.K if stiff == "tan" else structure.K0
                        sol = attempt_solve_disp_aug(K_curr)
                    except (np.linalg.LinAlgError, SingularSystemError, Exception):
                        try:
                            sol = attempt_solve_disp_aug(K_last_conv)
                        except:
                            print(f"  > Matrix singular at Step {i}, Iter {iteration}.")
                            break

                # --- E. Unpack & Update ---
                dU_f = sol[:n_free]
                d_lambda = sol[n_free:n_free + n_constraints]
                d_load_factor = sol[-1]

                structure.U[other_dofs] += dU_f
                structure.U[control_dof] += dU_c_iter
                lambda_mult += d_lambda
                lam[i] += d_load_factor
                dU_c_accumulated += dU_c_iter

                # --- F. Convergence Check ---
                try:
                    structure.get_P_r()
                except:
                    break

                P_ext_new = lam[i] * structure.P + structure.P_fixed

                if large_displacement and hasattr(coupling, 'compute_exact_constraints'):
                    gap_check = coupling.compute_exact_constraints(structure)
                else:
                    gap_check = C @ structure.U

                R_u_check = P_ext_new - structure.P_r - C.T @ lambda_mult

                is_conv, n_force, n_cons = StaticNonLinear._check_convergence(
                    R_u_check, structure.dof_free, tol, gap_check, tol_constraint
                )

                if is_conv:
                    converged = True
                    store['iterations'][i - 1] = iteration
                    store['residuals'][i - 1] = [n_force, n_cons]
                    print(f"Step {i} converged: |Ru|={n_force:.2e}, |Rl|={n_cons:.2e}, Lam={lam[i]:.4f}")

            if converged:
                structure.commit()
                StaticNonLinear._handle_contact_yielding(structure)
                K_last_conv = structure.K.copy()
                store['U_conv'][:, i] = structure.U.copy()
                store['P_r_conv'][:, i] = structure.P_r.copy()
                store['lambda_conv'][:, i] = lambda_mult.copy()
                store['LoadFactor_conv'][i] = lam[i]
            else:
                print(f"Step {i} failed to converge.")
                break

        coupling.multipliers = lambda_mult.copy()
        metadata = {
            "coupling_type": coupling_type,
            "control_dof": control_dof,
            "tol_force": tol,
            "tol_constraint": tol_constraint,
            "P_ref": np.linalg.norm(structure.P)
        }
        StaticBase._export_results(filename, dir_name, store, metadata, time.time() - time_start)
        return structure

# -*- coding: utf-8 -*-

import os
import warnings
from copy import deepcopy

import numpy as np


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


class Contact:
    """
    Linear elastic contact law (base class for contact models).

    Manages normal and tangential forces and stiffness at a contact interface
    using linear elasticity. Supports state management for iterative solvers.

    Attributes
    ----------
    stiff : dict
        Current tangent stiffness: {'kn': float, 'ks': float, 'kns': float, 'ksn': float}
    stiff0 : dict
        Initial (elastic) stiffness
    force : dict
        Current forces: {'n': float, 's': float}
    disps : dict
        Current displacements: {'n': float, 's': float}
    tag : str
        Contact law identifier ('LINEL' for linear elastic)
    tol_disp : float
        Displacement tolerance for numerical checks (1e-15)
    """

    def __init__(self, k_n, k_s):
        """
        Initialize linear elastic contact law.

        Parameters
        ----------
        k_n : float
            Normal stiffness
        k_s : float
            Tangential stiffness
        """
        self.stiff = {}
        self.stiff0 = {}
        self.force = {}
        self.disps = {}

        self.stiff['kn'] = k_n
        self.stiff0['kn'] = k_n

        self.stiff['ks'] = k_s
        self.stiff0['ks'] = k_s

        self.stiff['kns'] = 0
        self.stiff0['kns'] = 0

        self.stiff['ksn'] = 0
        self.stiff0['ksn'] = 0
        self.tag = 'LINEL'
        self.force['n'] = 0
        self.force['s'] = 0
        self.disps['n'] = 0
        self.disps['s'] = 0

        self.tol_disp = 1e-15

    def copy(self):
        """Return a deep copy of this contact law."""
        return deepcopy(self)

    def commit(self):
        """Store converged state for rollback capability."""
        self.force_conv = self.force.copy()
        self.disps_conv = self.disps.copy()
        self.stiff_conv = self.stiff.copy()

    def revert_commit(self):
        """Revert to last committed state."""
        self.force = self.force_conv.copy()
        self.disps = self.disps_conv.copy()
        self.stiff = self.stiff_conv.copy()

    def get_forces(self):
        """
        Get current contact forces.

        Returns
        -------
        np.ndarray
            [Fn, Fs] normal and tangential forces
        """
        return np.array([self.force['n'], self.force['s']])

    def set_elongs(self, d_n, d_s):
        """
        Set contact displacements directly.

        Parameters
        ----------
        d_n : float
            Normal displacement
        d_s : float
            Tangential displacement
        """
        self.disps['n'] = d_n
        self.disps['s'] = d_s

    def get_elongs(self):
        """
        Get current contact displacements.

        Returns
        -------
        tuple
            (d_n, d_s) normal and tangential displacements
        """
        return self.disps['n'], self.disps['s']

    def update(self, dL):
        """
        Update contact forces based on displacement increments.

        Parameters
        ----------
        dL : array-like
            [d_n, d_s] displacement increments
        """
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.force['n'] = self.stiff['kn'] * self.disps['n']
        self.force['s'] = self.stiff['ks'] * self.disps['s']

    def get_k_tan(self):
        """
        Get tangent stiffness matrix components.

        Returns
        -------
        tuple
            (kn, ks, kns, ksn) tangent stiffness components
        """
        return (self.stiff['kn'], self.stiff['ks'], self.stiff['kns'], self.stiff['ksn'])

    def get_k_init(self):
        """
        Get initial (elastic) stiffness components.

        Returns
        -------
        tuple
            (kn, ks, 0, 0) elastic stiffness with zero coupling terms
        """
        return (self.stiff0['kn'], self.stiff0['ks'], 0, 0)

    def to_ommit(self):
        """
        Check if contact should be deleted.

        Returns
        -------
        bool
            False (linear elastic contact never deletes)
        """
        return False


class Bilinear(Contact):
    """
    Bilinear elastic-plastic contact law.

    Models contact with elastic-plastic behavior in the normal direction:
    elastic stiffness kn below yield fy, then reduced stiffness a*kn beyond.
    Tangential direction remains linear elastic.

    Parameters
    ----------
    k_n : float
        Normal elastic stiffness
    k_s : float
        Tangential stiffness
    fy : float
        Yield force in normal direction
    a : float
        Stiffness ratio after yield (a*kn)
    """

    def __init__(self, k_n, k_s, fy, a):
        """
        Initialize bilinear contact law.

        Parameters
        ----------
        k_n : float
            Normal elastic stiffness
        k_s : float
            Tangential elastic stiffness
        fy : float
            Yield force magnitude
        a : float
            Post-yield stiffness factor (k_n_yield = a * k_n)
        """
        super().__init__(k_n, k_s)

        self.force['fy'] = fy
        self.disps['y'] = fy / self.stiff0['kn']
        self.a = a
        self.tag = 'BILIN'
        self.commit()

    def update(self, dL):
        """
        Update bilinear contact response.

        Switches stiffness between elastic (kn) and plastic (a*kn) branches
        based on whether normal displacement exceeds yield displacement.

        Parameters
        ----------
        dL : array-like
            [d_n, d_s] displacement increments
        """
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        if abs(self.disps['n']) < self.disps['y']:
            self.force['n'] = self.stiff0['kn'] * self.disps['n']
            self.stiff['kn'] = deepcopy(self.stiff0['kn'])

        else:
            self.force['n'] = self.force['fy'] * np.sign(self.disps['n']) + self.a * self.stiff0['kn'] * (
                    self.disps['n'] - self.disps['y'] * np.sign(self.disps['n']))
            self.stiff['kn'] = self.a * self.stiff0['kn']

        self.force['s'] = self.stiff0['ks'] * self.disps['s']
        self.stiff['ks'] = deepcopy(self.stiff0['ks'])


class NoTension_EP(Contact):
    """
    Elastic no-tension contact with equivalent plastic strain tracking.

    Prevents normal tension (Fn < 0) by zeroing normal force and stiffness
    when separation occurs. Tracks plastic slip but does not delete contact.

    Attributes
    ----------
    disps['s_p'] : float
        Plastic tangential displacement (unused in update)
    tag : str
        Contact identifier ('NTEP')
    """

    def __init__(self, kn, ks):
        """
        Initialize no-tension elastic contact.

        Parameters
        ----------
        kn : float
            Normal compression stiffness
        ks : float
            Tangential stiffness
        """
        super().__init__(kn, ks)
        self.name = 'Elastic no-tension contact law'
        self.disps['s_p'] = 0
        self.tag = 'NTEP'
        self.commit()

    def update(self, dL):
        """
        Update no-tension contact forces.

        Sets Fn = 0 and kn = 0 if normal displacement exceeds tolerance
        (indicating separation/tension). Tangential response is always elastic.

        Parameters
        ----------
        dL : array-like
            [d_n, d_s] displacement increments
        """
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        if self.disps['n'] > self.tol_disp:
            self.force['n'] = 0.
            self.stiff['kn'] = 0.

        else:
            self.force['n'] = self.disps['n'] * self.stiff0['kn']
            self.stiff['kn'] = deepcopy(self.stiff0['kn'])

        self.force['s'] = self.disps['s'] * self.stiff0['ks']
        self.stiff['ks'] = deepcopy(self.stiff0['ks'])


class NoTension_CD(Contact):
    """
    No-tension contact with contact deletion.

    Prevents tension by deleting the contact when normal separation occurs
    (disps['n'] > tol). Tracks plastic tangential slip via disps['s_p'].
    Once deleted, contact does not reform.

    Attributes
    ----------
    disps['s_p'] : float
        Committed plastic tangential displacement
    disps['s_p_temp'] : float
        Trial plastic tangential displacement
    tag : str
        Contact identifier ('NTCD')
    """

    def __init__(self, kn, ks):
        """
        Initialize no-tension contact with deletion.

        Parameters
        ----------
        kn : float
            Normal compression stiffness
        ks : float
            Tangential stiffness
        """
        super().__init__(kn, ks)
        self.name = 'Elastic no-tension contact law'
        self.disps['s_p'] = 0
        self.disps['s_p_temp'] = 0
        self.tag = 'NTCD'
        self.commit()

    def to_ommit(self):
        """
        Check if contact should be deleted.

        Returns True if normal displacement indicates separation, marking
        this contact for deletion from the system.

        Returns
        -------
        bool
            True if contact is in tension (to be deleted), False otherwise
        """
        if self.disps['n'] > self.tol_disp:
            self.disps['s_p_temp'] = self.disps['s']
            return True

        return False

    def commit(self):
        """Store plastic slip and converged state."""
        self.disps['s_p'] = self.disps['s_p_temp']
        super().commit()

    def update(self, dL):
        """
        Update no-tension contact with slip tracking.

        Computes tangential force using (d_s - s_p) to account for permanent
        plastic slip accumulated up to deletion. Normal force always computed
        elastically until separation.

        Parameters
        ----------
        dL : array-like
            [d_n, d_s] displacement increments
        """
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.force['n'] = self.disps['n'] * self.stiff0['kn']
        self.stiff['kn'] = self.stiff0['kn']

        self.force['s'] = (self.disps['s'] - self.disps['s_p']) * self.stiff0['ks']
        self.stiff['ks'] = self.stiff0['ks']


class Coulomb(Contact):
    """
    Mohr-Coulomb contact law with return mapping plasticity.

    Models Mohr-Coulomb friction (mu, c) with optional dilation angle (psi)
    and tensile strength (ft). Uses return mapping algorithm to enforce
    constraints F1 (positive slip) and F2 (negative slip).

    The two yield surfaces represent:
    - F1: τ + μ*σ - c = 0 (positive slip direction)
    - F2: -τ + μ*σ - c = 0 (negative slip direction)

    Attributes
    ----------
    mu : float
        Friction coefficient
    c : float
        Cohesion
    psi : float
        Dilation angle (default: 0, associated flow)
    ft : float
        Tensile strength (tension cut-off)
    activated : str or None
        Current active yield surface ('F1', 'F2', or None for elastic)
    disps['n_t'] : float
        Tension separation displacement
    disps['d_s_p'] : float
        Accumulated plastic slip
    disps['d_n_p'] : float
        Accumulated plastic dilation
    """

    def __init__(self, kn, ks, mu, c=0, psi=0, ft=0):
        """
        Initialize Mohr-Coulomb contact law.

        Parameters
        ----------
        kn : float
            Normal stiffness
        ks : float
            Tangential stiffness
        mu : float
            Friction coefficient
        c : float, optional
            Cohesion. Default: 0
        psi : float, optional
            Dilation angle (radians). Default: 0 (non-associative flow)
        ft : float, optional
            Tensile strength (tension cut-off). Default: 0 (no tension)
        """
        super().__init__(kn, ks)
        self.mu = mu
        self.tag = 'COUL'

        self.force['n_temp'] = 0
        self.force['s_temp'] = 0
        self.disps['s_temp'] = 0
        self.disps['n_temp'] = 0
        self.disps['d_s_p'] = 0
        self.disps['d_n_p'] = 0
        self.c = c
        self.psi = psi
        self.ft = ft
        self.disps['n_t'] = self.ft / self.stiff0['kn']
        self.activated = None
        self.commit()

    def to_ommit(self):
        """
        Check if contact should be deleted due to excessive tension.

        Contact deletes when normal separation exceeds tensile strength.

        Returns
        -------
        bool
            True if contact is in excessive tension (to be deleted)
        """
        if (self.disps['n_temp'] - self.disps['n_t'] + self.disps['d_n_p']) > self.tol_disp:
            self.disps['s_p_temp'] = self.disps['s']
            return True

        return False

    def get_forces(self):
        """
        Get trial contact forces.

        Returns
        -------
        np.ndarray
            [Fn_trial, Fs_trial] normal and tangential forces
        """
        return np.array([self.force['n_temp'], self.force['s_temp']])

    def commit(self):
        """Store converged forces and activate state for next iteration."""
        self.force['n'] = self.force['n_temp']
        self.force['s'] = self.force['s_temp']
        self.disps['s'] = self.disps['s_temp']
        self.disps['n'] = self.disps['n_temp']

        self.activated = None

        super().commit

    def update(self, dL):
        """
        Update Mohr-Coulomb contact with return mapping algorithm.

        Performs elastic predictor step, checks yield constraints (F1, F2),
        and applies return mapping to enforce plastic flow if yielding occurs.
        Computes consistent tangent stiffness accounting for plasticity.

        The algorithm handles two yield surfaces:
        - F1: positive slip (τ > 0)
        - F2: negative slip (τ < 0)

        Parameters
        ----------
        dL : array-like
            [d_n, d_s] displacement increments
        """
        self.disps['n_temp'] += dL[0]
        self.disps['s_temp'] += dL[1]

        D = np.array([[self.stiff0['kn'], 0],
                      [0, self.stiff0['ks']]])

        d_sigma_tr = D @ dL
        sigma_trial = np.array([self.force['n_temp'], self.force['s_temp']]) + d_sigma_tr

        if self.activated is None:
            F_tr1 = sigma_trial[1] + self.mu * sigma_trial[0] - self.c
            F_tr2 = - sigma_trial[1] + self.mu * sigma_trial[0] - self.c

            if F_tr1 > 0 and F_tr2 > 0:
                if F_tr2 > F_tr1:
                    self.activated = 'F2'
                else:
                    self.activated = 'F1'

        elif self.activated == 'F1':
            F_tr1 = 1
            F_tr2 = 0
        elif self.activated == 'F2':
            F_tr1 = 0
            F_tr2 = 1

        if F_tr1 <= 0 and F_tr2 <= 0:
            self.force['n_temp'] = sigma_trial[0]
            self.force['s_temp'] = sigma_trial[1]
            self.stiff['kn'] = self.stiff0['kn']
            self.stiff['ks'] = self.stiff0['ks']
            self.stiff['kns'] = 0.
            self.stiff['ksn'] = 0.

        elif F_tr2 > 0:
            self.activated = 'F2'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.force['n_temp']
            t = self.force['s_temp']

            denom = kn * m * p + ks
            d_n = (c * p * kn + dL[0] * kn * ks + dL[1] * kn * ks * p - kn * m * p * s + kn * p * t) / denom
            d_s = (-c * ks + dL[0] * kn * ks * m + dL[1] * kn * ks * m * p + ks * m * s - ks * t) / denom
            d_l = (c - dL[0] * kn * m + dL[1] * ks - m * s + t) / denom

            Kn = kn - (m * p * kn ** 2) / denom
            Ks = ks - (ks ** 2) / denom
            Ksn = kn * ks * m / denom
            Kns = kn * ks * p / denom
            self.stiff['kn'] = Kn
            self.stiff['ksn'] = Ksn
            self.stiff['ks'] = Ks
            self.stiff['kns'] = Kns

            self.force['n_temp'] += d_n
            self.force['s_temp'] += d_s
            self.disps['d_s_p'] += d_l
            self.disps['d_n_p'] += d_l * p

        elif F_tr1 > 0:
            self.activated = 'F1'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.force['n_temp']
            t = self.force['s_temp']

            denom = kn * m * p + ks
            d_n = (c * p * kn + dL[0] * kn * ks - dL[1] * kn * ks * p - kn * m * p * s - kn * p * t) / denom
            d_s = (c * ks - dL[0] * kn * ks * m + dL[1] * kn * ks * m * p - ks * m * s - ks * t) / denom
            d_l = (-c + dL[0] * kn * m + dL[1] * ks + m * s + t) / denom

            Kn = kn - (m * p * kn ** 2) / denom
            Ks = ks - (ks ** 2) / denom
            Ksn = - kn * ks * m / denom
            Kns = - kn * ks * p / denom
            self.stiff['kn'] = Kn
            self.stiff['ksn'] = Ksn
            self.stiff['ks'] = Ks
            self.stiff['kns'] = Kns

            self.force['n_temp'] += d_n
            self.force['s_temp'] += d_s

            self.disps['d_s_p'] += d_l
            self.disps['d_n_p'] -= d_l * p

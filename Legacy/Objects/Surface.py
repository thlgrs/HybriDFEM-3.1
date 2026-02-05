# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""

import numpy as np
from warnings import warn
import warnings
import os
import matplotlib.pyplot as plt
from copy import deepcopy


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    file_short_name = file_short_name.replace("\\", "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"

warnings.formatwarning = custom_warning_format


class Surface: 
    
    def __init__(self, k_n, k_s):
        self.stiff = {}
        self.stiff0 = {}
        self.stress = {}
        self.disps = {}

        self.tag = 'LINEL'
        self.stiff['kn'] = k_n
        self.stiff0['kn'] = k_n

        self.stiff['ks'] = k_s
        self.stiff0['ks'] = k_s

        self.stiff['ksn'] = 0
        self.stiff0['ksn'] = 0
        self.stiff['kns'] = 0
        self.stiff0['kns'] = 0
    
        self.stress['s'] = 0
        self.stress['t'] = 0 
        self.disps['n'] = 0 
        self.disps['s'] = 0

        self.tol_disp = 1e-30

    def copy(self): 
        
        return deepcopy(self)

    def commit(self):
        self.stress_conv = self.stress.copy()
        self.disps_conv = self.disps.copy()
        self.stiff_conv = self.stiff.copy()

    def revert_commit(self): 
        
        self.stress = self.stress_conv.copy()
        self.disps = self.disps_conv.copy()
        self.stiff = self.stiff_conv.copy()

    def get_forces(self): 
        
        return np.array([self.stress['s'], self.stress['t']])

    def set_elongs(self, d_n, d_s):
        self.disps['n'] = d_n
        self.disps['s'] = d_s

    def update(self, dL): 
        
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.stress['s'] = self.stiff['kn'] * self.disps['n']
        self.stress['t'] = self.stiff['ks'] * self.disps['s']

    def get_k_tan(self): 
        
        return (self.stiff['kn'], self.stiff['ks'], self.stiff['kns'], self.stiff['ksn'])

    def get_k_init(self): 

        return (self.stiff0['kn'], self.stiff0['ks'], 0, 0)

    def to_ommit(self): 
        
        return False


class NoTension_EP(Surface):

    def __init__(self, kn, ks): 
        
        super().__init__(kn, ks)
        self.tag = 'NTEP'
        # Initialize accumulated plastic shear strain and state variables
        self.commit()

    def to_ommit(self): 
        
        return False

    def update(self, dL): 
        
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        if self.disps['n'] > self.tol_disp:
            #     # No tension allowed, set stress and stiffness to zero
            # print('Ommitting')
            self.stress['s'] = 0.
            self.stiff['kn'] = 0.
            # self.stress['t'] = 0.
            # self.stiff['ks'] = 0.
        else: 
            # Elastic behavior for normal stress
            self.stress['s'] = self.disps['n'] * self.stiff0['kn']
            self.stiff['kn'] = self.stiff0['kn']

        self.stress['t'] = (self.disps['s']) * self.stiff0['ks']
        self.stiff['ks'] = self.stiff0['ks']


class NoTension_CD(Surface):

    def __init__(self, kn, ks): 
        
        super().__init__(kn, ks)
        self.disps['s_p'] = 0
        self.disps['s_p_temp'] = 0
        # Initialize accumulated plastic shear strain and state variables
        self.tag = 'NTCD'
        self.commit()

    def to_ommit(self): 
        if self.disps['n'] > self.tol_disp:
            # print('Omitted')

            self.disps['s_p_temp'] = self.disps['s']
            # print('s_p', self.disps['s_p_temp'])
            return True

        return False

    def commit(self): 
        
        # if self.to_ommit(): 
        self.disps['s_p'] = self.disps['s_p_temp']
        super().commit()

    def update(self, dL): 
        
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.stress['s'] = self.disps['n'] * self.stiff0['kn']
        self.stiff['kn'] = self.stiff0['kn']

        self.stress['t'] = (self.disps['s'] - self.disps['s_p']) * self.stiff0['ks']
        self.stiff['ks'] = self.stiff0['ks']


class Coulomb(Surface):

    def __init__(self, kn, ks, mu, c=0, psi=0, ft=0): 
        
        super().__init__(kn, ks)
        self.mu = mu
        self.tag = 'COUL'

        self.stress['s_temp'] = 0
        self.stress['t_temp'] = 0
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
        
        if (self.disps['n_temp'] - self.disps['n_t'] + self.disps['d_n_p']) > self.tol_disp:
            self.disps['s_p_temp'] = self.disps['s']
            # print('Ommiting')
            # print(self.disps['n_temp'])
            return True

        return False

    def get_forces(self): 
        
        return np.array([self.stress['s_temp'], self.stress['t_temp']])

    def commit(self): 
        
        self.stress['s'] = self.stress['s_temp']
        self.stress['t'] = self.stress['t_temp']
        self.disps['s'] = self.disps['s_temp']
        self.disps['n'] = self.disps['n_temp']

        self.activated = None

        super().commit()

    def update(self, dL):

        self.disps['n_temp'] += dL[0]
        self.disps['s_temp'] += dL[1]

        D = np.array([[self.stiff0['kn'], 0],
                      [0, self.stiff0['ks']]])

        d_sigma_tr = D @ dL
        sigma_trial = np.array([self.stress['s_temp'], self.stress['t_temp']]) + d_sigma_tr

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

        if F_tr1 <= 0 and F_tr2 <= 0:  # Elastic step
            # print('Elastic')
            self.stress['s_temp'] = sigma_trial[0]
            self.stress['t_temp'] = sigma_trial[1] 
            self.stiff['kn'] = self.stiff0['kn']
            self.stiff['ks'] = self.stiff0['ks']
            self.stiff['kns'] = 0.
            self.stiff['ksn'] = 0.

        elif F_tr2 > 0:  # Projection on F2
            # print('F2')
            self.activated = 'F2'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.stress['s_temp']
            t = self.stress['t_temp']

            denom = kn * m * p + ks
            d_n = (c * p * kn + dL[0] * kn * ks + dL[1] * kn * ks * p - kn * m * p * s + kn * p * t) / denom
            d_s = (-c * ks + dL[0] * kn * ks * m + dL[1] * kn * ks * m * p + ks * m * s - ks * t) / denom
            d_l = (c - dL[0] * kn * m + dL[1] * ks - m * s + t) / denom
            # d_l_n = 

            Kn = kn - (m * p * kn ** 2) / denom
            Ks = ks - (ks ** 2) / denom
            Ksn = kn * ks * m / denom
            Kns = kn * ks * p / denom
            self.stiff['kn'] = Kn
            self.stiff['ksn'] = Ksn
            self.stiff['ks'] = Ks
            self.stiff['kns'] = Kns

            self.stress['s_temp'] += d_n
            self.stress['t_temp'] += d_s
            self.disps['d_s_p'] += d_l 
            self.disps['d_n_p'] -= d_l * p

        elif F_tr1 > 0: 
            # print('F1')
            self.activated = 'F1'

            c = self.c
            p = self.psi
            m = self.mu
            kn = self.stiff0['kn']
            ks = self.stiff0['ks']
            s = self.stress['s_temp']
            t = self.stress['t_temp']

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

            self.stress['s_temp'] += d_n
            self.stress['t_temp'] += d_s

            self.disps['d_s_p'] += d_l 
            self.disps['d_n_p'] += d_l * p


class bond_slip_tc(Surface):

    def __init__(self, tb0, tb1): 
        
        kn = 1e17
        # ks = 33e9 / (2 * .1)
        ks = 1e12
        # ks = 5e9

        super().__init__(kn, ks)

        self.stress['tb0'] = tb0
        self.stress['tb1'] = tb1

        self.tag = 'BSTC'
        self.disps['s_p'] = 0

        self.reduced = False
        # Initialize accumulated plastic shear strain and state variables
        self.commit()

    def to_ommit(self): 
        
        return False

    def update(self, dL): 
        
        self.disps['n'] += dL[0]
        self.disps['s'] += dL[1]

        self.stress['s'] = self.disps['n'] * self.stiff0['kn']

        s_tr = self.stiff0['ks'] * (self.disps['s'] - self.disps['s_p'])
        f_tr = abs(s_tr) - (self.stress['tb0'])

        # Elastic step
        if f_tr <= 0: 
            self.stress['t'] = s_tr
            self.stiff['ks'] = deepcopy(self.stiff0['ks'])

        # Plastic step
        else: 
            d_g = f_tr / (self.stiff0['ks'])

            self.stress['t'] = (self.stress['tb0']) * np.sign(self.disps['s'])
            self.disps['s_p'] += d_g * np.sign(s_tr)

            self.stiff['ks'] = 0.


class ElastoPlastic(Surface):

    def __init__(self, knn, ktt, knt, ktn, c, phi, psi, ft, Cnn, Css, Cn, fc):

        super().__init__(knn, ktt)
        self.tag = 'ELPL'

        self.stiff['ksn'] = ktn
        self.stiff['kns'] = knt
        self.stiff0['ksn'] = ktn
        self.stiff0['kns'] = knt

        self.D_el = np.array([[knn, knt],
                              [ktn, ktt]])

        self.D_ep = self.D_el.copy()

        self.D_el_inv = np.linalg.inv(self.D_el)

        self.c = c
        self.phi = phi
        self.psi = psi

        self.ft = ft

        self.Cnn = Cnn
        self.Css = Css
        self.Cn = Cn
        self.fc = fc

        self.tol_1 = 1e-10
        self.tol_2 = 1e-10

        self.normg_1v = []
        self.normg_2v = []

        self.n_iter = 40

        self.error = -1

        self.sigma = np.zeros(2)
        self.eps_tot = np.zeros(2)
        self.eps_tot_pl = np.zeros(2)
        self.eps_tot_el = np.zeros(2)

        self.disps['n_t'] = self.ft / self.stiff0['kn']

        self.commit()

    def to_ommit(self):

        # if (self.disps['n'] - self.disps['n_t']  ) > self.tol_disp:
        #     # self.eps_tot_pl[0] = 0
        #     # self.eps_tot_pl[1] = 0
        #     # self.eps_tot_pl[1] = 0
        #     # self.disps['s_p_temp'] = self.disps['s']
        #     # print('Ommiting')
        #     # print(self.disps['n_temp'])
        # return True

        return False

    def plot_failure_dom(self):

        plt.figure(figsize=(5, 5))

        # print(self.phi)
        sig24 = (self.c - (2 * self.Cnn * self.c + self.Cn * self.phi + self.phi * (
                    self.Cn ** 2 - 4 * self.Css * self.Cn * self.c * self.phi - 4 * self.Cnn * self.Css * self.c ** 2 + 4 * self.Css * self.fc ** 2 * self.phi ** 2 + 4 * self.Cnn * self.fc ** 2) ** (
                                       1 / 2)) / (2 * (self.Css * self.phi ** 2 + self.Cnn))) / self.phi;
        # tau24 = (2*self.Cnn*self.c + self.Cn*self.phi + self.phi*(self.Cn**2 - 4*self.Css*self.Cn*self.c*self.phi - 4*self.Cnn*self.Css*self.c**2 + 4*self.Css*self.fc**2*self.phi**2 + 4*self.Cnn*self.fc**2)**(1/2))/(2*(self.Css*self.phi**2 + self.Cnn));

        if self.fc < np.inf:
            sigV = np.linspace(1.3 * sig24, 1.4 * self.ft, 1000)
        else:
            sigV = 1.4 * self.ft

        plt.plot([self.ft, self.ft], [-1.2 * self.c, 1.2 * self.c], color='red', linewidth=1.5)
        plt.plot(sigV, -sigV * self.phi + self.c, color='red', linewidth=1.5)
        plt.plot(sigV, sigV * self.phi - self.c, color='red', linewidth=1.5)

        if self.fc < np.inf:
            sigC = np.linspace(-self.fc, .7 * sig24, 1000)
            tauC = np.sqrt((-self.Cnn * self.sigC ** 2 - self.Cn * self.sigC + self.fc ** 2) / self.Css)
            plt.plot(sigC, tauC, color='red', linewidth=1.5)
            plt.plot(sigC, -tauC, color='red', linewidth=1.5)

        # plt.title('Isocontour Map')
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$\tau$')
        plt.axis('equal')
        plt.grid(True, linewidth=.25)

        plt.tight_layout()
        plt.show()

    def commit(self):
        self.sigma_prev = self.sigma.copy()
        self.eps_tot_prev = self.eps_tot.copy()
        self.eps_pl_prev = self.eps_tot_pl.copy()
        self.eps_el_prev = self.eps_tot_el.copy()

    def revert_commit(self):
        self.sigma = self.sigma_prev.copy()
        self.eps_tot = self.eps_tot_prev.copy()
        self.eps_tot_pl = self.eps_pl_prev.copy()
        self.eps_tot_el = self.eps_el_prev.copy()

    def update(self, d_eps_tot, N_elpl=1):

        # Trial strains
        d_eps_el = d_eps_tot.copy()
        d_eps_pl = np.zeros(2)

        self.disps['n'] += d_eps_tot[0]
        self.disps['s'] += d_eps_tot[1]

        if self.to_ommit():
            self.eps_tot += d_eps_tot
            # All variation in eps is converted into plastic strain
            self.eps_tot_pl = self.eps_tot.copy()
            self.eps_tot_el = np.zeros(2)
            # self.D_ep = self.D_el.copy()
            # self.stiff['kn'] = self.D_ep[0,0]
            # self.stiff['ks'] = self.D_ep[1,1]
            # self.stiff['ksn'] = self.D_ep[1,0]
            # self.stiff['kns'] = self.D_ep[0,1]
            self.commit()

        else:
            # Trial stress
            d_sig_tr = self.D_el @ d_eps_el
            self.sigma = self.sigma_prev + d_sig_tr

            if N_elpl == 0:

                self.Nact = 0

                self.eps_tot = self.eps_tot_prev + d_eps_tot
                self.eps_tot_pl = self.eps_pl_prev + d_eps_pl
                self.eps_tot_el = self.eps_el_prev + d_eps_el

                self.D_ep = self.D_el.copy()
                self.dL = 0.
                self.Active = 0
                self.El_Energy = 0
                self.Pl_Energy = 0
                self.error = 100
                self.nNR = 0

                # print('Elastic increment')

            else:

                # Evaluate yield surfaces
                self.check_surf(0)

                if self.Nact == 0:

                    self.eps_tot = self.eps_tot_prev + d_eps_tot
                    self.eps_tot_pl = self.eps_pl_prev + d_eps_pl
                    self.eps_tot_el = self.eps_el_prev + d_eps_el

                    self.D_ep = self.D_el.copy()
                    self.dL = 0.
                    self.Active = 0
                    self.El_Energy = 0
                    self.Pl_Energy = 0
                    self.error = 100
                    self.nNR = 0

                    # print('Elastic Increment')

                else:

                    # Save iniitally active surfaces
                    Fact_0 = self.Fact.copy()
                    Jact_0 = self.Jact.copy()

                    # Initial conditions
                    self.dL = np.zeros(self.Nact)
                    self.sigma_f = self.sigma.copy()
                    self.Nq = 0
                    self.get_grad()
                    self.get_residue()

                    # Initialization
                    self.Ndrop = 5
                    Normg1 = 1
                    Normg2 = 1
                    self.nNR = 0

                    while self.Ndrop > 0:

                        # print(self.Jact)

                        while Normg1 >= self.tol_1 or Normg2 >= self.tol_2:

                            self.nNR += 1

                            self.get_jac()

                            e_sol = self.JAC_inv @ self.eR

                            self.sigma_f -= e_sol[:2]
                            self.dL -= e_sol[2:]
                            # print(f'd_lambda  is {self.dL}')

                            self.evaluate_surf(sig=self.sigma_f)
                            self.Fact = self.F[self.Jact]

                            self.get_grad()
                            self.get_residue()

                            # print(f'Residue is {self.eR}')

                            # Check tolerances
                            if self.Nq != 0:
                                Normg1 = np.linalg.norm(self.eR)
                            else:
                                Normg1 = np.linalg.norm(self.eR[:2 + self.Nact])

                            Normg2 = np.linalg.norm(self.eR[2:2 + self.Nact])

                            self.normg_1v.append(Normg1)
                            self.normg_2v.append(Normg2)

                            if self.nNR >= self.n_iter:
                                self.error = -1
                                self.sigma = self.sigma_f.copy()
                                self.eps_tot = self.eps_tot_prev + d_eps_tot
                                self.eps_tot_el = self.eps_el_prev + self.D_el_inv @ (self.sigma_f - self.sigma_prev)
                                self.eps_tot_pl = self.eps_tot - self.eps_tot_el
                                d_eps_pl = self.eps_tot_pl - self.eps_pl_prev
                                self.D_ep = 0.
                                self.Active = self.Jact.copy()

                                print('Elastoplastic increment: exit with too many iterations')
                                break

                        self.get_ndrop()

                        if self.Ndrop > 0:
                            self.Nact -= self.Ndrop
                            self.dL = np.zeros(self.Nact)
                            self.sigma_f = self.sigma.copy()
                            eQa = 0;

                            self.check_new_active_surfaces(Fact_0, Jact_0)
                            self.get_grad()
                            self.get_residue()

                            Fact_0 = self.Fact.copy()
                            Jact_0 = self.Jact.copy()

                            Normg1 = 10
                            Normg2 = 10

                    # print(f'Elastoplastic increment converged after {self.nNR} steps')  

                    self.sigma = self.sigma_f.copy()
                    self.eps_tot = self.eps_tot_prev + d_eps_tot
                    self.eps_tot_el = self.eps_el_prev + self.D_el_inv @ (self.sigma_f - self.sigma_prev)
                    self.eps_tot_pl = self.eps_tot - self.eps_tot_el
                    d_eps_pl = self.eps_tot_pl - self.eps_pl_prev

                    # Check plastic strain computation
                    d_eps_pl2 = np.zeros(2)

                    for i in range(self.Nact):
                        d_eps_pl2 += self.dL[i] * self.dGdS[i]

                    if np.linalg.norm(d_eps_pl2 - d_eps_pl) > 1e-14:
                        self.error = 2
                        # print("Error in computation of d_eps_pl")

                    # Check normality rule
                    if self.Nact == 1:
                        det_val = np.cross(d_eps_pl, self.dGdS)
                        # if abs(det_val) < 1e-12: 
                        # print('Normality rule is satisfied')
                        # else: 
                        # print('Normality rule is NOT satisfied')

                    # Check stress orthogonality - to be checked. 

                    if self.nNR == 1:

                        sig_tr = self.sigma_prev + d_sig_tr
                        stress_proj = self.sigma - sig_tr

                        if self.Nact == 1:
                            det_val = np.cross(stress_proj, self.dFdS)

                            # if abs(det_val) < 0.02: 
                            # print('Stress orthogonality is satisfied')
                            # else: 
                            # print('Stress orthogonality is NOT satisfied')

                    self.get_dep()

            self.stiff['kn'] = self.D_ep[0, 0]
            self.stiff['ks'] = self.D_ep[1, 1]
            self.stiff['ksn'] = self.D_ep[1, 0]
            self.stiff['kns'] = self.D_ep[0, 1]

            self.stress['s'] = self.sigma[0]
            self.stress['t'] = self.sigma[1]

            # print(np.around(self.stress['s']))

            self.commit()

    def check_surf(self, Q):

        # dGds
        n0 = np.array([1, 0], dtype=float)
        n1 = np.array([self.psi, 1], dtype=float)
        n2 = np.array([self.psi, -1], dtype=float)

        n0 /= np.linalg.norm(n0)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        def get_n3(s):
            n3 = np.array([self.Cn + 2 * self.Cnn * s[0], 2 * self.Css * s[1]])
            return n3 / np.linalg.norm(n3)

        # Evaluate yield criteria:
        self.evaluate_surf()

        self.Jact = []

        for i in range(4):
            if self.F[i] >= 1e-15:
                self.Jact.append(i)

        self.Nact = len(self.Jact)

        i_drop = []

        for j in self.Jact:

            if j == 0:  # check admissible domain for F1 - Tension
                sig_01 = self.get_surf_intersect(0, 1)
                if np.cross(sig_01, n1) - np.cross(self.sigma, n1) > 0:
                    i_drop.append(j)
                sig_02 = self.get_surf_intersect(0, 2)
                if np.cross(sig_02, n2) - np.cross(self.sigma, n2) < 0:
                    i_drop.append(j)

            elif j == 1:  # check admissible domain for F2
                sig_01 = self.get_surf_intersect(1, 0)
                if np.cross(sig_01, n0) - np.cross(self.sigma, n0) < 0:
                    i_drop.append(j)

                sig_13 = self.get_surf_intersect(1, 3)
                n3 = get_n3(sig_13)
                if - np.cross(self.sigma, n3) + np.cross(sig_13, n3) > 0:
                    i_drop.append(j)

            elif j == 2:
                sig_02 = self.get_surf_intersect(0, 2)
                if - np.cross(self.sigma, n0) + np.cross(sig_02, n0) > 0:
                    i_drop.append(j)

                sig_23 = self.get_surf_intersect(2, 3)
                n3 = get_n3(sig_23)
                if - np.cross(self.sigma, n3) + np.cross(sig_23, n3) < 0:
                    i_drop.append(j)

            elif j == 3:
                if self.sigma[1] > 0:
                    sig_13 = self.get_surf_intersect(1, 3)
                    if np.cross(sig_13, n1) - np.cross(self.sigma, n1) < 0:
                        i_drop.append(j)
                elif self.sigma[1] < 0:
                    sig_23 = self.get_surf_intersect(2, 3)
                    if np.cross(sig_23, n2) - np.cross(self.sigma, n2) > 0:
                        i_drop.append(j)

        self.Jact = [x for x in self.Jact if x not in i_drop]
        self.Fact = [x for i, x in enumerate(self.F) if i in self.Jact]

        self.Nact = len(self.Jact)

    def get_surf_intersect(self, surf1, surf2):

        surfaces = [surf1, surf2]
        surfaces.sort()

        if surfaces == [0, 1]:
            sig = self.ft
            tau = self.c - self.ft * self.phi
            return np.array([sig, tau])

        elif surfaces == [0, 2]:
            sig = self.ft
            tau = - (self.c - self.ft * self.phi)
            return np.array([sig, tau])

        elif surfaces == [0, 3]:
            pass

        elif surfaces == [1, 2]:
            pass

        elif surfaces[1] == 3:
            sig = (self.c - (2 * self.Cnn * self.c + self.Cn * self.phi + self.phi \
                             * (
                                         self.Cn ** 2 - 4 * self.Css * self.Cn * self.c * self.phi - 4 * self.Cnn * self.Css * self.c ** 2 \
                                         + 4 * self.Css * self.fc ** 2 * self.phi ** 2 + 4 * self.Cnn * self.fc ** 2) ** (
                                         1 / 2)) \
                   / (2 * (self.Css * self.phi ** 2 + self.Cnn))) / self.phi
            tau = (2 * self.Cnn * self.c + self.Cn * self.phi + self.phi \
                   * (self.Cn ** 2 - 4 * self.Css * self.Cn * self.c * self.phi \
                      - 4 * self.Cnn * self.Css * self.c ** 2 + 4 * self.Css * self.fc ** 2 * self.phi ** 2 \
                      + 4 * self.Cnn * self.fc ** 2) ** (1 / 2)) / (2 * (self.Css * self.phi ** 2 + self.Cnn))

            if surfaces[0] == 1:
                return np.array([sig, tau])
            elif surfaces[0] == 2:
                return np.array([sig, -tau])

    def evaluate_surf(self, sig=None):

        if sig is not None:
            s = sig[0]
            t = sig[1]
        else:
            s = self.sigma[0]
            t = self.sigma[1]

        self.F = np.ones(4) * -1

        if self.ft > self.c / self.phi:
            self.F[0] = -100
        else:
            self.F[0] = s - self.ft
        self.F[1] = t + self.phi * s - self.c
        self.F[2] = - t + self.phi * s - self.c
        self.F[3] = self.Cnn * s ** 2 + self.Css * t ** 2 + self.Cn * s - self.fc ** 2

    def get_grad(self):

        sig = self.sigma_f[0]
        tau = self.sigma_f[1]

        self.dFdS = np.zeros((4, 2))
        self.dFdS[0] = np.array([1, 0], dtype=float)
        self.dFdS[1] = np.array([self.phi, 1], dtype=float)
        self.dFdS[2] = np.array([self.phi, -1], dtype=float)
        self.dFdS[3] = np.array([self.Cn + 2 * sig * self.Cnn, 2 * self.Css * tau], dtype=float)

        self.dGdS = np.zeros((4, 2))
        self.dGdS[0] = np.array([1, 0], dtype=float)
        self.dGdS[1] = np.array([self.psi, 1], dtype=float)
        self.dGdS[2] = np.array([self.psi, -1], dtype=float)
        self.dGdS[3] = np.array([self.Cn + 2 * sig * self.Cnn, 2 * self.Css * tau], dtype=float)

        self.ddGddSS = np.zeros((4, 2, 2))
        self.ddGddSS[3] = np.array([[2 * self.Cnn, 0],
                                    [0, 2 * self.Css]])
        self.ddGddSQ = np.zeros((2, 1, self.Nact))

        self.dHdQ = np.zeros((2, 2, self.Nact))
        self.dFdQ = np.zeros((2, 2, self.Nact))
        self.ddHddQS = np.zeros((1, 2, self.Nact))
        self.ddHddQQ = np.zeros((1, 1, self.Nact))

        # Only active gradients: 
        self.dFdS = self.dFdS[self.Jact]
        self.dGdS = self.dGdS[self.Jact]
        self.ddGddSS = self.ddGddSS[self.Jact]

    def get_residue(self):

        self.eR = np.zeros(2 + self.Nact + self.Nq)

        for k1 in range(self.Nact):
            self.eR[:2] += self.dL[k1] * self.dGdS[k1]
            self.eR[2 + k1] = self.Fact[k1]

            # if self.Nq != 0: 
            #     self.eR[2+self.Nact+Nq] += self.dL(k1) * self.dHdQ[k1]

        self.eR[:2] += self.D_el_inv @ (self.sigma_f - self.sigma)

    def get_jac(self):

        self.JAC = np.zeros((2 + self.Nact + self.Nq, 2 + self.Nact + self.Nq))
        r1d1 = np.zeros((2, 2))

        for i in range(self.Nact):
            r1d1 += self.dL[i] * self.ddGddSS[i]

        self.JAC[:2, :2] = self.D_el_inv + r1d1
        self.JAC[:2, -self.Nact:] = self.dGdS.T
        self.JAC[-self.Nact:, :2] = self.dFdS

        self.JAC_inv = np.linalg.pinv(self.JAC)

    def get_ndrop(self):

        self.Jdrop = []

        for i, drop in enumerate(self.Jact):
            if self.dL[i] < 0:
                self.Jdrop.append(drop)

        self.Ndrop = len(self.Jdrop)

    def check_new_active_surfaces(self, Fact_0, Jact_0):

        self.Jact = [i for i in Jact_0 if i not in self.Jdrop]
        self.Fact = [Fact_0[i] for i, x in enumerate(Jact_0) if x not in self.Jdrop]

    def get_dep(self):

        eG = np.zeros((self.Nact, self.Nact))
        eH = np.zeros((self.Nact, self.Nact))
        eM = np.zeros((self.Nact, self.Nact))
        eM_inv = np.zeros((self.Nact, self.Nact))
        eCP = np.zeros((2, 2))

        # To be updated in other functions when damage variable is introduced properly
        eKAPPA = np.zeros((2, 2))

        # G Matrix
        for i in range(self.Nact):
            for j in range(self.Nact):
                eG[i, j] = self.dFdS[i, :] @ self.D_el @ self.dGdS[j, :].T

        # H matrix
        # for i in range(self.Nact): 
        #     for j in range(self.Nact):   

        #         eH[i,j] = self.dFdQ[i,:] @ eKAPPA @ self.dHdQ[j,:].T

        # F matrix
        eM = eG + self.Nq * eH
        eM_inv = np.linalg.inv(eM)

        # Multiplications
        eCP = self.D_el @ self.dGdS.T @ eM_inv @ self.dFdS @ self.D_el

        self.D_ep = self.D_el - eCP

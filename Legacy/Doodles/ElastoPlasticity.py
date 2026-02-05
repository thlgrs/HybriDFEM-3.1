# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:59:04 2025

@author: ibouckaert
"""

import numpy as np
import sys


class ElPl_Mat:

    def __init__(self, knn, ktt, knt, ktn, c, phi, psi, ft, Cnn, Css, Cn, fc):

        self.knn = knn
        self.knt = knt
        self.ktn = ktn
        self.ktt = ktt

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

        self.commit()

    def commit(self):
        self.sigma_prev = self.sigma.copy()
        self.eps_tot_prev = self.eps_tot.copy()
        self.eps_pl_prev = self.eps_tot_pl.copy()
        self.eps_el_prev = self.eps_tot_el.copy()

    def update(self, N_elpl, d_eps_tot):

        # Trial strains
        d_eps_el = d_eps_tot.copy()
        d_eps_pl = np.zeros(2)

        # Trial stress
        d_sig_tr = self.D_el @ d_eps_el
        self.sigma = self.sigma_prev + d_sig_tr

        if N_elpl == 0:

            self.Nact = 0

            self.eps_tot = self.eps_tot_prev + d_eps_tot
            self.eps_tot_pl = self.eps_pl_prev + d_eps_pl
            self.eps_tot_el = self.eps_el_prev + d_eps_el

            self.d_l = 0
            self.active = 0
            self.energy_el = 0
            self.energy_pl = 0
            self.error = 0

            print('Elastic increment')

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

                print('Elastic Increment')

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

                print(f'Elastoplastic increment converged after {self.nNR} steps')

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
                    print("Error in computation of d_eps_pl")

                # Check normality rule
                if self.Nact == 1:
                    det_val = np.cross(d_eps_pl, self.dGdS)
                    if abs(det_val) < 1e-12:
                        print('Normality rule is satisfied')
                    else:
                        print('Normality rule is NOT satisfied')

                # Check stress orthogonality - to be checked. 

                if self.nNR == 1:

                    sig_tr = self.sigma_prev + d_sig_tr
                    stress_proj = self.sigma - sig_tr

                    if self.Nact == 1:
                        det_val = np.cross(stress_proj, self.dFdS)

                        if abs(det_val) < 0.02:
                            print('Stress orthogonality is satisfied')
                        else:
                            print('Stress orthogonality is NOT satisfied')

                self.get_dep()

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
            if self.F[i] >= 1e-10:
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

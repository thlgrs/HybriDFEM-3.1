# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:59:50 2025

@author: ibouckaert
"""

import ElastoPlasticity as EP
import numpy as np
import matplotlib.pyplot as plt

knn = 1e9;
knt = 0.;
ktn = 0.;
ktt = 1e9
c = 0.0;
phi = 10;
psi = 0
ft = 0.0
Cnn = 1.0;
Css = 9.0;
Cn = 0.;
fc = 2e16

Mat = EP.ElPl_Mat(knn, ktt, knt, ktn, c, phi, psi, ft, Cnn, Css, Cn, fc)

# sigma = np.array([-1., 0])
d_eps_tot = np.array([0.15, 0])
Mat.update(1, d_eps_tot)
Mat.commit()
print(Mat.sigma)
print(Mat.eps_tot)
print(Mat.eps_tot_el)
print(Mat.eps_tot_pl)
print(Mat.D_ep)

d_eps_tot = np.array([-0.01, 0])
# print(d_eps_tot)

Mat.update(1, d_eps_tot)

print(Mat.sigma)
print(Mat.eps_tot)
print(Mat.eps_tot_el)
print(Mat.eps_tot_pl)
print(Mat.D_ep)
# %% Plotting error maps
# nb_steps = 20

# d_eps_tot_sig = np.linspace(-1.2, 1.5, nb_steps)
# d_eps_tot_tau = np.linspace(-3, 3, nb_steps)

# Sigma_tr_sig = np.zeros(nb_steps)
# Sigma_tr_tau = np.zeros(nb_steps)

# nNR = np.zeros((nb_steps, nb_steps))

# for i in range(nb_steps): 
#     for j in range(nb_steps): 

#         Mat = EP.ElPl_Mat(knn, ktt, knt, ktn, c, phi, psi, ft, Cnn, Css, Cn, fc)

#         sigma = np.array([-1, 0])
#         d_eps_tot = np.array([sigma[0]/knn, sigma[1]/ktt])

#         Mat.update(0, d_eps_tot)
#         Mat.commit()

#         d_eps_tot = np.array([d_eps_tot_sig[i], d_eps_tot_tau[j]])
#         # print('d_eps_tot', d_eps_tot)

#         Mat.update(1, d_eps_tot)

#         nNR[j,i] = Mat.nNR

#         Sigma_tr_sig[i] = sigma[0] + d_eps_tot_sig[i] * knn
#         Sigma_tr_tau[j] = sigma[1] + d_eps_tot_tau[j] * ktt

# #%% Plotting results 
# plt.figure(figsize=(5,5))
# X, Y = np.meshgrid(Sigma_tr_sig, Sigma_tr_tau)

# contours = plt.contourf(X,Y, nNR*1.01, levels=int(np.max(nNR)), cmap='viridis')
# plt.contour(X,Y, nNR, levels=int(np.max(nNR)), colors='black',linewidths=.5)

# sig24 = (c - (2*Cnn*c + Cn*phi + phi*(Cn**2 - 4*Css*Cn*c*phi - 4*Cnn*Css*c**2 + 4*Css*fc**2*phi**2 + 4*Cnn*fc**2)**(1/2))/(2*(Css*phi**2 + Cnn)))/phi;
# tau24 = (2*Cnn*c + Cn*phi + phi*(Cn**2 - 4*Css*Cn*c*phi - 4*Cnn*Css*c**2 + 4*Css*fc**2*phi**2 + 4*Cnn*fc**2)**(1/2))/(2*(Css*phi**2 + Cnn));

# if fc < np.inf: 
#     sigV = np.linspace(1.3*sig24, 1.4*ft, 1000)
# else: 
#     sigV = np.linspace(np.min(Sigma_tr_sig),1.4*ft)    

# plt.clabel(contours, inline=True, fontsize=8)

# plt.plot([ft, ft], [-1.2*c, 1.2*c], color='red', linewidth=1.5)
# plt.plot(sigV, -sigV*phi+c, color='red', linewidth=1.5)
# plt.plot(sigV, sigV*phi-c, color='red', linewidth=1.5)
# plt.plot(sigma[0], sigma[1],marker='o',color='black')

# sigC = np.linspace(-fc, .7*sig24,1000)
# tauC = np.sqrt((-Cnn*sigC**2 - Cn*sigC + fc**2)/Css)
# plt.plot(sigC, tauC, color='red', linewidth=1.5)
# plt.plot(sigC, -tauC, color='red', linewidth=1.5)

# # plt.title('Isocontour Map')
# plt.xlabel(r'$\sigma$')
# plt.ylabel(r'$\tau$')
# plt.axis('equal')
# plt.grid(True, linewidth=.25)

# plt.colorbar(contours, label='Z value')
# plt.tight_layout()
# plt.show()

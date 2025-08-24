#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:22:40 2024

Uncoupled ROM for Rayleigh Benard convection
    - Import ROM coefficients from h5 file
    - Simulate ROM
    - Plot and export snapshots
@author: efloresm
"""

import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
import multiprocessing   as mp
import time              as tm
import h5py              as h5
import os                as os
import shutil            as shutil
import scipy.sparse      as sp
from functools       import partial
from scipy.integrate import solve_ivp
from FUN             import *
from uncoupled.RB_uncoupled_FUN import *
from uncoupled.RB_uncoupled_FUN import ROM_sparse, jac_uncoupled

# =============================================================================
# Import ROM coefficients
# =============================================================================
n_alpha, n  = 8, 12                     # ROM parameters
nx, ny      = 4*(n_alpha-1) + 2, 64     # ROM discretization
nmodes      = n_alpha*n                 # Number of modes in ROM
ndim        = nmodes*2                  # ROM dimension

name        = 'RB_uncoupled_nx{:03.0f}_ny{:03.0f}_mX{:02.0f}_mY{:02.0f}_n{:03.0f}.h5'.format(nx, ny, n_alpha, n, ndim)
path        = 'uncoupled/ROM'
input_file  = os.path.join(path, name)
print('File name = {}'.format(name))
# Import the h5 file
with h5.File(input_file, 'r') as h5file:
    # Read the datasets from the file and store them in variables
    X       = h5file['X'][:]
    Y       = h5file['Y'][:]
    TT0     = h5file['TT0'][:]
    Ubase   = h5file['Ubase'][:]
    Tbase   = h5file['Tbase'][:]
    u_For0  = h5file['u_For0'][:]
    u_For1  = h5file['u_For1'][:]
    u_Diff  = h5file['u_Diff'][:]
    u_Nlin  = h5file['u_Nlin'][:]
    T_Line  = h5file['T_Line'][:]
    T_Diff  = h5file['T_Diff'][:]
    T_Nlin  = h5file['T_Nlin'][:]
print('File successfully read from', input_file)


# Domain size
Lx = 2
Ly = 1
Y0 = np.min(Y)
# Fundamental wavenumber in the x direction
alpha   = 2*np.pi/Lx
beta    = 2*np.pi/Ly

# Wavenumber in the x direction
kx      = alpha*np.arange(n_alpha)
ky      = beta

# Grid points
X       = np.flip(np.linspace(Lx, 0, nx, endpoint = False))
X       = np.linspace(0, Lx, nx, endpoint = False)
Y, W    = clenshaw_curtis_compute(ny, Ly, Y0)
Y, DY   = cheb(ny, Ly, Y0)
DY2     = DY@DY

# 2D grid
xx, yy  = np.meshgrid(X, Y)

# Convert nonlinear terms to sparse
u_Nlin_sparse = sp.bsr_matrix(u_Nlin.reshape(nmodes, -1), blocksize=(n,n))  # Flatten to 2D sparse
T_Nlin_sparse = sp.bsr_matrix(T_Nlin.reshape(nmodes, -1), blocksize=(n,n))  # Flatten to 2D sparse

# Import critical Ra number
_,_,Ra_c = linear_analysis()


#%% =============================================================================
# Simulate ROM with randomly perturbed initial conditions
# =============================================================================

# Define partial functions for time integration
# ROM dynamics
ROM_partial = partial(ROM_sparse, u_For0=u_For0, u_For1=u_For1, u_Diff=u_Diff,
                      u_Nlin =u_Nlin_sparse, T_Line=T_Line, T_Diff=T_Diff, 
                      T_Nlin =T_Nlin_sparse, nmodes=nmodes)
# ROM jacobian
jac_partial = partial(jac_uncoupled,   u_For1 = u_For1, 
                      u_Diff = u_Diff, u_Nlin = u_Nlin, 
                      T_Line = T_Line, T_Diff = T_Diff, 
                      T_Nlin = T_Nlin, nmodes = nmodes)

# Import critical Rayleigh number from linear analysis
_,_, Ra_c = linear_analysis()

# Simulation parameters
Ra = 100*Ra_c
Pr = 10

# Initial conditions
ci0 = np.random.normal(0,0.01, nmodes*2)

# Time array
tend    = 500
nt      = 501
t       = np.linspace(0,tend, nt)

# Integrate ROM equations
out     = solve_ivp(ROM_partial,  [0, tend], 
                    ci0, args   = (Pr, Ra), 
                    t_eval      = t,
                    jac         = jac_partial,
                    method      = 'LSODA')
# Extract solution and time array
ci      = out.y
t       = out.t
# Split velocity and temperature coefficients
ai  = ci[:nmodes]
bi  = ci[nmodes:]
# Reconstruct the velocity and temperature profiles
U       = np.zeros([nt, ny, nx])
V       = np.zeros([nt, ny, nx])
Vor     = np.zeros([nt, ny, nx])
theta0 = np.ones([nt, ny, nx])*np.reshape(TT0, [1, ny, nx])
theta1 = np.zeros([nt, ny, nx])

for i in range(nt):
    for j in range(nmodes):
        U[i]        = U[i]  + Ubase[0,j]*ai[j,i]
        V[i]        = V[i]  + Ubase[1,j]*ai[j,i]
        theta1[i]   = theta1[i] + Tbase[0,j]*bi[j,i]
    _, dUdy = grad(U[i], X, Y, DY)
    dVdx, _ = grad(V[i], X, Y, DY)
    Vor[i]      = dVdx - dUdy
THETA   = theta0 + theta1

#%% =============================================================================
# PLOTTING
# =============================================================================


plt.close('all')

f, ax = plt.subplots(2,1, constrained_layout=True)
ax[0].plot(t, ai.T)
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$a_i$')
ax[0].grid(True)
ax[1].plot(t, bi.T)
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$b_i$')
ax[1].grid(True)
plt.show()

amp     = np.max([np.max(np.abs(ai)), 0.02])
bmp     = np.max([np.max(np.abs(bi)), 0.02])
nh, nv  = 3,4
    
f, ax = plt.subplots(nh,nv, figsize = (8,4), constrained_layout = True)
plt.suptitle('Vorticity field')
for i in range(nh):
    for j in range(nv):
        t_idx = int((4*i + j)/(nh*nv )*nt)
        ax[i,j].set_title(f't = {t[t_idx]}')
        ax[i,j].pcolormesh(xx, yy, Vor[t_idx], vmin = -10*amp, vmax = 10*amp, shading = 'auto', cmap = 'RdBu')
        ax[i,j].axis('equal')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_xlim([0, Lx])
        ax[i,j].set_ylim([0, Ly])  
f, ax = plt.subplots(nh,nv, figsize = (8,4), constrained_layout = True)
plt.suptitle('Temperature field')
for i in range(nh):
    for j in range(nv):
        t_idx = int((4*i + j)/(nh*nv )*nt)
        ax[i,j].set_title(f't = {t[t_idx]}')
        ax[i,j].pcolormesh(xx, yy, THETA[t_idx], vmin = 0, vmax = 1, shading = 'auto', cmap = 'RdBu_r')
        ax[i,j].axis('equal')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_xlim([0, Lx])
        ax[i,j].set_ylim([0, Ly]) 

#%% =============================================================================
# GENERATE SNAPSHOTS
# =============================================================================

name            = f'Ra_{Ra:06.0f}_Pr_{Pr:04.0f}'
project_path    = 'uncoupled/frames'
listdir         = os.listdir(project_path)
path            = os.path.join(project_path, name)
if name in(listdir):
    shutil.rmtree(path)
os.mkdir(path)

plt.close('all')
f, ax = plt.subplots(2, 2, figsize = (8.58,6.05), constrained_layout = True)

for i in range(0,nt,5):
    print('Frame = ', i)
    f.suptitle('Ra = {:6.0f}  Pr = {:6.1f}  t = {:7.2f}'.format(Ra, Pr, t[i]))
    ax[0,0].set_title('Vorticity')
    ax[0,0].pcolormesh(xx, yy, Vor[i], vmin =-10*amp, vmax = 10*amp, shading = 'auto', cmap = 'RdBu')
    ax[0,0].axis('equal')
    ax[0,0].set_xlim([0, Lx])
    ax[0,0].set_ylim([0, Ly])   
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    
    plt.gca().set_prop_cycle(None)
    ax[0,1].plot(t,ai.T)
    ax[0,1].plot(t[i]*np.ones(nmodes), ai[:,i].T, 'ro', markersize = 4)
    ax[0,1].set_xlabel('t')
    ax[0,1].set_ylabel('$a_i$')
    ax[0,1].set_ylim([-amp, amp])
    ax[0,1].grid(True)
    
    ax[1,0].set_title('Temperature')
    ax[1,0].pcolormesh(xx, yy, THETA[i], vmin = 0, vmax = 1, shading = 'auto', cmap = 'RdBu_r')
    ax[1,0].axis('equal')
    ax[1,0].set_xlim([0, Lx])
    ax[1,0].set_ylim([0,Ly])
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    
    ax[1,1].plot(t,bi.T)
    ax[1,1].plot(t[i]*np.ones(nmodes), bi[:,i].T, 'ro', markersize = 4)
    ax[1,1].set_xlabel('t')
    ax[1,1].set_ylabel('$b_i$')
    ax[1,1].set_ylim([-bmp, bmp])
    ax[1,1].grid(True)

    figpath = os.path.join(path,'{:07.0f}.png'.format(i))
    plt.gca().set_prop_cycle(None)

    plt.savefig(figpath)
    ax[0,0].clear()
    ax[0,1].clear()
    ax[1,0].clear()
    ax[1,1].clear()
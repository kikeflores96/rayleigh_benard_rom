#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:56:34 2024

Coupled ROM for Rayleigh Benard convection
    - Import ROM coefficients form h5 file
    - Simulate the ROM
    - Plot and export snapshots

@author: efloresm
"""
import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
import multiprocessing   as mp
import time              as time
import h5py              as h5
import os                as os
import shutil            as shutil
import scipy.sparse      as sp
from functools       import partial
from scipy.integrate import solve_ivp
from FUN             import *
from coupled.RB_coupled_FUN import *



# =============================================================================
# Import ROM coefficients
# =============================================================================
n_alpha, n      = 6, 16                     # ROM parameters
nx, ny          = 4*(n_alpha-1) + 2, 64     # ROM discretization
ROM_Pr, ROM_Ra  = 1, 1                      # ROM Ra and Pr
ROM_g2          = 1                         # Temperauture scaling in ROM
nmodes          = n_alpha*n                 # Number of modes
ndim            = nmodes                    # ROM dimension

Pr_text     = str(ROM_Pr).replace('.', 'p')
Ra_text     = str(ROM_Ra).replace('.', 'p')
g2_text     = str(ROM_g2).replace('.', 'p')
name        = 'RB_coupled_nx{:03.0f}_ny{:03.0f}_mX{:02.0f}_mY{:02.0f}_N{:03.0f}_Pr{}_Ra{}_g2_{}.h5'.format(nx, ny, n_alpha, n, nmodes, Pr_text, Ra_text, g2_text)
path        = 'coupled/ROM'
input_file  = os.path.join(path, name)
print('File name = {}'.format(name))
# Import the h5 file
with h5.File(input_file, 'r') as h5file:
    # Read the datasets from the file and store them in variables
    X       = h5file['X'][:]
    Y       = h5file['Y'][:]
    TT0     = h5file['TT0'][:]
    base    = h5file['base'][:]
    For0    = h5file['For0'][:]
    For1    = h5file['For1'][:]
    u_Diff  = h5file['u_Diff'][:]
    Nlin    = h5file['Nlin'][:]
    Line    = h5file['Line'][:]
    T_Diff  = h5file['T_Diff'][:]
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
X       = np.linspace(0, Lx, nx, endpoint = False)
Y, W    = clenshaw_curtis_compute(ny, Ly, Y0)
Y, DY   = cheb(ny, Ly, Y0)
DY2     = DY@DY

# 2D grid
xx, yy  = np.meshgrid(X, Y)

# Convert nonlinear term to sparse
Nlin_sparse = sp.bsr_matrix(Nlin.reshape(nmodes, -1), blocksize=(n,n))

# Import critical Ra number
_,_,Ra_c = linear_analysis()

#%% =============================================================================
# Simulate ROM with randomly perturbed initial conditions
# ===============================================================================

# Define partial functions for time integration
# ROM dynamics
ROM_partial = partial(ROM_sparse, 
                      For0  = For0, 
                      For1  = For1, 
                      u_Diff= u_Diff,
                      T_Diff= T_Diff,
                      Line  = Line, 
                      Nlin  = Nlin_sparse)
# ROM jacobian
jac_partial = partial(jac_coupled,
                      For1  = For1, 
                      u_Diff= u_Diff,
                      T_Diff= T_Diff,
                      Line  = Line, 
                      Nlin  = Nlin)

# Import critical Rayleigh number from linear analysis
_,_, Ra_c = linear_analysis()

# Simulation parameters
Ra = 100*Ra_c
Pr = 10

# Initial conditions
ci0 = np.random.normal(0, 0.01, ndim)

# Time array
tend    = 500/np.sqrt(Pr)
nt      = 501
t       = np.linspace(0, tend, nt)

# Integrate the ROM equations
out     = solve_ivp(ROM_partial,  [0, tend], 
                    ci0, args   = (Pr, Ra), 
                    t_eval      = t,
                    jac         = jac_partial,
                    method      = 'LSODA')
ci      = out.y
t       = out.t


# Extract the solution
U       = np.zeros([nt, ny, nx])
V       = np.zeros([nt, ny, nx])
Vor     = np.zeros([nt, ny, nx])
theta0  = np.ones([nt, ny, nx])*np.reshape(TT0, [1, ny, nx])
theta1  = np.zeros([nt, ny, nx])

for i in range(nt):
    for j in range(nmodes):
        U[i]        = U[i]  + base[0,j]*ci[j,i]
        V[i]        = V[i]  + base[1,j]*ci[j,i]
        theta1[i]   = theta1[i] + base[2,j]*ci[j,i]
        
    _, dUdy  = grad(U[i], X, Y, DY)
    dVdx, _  = grad(V[i], X, Y, DY)
    Vor[i]      = dVdx - dUdy
THETA = theta0 + theta1

#%% =============================================================================
# PLOTTING
# =============================================================================

plt.close('all')
amp = np.max(np.abs(ci))

plt.figure(constrained_layout=True)
plt.plot(t, ci.T)
plt.xlabel('$t$')
plt.ylabel('$c_i$')
plt.grid(True)
plt.show()

nh, nv = 3,4
    
f, ax = plt.subplots(nh,nv, figsize = (8,4), 
                     constrained_layout = True)
plt.suptitle('Vorticity field')
for i in range(nh):
    for j in range(nv):
        t_idx = int((4*i + j)/(nh*nv )*nt)
        ax[i,j].set_title(f't = {t[t_idx]:.0f}')
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
        ax[i,j].set_title(f't = {t[t_idx]:.0f}')
        ax[i,j].pcolormesh(xx, yy, THETA[t_idx], vmin = 0, vmax = 1, shading = 'auto', cmap = 'RdBu_r')
        ax[i,j].axis('equal')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_xlim([0, Lx])
        ax[i,j].set_ylim([0, Ly]) 


#%% ===========================================================================
# Save frames
# =============================================================================

name            = f'Ra_{Ra:06.0f}_Pr_{Pr:04.0f}'
project_path    = 'coupled/frames'
listdir         = os.listdir(project_path)
path            = os.path.join(project_path, name)
if name in(listdir):
    shutil.rmtree(path)
os.mkdir(path)

plt.close('all')
f, ax = plt.subplots(2, 2, figsize = (7.58,6.05), constrained_layout = True)

for i in range(0,nt,5):
    print('Frame = ', i)
    
    f.suptitle('Ra = {:6.0f}   t = {:7.2f}'.format(Ra, t[i]))
    ax[0,0].set_title('Vorticity')
    ax[0,0].pcolormesh(xx, yy, Vor[i], vmin =-10*amp, vmax = 10*amp, shading = 'auto', cmap = 'RdBu')
    ax[0,0].axis('equal')
    ax[0,0].set_xlim([0, Lx])
    ax[0,0].set_ylim([0, Ly])   
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    
    plt.gca().set_prop_cycle(None)
    ax[0,1].plot(t,ci.T)
    ax[0,1].plot(t[i]*np.ones(nmodes), ci[:,i].T, 'ro', markersize = 4)
    ax[0,1].set_xlabel('t')
    ax[0,1].set_ylabel('$c_i$')
    ax[0,1].set_ylim([-amp, amp])
    ax[0,1].grid(True)
    
    ax[1,0].set_title('Temperature')
    ax[1,0].pcolormesh(xx, yy, THETA[i], vmin = 0, vmax = 1, shading = 'auto', cmap = 'RdBu_r')
    ax[1,0].axis('equal')
    ax[1,0].set_xlim([0, Lx])
    ax[1,0].set_ylim([0, Ly])
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    

    figpath = os.path.join(path,'{:07.0f}.png'.format(i))
    plt.gca().set_prop_cycle(None)

    plt.savefig(figpath)
    ax[0,0].clear()
    ax[0,1].clear()
    ax[1,0].clear()
    ax[1,1].clear()
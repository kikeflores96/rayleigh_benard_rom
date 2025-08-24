#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:56:34 2024

Uncoupled ROM for Rayleigh Benard convection
    - Build two independent orthonormal modal bases using Stokes modes
    - Project the equations to extract the ROM coefficients
    - Simulate the ROM
    - Plot and export snapshots

@author: efloresm
"""

import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
import time              as time
import os                as os
import shutil            as shutil
from functools           import partial
from scipy.sparse        import linalg
from scipy.linalg        import solve_sylvester
from scipy.integrate     import solve_ivp
from FUN                 import *
from uncoupled.RB_uncoupled_FUN import *

# =============================================================================
# Build two independent modal bases using Stokes mdoes
# =============================================================================

# True/False to log
log = False

# Number of modes per wavenumber 
n       = 8
# Number of wavenumbers
n_alpha = 6
# Number of modes
nmodes  = n*n_alpha
# ROM dimension
ndim    = 2*nmodes
# Domain size
Lx, Ly  = 2, 1
# Y axis origin
y0      = 0
# Fundamental wavenumber in the x,y direction
alpha   = 2*np.pi/Lx
beta    = 2*np.pi/Ly
# Wavenumbers to evaluate in the x direction
kx      = alpha*np.arange(n_alpha)
# Prandtl and Rayleigh numbers
Pr      = 1
Ra      = 1
# X and Y axis distretization
nx, ny  = 4*(n_alpha - 1) + 2, 64
# Grid points and discretized differential operators
X       = np.linspace(0, Lx, nx, endpoint = False)
Y, W    = clenshaw_curtis_compute(ny, Ly, y0)
Y, DY   = cheb(ny, Ly, y0)
DY2     = DY@DY
# 2D grid
xx, yy  = np.meshgrid(X, Y)
# Baseline temperature profile
T0      = (1 -Y)
# Velocity and Temperature modes
u       = np.zeros([nmodes, ny, nx])
v       = np.zeros([nmodes, ny, nx])
theta   = np.zeros([nmodes, ny, nx])

# Iterate over the wavenumbers and compute the modes
for i in range(n_alpha):
    init_index  = i*n
    end_index   = (i+1)*n 
    theta[init_index:end_index] = temp_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n, log)
    u[init_index:end_index],    \
    v[init_index:end_index]     =  vel_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n, log)

# Store the velocity modes in a single array object
Ubase    = np.zeros([2, nmodes, ny, nx])
Ubase[0] = u
Ubase[1] = v
Tbase    = theta.reshape([1, nmodes, ny, nx])


# Normalize the modal basis
Ubase, V_eig_inner_norm = normalize_modes(Ubase, X, Y, W, log)
Tbase, T_eig_inner_norm = normalize_modes(Tbase, X, Y, W, log)
# Show normalization matrices
if log:        
    plt.figure(constrained_layout = True)
    plt.imshow(V_eig_inner_norm)
    plt.xlabel('$a_i$')
    plt.ylabel('$a_j$')
    plt.title('$\\langle V_i, V_j \\rangle$')
    plt.show()
    
    plt.figure(constrained_layout = True)
    plt.imshow(T_eig_inner_norm)
    plt.xlabel('$b_i$')
    plt.ylabel('$b_j$')
    plt.title('$\\langle T_i, T_j \\rangle$')
    plt.show()

#%%=============================================================================
# Project the equations to build a ROM
# =============================================================================
# Initialize counter
t0          = time.time()
# Build a 2D baseline temperature field
TT0         = np.ones([ny,nx])*T0.reshape([ny,1])
TT0_ey      = np.zeros([2, ny, nx])
TT0_ey[1]   = TT0
Tj_ey       = np.zeros([2, ny, nx])
# Define the arrays to store the ROM coefficients
# Momentum equation
u_Nlin      = np.zeros([nmodes, nmodes, nmodes])
u_For1      = np.zeros([nmodes, nmodes])
u_Diff      = np.zeros([nmodes, nmodes])
u_For0      = np.zeros([nmodes])
# Temperature equation
T_Nlin      = np.zeros([nmodes, nmodes, nmodes])
T_Line      = np.zeros([nmodes, nmodes])
T_Diff      = np.zeros([nmodes, nmodes])
# Triad rule marker
marker      = np.zeros([nmodes, nmodes, nmodes])
# Number of Computed Projections for the nonlinear term
NCM         = 0
# Loop over ijk
for i in range(nmodes):
    print("Processing mode = {:2.0f}/{:2.0f}".format(i + 1, nmodes))
    Vi = Ubase[:,i]
    Ti = Tbase[:,i]
    # Baseline temperature profile forcing term
    u_For0[i] = Inner_prod(Vi, TT0_ey, X, Y, W)
    for j in range(nmodes):
        Tj_ey[1]    = Tbase[:,j]
        Tj          = Tbase[:,j]
        Vj          = Ubase[:,j]
        # Laplacian of velocity field
        lapl_Vj     = Lapl_2D(Vj, X, Y, DY2)
        # Laplacian of temperature field
        lapl_Tj     = Lapl_2D(Tj, X, Y, DY2)
        # Linear convective term in energy eq.
        Line_CONV   = CONV(TT0.reshape([1, ny, nx]), Vj, X, Y, DY)
        # Temperature perturbations forcing term
        u_For1[i,j]     = Inner_prod(Vi, Tj_ey, X, Y, W)
        # Viscous dissipation term
        u_Diff[i,j]     = Inner_prod(Vi, lapl_Vj, X, Y, W)
        # Linear convective term in energy equation  
        T_Line[i,j]   = Inner_prod(Ti, Line_CONV, X, Y, W)
        # Diffusion term in energy equation
        T_Diff[i,j]   = Inner_prod(Ti, lapl_Tj, X, Y, W)
        for k in range(nmodes):
            # Extract wavenumber based on iteration indices ijk
            i_wav = i//n
            j_wav = j//n
            k_wav = k//n            
            # Determine if ijk satisfy Triad rule
            triad_it = (i_wav + j_wav == k_wav) | (j_wav + k_wav == i_wav) | (k_wav + i_wav == j_wav)
            if triad_it:
                # If Triad rule is satisfied project the nonlinear term
                marker[i,j,k]   = 1
                # Update counter
                NCM             = NCM + 1
                Vk              = Ubase[:2,k]
                # Compute convective term in momentum and energy equations
                gradVU          = CONV(Vj, Vk, X, Y, DY)
                gradTU          = CONV(Tj, Vk, X, Y, DY)
                # Project the nonlinear terms
                u_Nlin[i,j,k]   = Inner_prod(Vi, gradVU, X, Y, W)
                T_Nlin[i,j,k]   = Inner_prod(Ti, gradTU, X, Y, W)
            
            # Remove the conditional to check that triad rule is satisfied
            if np.abs(u_Nlin[i,j,k])>1e-7 and not(triad_it):
                print('WARNING VELOCITY: non zero influence was overlooked')
                print('i_wav = {:3.0f}\t j_wav = {:3.0f} \t k_wav = {:3.0f} \t {}'.format(i_wav, j_wav, k_wav, triad_it))
            if np.abs(T_Nlin[i,j,k])>1e-7 and not(triad_it):
                print('WARNING TEMPERATURE: non zero influence was overlooked')
                print('i_wav = {:3.0f}\t j_wav = {:3.0f} \t k_wav = {:3.0f} \t {}'.format(i_wav, j_wav, k_wav, triad_it))

# Check that the nonlinear term is lossless
print('Sum over ijk in Nonlinear term = {:7.6e}'.format(u_Nlin.sum()))
print('Number of elements in Nlin = {}'.format(nmodes**3))
print('Number of elements in Nlin computed = {}'.format(NCM))
print('Fraction of projections saved = {:6.4f} %'.format(100*(1 - NCM/nmodes**3)))
# Update timer after projection and output projection time
t1      = time.time()
C_time  = t1 - t0
print('Nmodes = {:5.0f}\t\t time = {:7.4f} s'.format(nmodes, C_time))

# Create sparse arrays for Nonlinear terms
u_Nlin_sparse = sp.bsr_matrix(u_Nlin.reshape(nmodes, -1), blocksize=(n,n))  # Flatten to 2D sparse
T_Nlin_sparse = sp.bsr_matrix(T_Nlin.reshape(nmodes, -1), blocksize=(n,n))  # Flatten to 2D sparse


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
tend    = 500/np.sqrt(Pr)
nt      = 501
t       = np.linspace(0, tend, nt)

# Integrate the ROM equations
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
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$a_i$')
ax[0].grid(True)
ax[1].plot(t, bi.T)
ax[1].set_xlabel('$t$')
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
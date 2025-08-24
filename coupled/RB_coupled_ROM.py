#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:56:34 2024

Coupled ROM for Rayleigh Benard convection
    - Build a coupled orthonormal modal bases using controllability modes
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
import shutil            as shutil
import os                as os
import scipy.sparse      as sp
from scipy.sparse        import linalg
from scipy.linalg        import solve_sylvester
from scipy.integrate     import solve_ivp
from functools           import partial
from FUN                 import *
from coupled.RB_coupled_FUN import *


#%% =============================================================================
# Build a coupled orthonormal basis
# =============================================================================

# True/False to log
log     = True

# Number of modes per wavenumber 
n       = 16
# Number of wavenumbers
n_alpha = 6
# Number of modes
nmodes  = n*n_alpha
# ROM dimension
ndim    = nmodes
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
g2      = 1
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
    
    if kx[i] == 0:
        # Use Stokes modes for kx=0
        theta[init_index+1:end_index+1:2] = temp_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n//2, log)
        u[init_index:end_index:2], \
        v[init_index:end_index:2] = vel_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n//2, log)
    else:
        theta[init_index:end_index], \
        u[init_index:end_index], \
        v[init_index:end_index] = coupled_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n, log, g2=g2) 



# Store the modes in a single array object
dim = 3
base    = np.zeros([dim, nmodes, ny, nx])
base[0] = u
base[1] = v
base[2] = theta

#% Normalize the modal basis
# Define a normalization matrix
# Scale the energy of the temperature field by g2
norm_mat                = np.ones([3,ny, nx])
norm_mat[2]             = g2
base, eig_inner_norm    = normalize_modes(base, X, Y, W, log=False, weight=norm_mat)
# Show normalization matrix

if log:    
    plt.figure(constrained_layout = True)
    plt.imshow(eig_inner_norm)
    plt.xlabel('$a_i$')
    plt.ylabel('$a_j$')
    plt.title('$\\langle \\chi_i, \\chi_j \\rangle$')
    plt.show()  
    
    


#%%=============================================================================
# Build the ROM from the orthonormal mode basis
# =============================================================================
plt.close('all')
# Initialize counter
t0 = time.perf_counter_ns()
# Build a 2D baseline temperature field
TT0         = np.ones([ny,nx])*T0.reshape([ny,1])
TT0_ey      = np.zeros([3, ny, nx])
TT0_et      = np.zeros([3, ny, nx])
TT0_ey[1]   = TT0
TT0_et[2]   = TT0
# Define the arrays to store the ROM coefficients
For0    = np.zeros([nmodes])
For1    = np.zeros([nmodes, nmodes])
u_Diff  = np.zeros([nmodes, nmodes])
T_Diff  = np.zeros([nmodes, nmodes])
Line    = np.zeros([nmodes, nmodes])
Nlin    = np.zeros([nmodes, nmodes, nmodes])
# Auxiliary arrays
Xi      = np.zeros([3, ny,nx])
Vi      = np.zeros([2, ny,nx])
Vj      = np.zeros([2, ny,nx])
Xj      = np.zeros([3, ny,nx])
Vk      = np.zeros([2, ny,nx])
Tj_ey   = np.zeros([3, ny, nx])
# Triad rule marker
marker      = np.zeros([nmodes, nmodes, nmodes])
# Number of Computed Projections for the nonlinear term
NCM         = 0
# Loop over ijk
for i in range(nmodes):
    print("Processing mode = {:2.0f}/{:2.0f}".format(i + 1, nmodes))
    Ti = base[2:,i]
    Vi = base[:2,i]
    Xi = base[:,i]
    # Baseline temperature profile forcing term
    For0[i] = Inner_prod(Xi[1:2], TT0_ey[1:2], X, Y, W)
    for j in range(nmodes):
        Tj_ey[1]    = base[2, j]
        Tj          = base[2:,j]
        Vj          = base[:2,j]
        Xj          = base[:,j]
        # Laplacian and convective terms
        lapl_Vj     = Lapl_2D(Vj, X, Y, DY2)
        lapl_Tj     = Lapl_2D(Tj, X, Y, DY2)
        Line_CONV   = CONV(TT0_et, Vj, X, Y, DY)
        # Projection of linear elements
        For1[i,j]   = Inner_prod(Xi[1:2], Tj_ey[1:2], X, Y, W)
        u_Diff[i,j] = Inner_prod(Vi, lapl_Vj, X, Y, W)
        T_Diff[i,j] = Inner_prod(Ti, lapl_Tj, X, Y, W, weight = norm_mat[2:])
        Line[i,j]   = Inner_prod(Xi[2:], Line_CONV[2:], X, Y, W, weight = norm_mat[2:])
        for k in range(nmodes):
            # Extract wavenumber based on iteration indices ijk
            i_wav       = i//n
            j_wav       = j//n
            k_wav       = k//n
            # Determine if ijk satisfy Triad rule       
            triad_it    = (i_wav + j_wav == k_wav) | (j_wav + k_wav == i_wav) | (k_wav + i_wav == j_wav)
            if triad_it:
                # If Triad rule is satisfied project the nonlinear term
                marker[i,j,k]   = 1
                # Update counter
                NCM             = NCM + 1
                Vk              = base[:2,k]
                # Compute convective term 
                gradXU          = CONV(Xj, Vk, X, Y, DY)
                # Project the nonlinear terms
                Nlin[i,j,k]     = Inner_prod(Xi, gradXU, X, Y, W, weight = norm_mat)
            # Remove the conditional to check that triad rule is satisfied
            if np.abs(Nlin[i,j,k])>1e-7 and not(triad_it):
                print('WARNING VELOCITY: non zero influence was overlooked')
                print('i_wav = {:3.0f}\t j_wav = {:3.0f} \t k_wav = {:3.0f} \t {}'.format(i_wav, j_wav, k_wav, triad_it))

# Check that the nonlinear term is lossless
print('Sum over ijk in Nonlinear term = {:7.6e}'.format(Nlin.sum()))
print('Number of elements in Nlin = {}'.format(nmodes**3))
print('Number of elements in Nlin computed = {}'.format(NCM))
print('Fraction of projections saved = {:6.4f} %'.format(100*(1 - NCM/nmodes**3)))
# Update timer after projection and output projection time          
t1 = time.perf_counter_ns()
print('Projection time = {:8.6f} s'.format((t1 - t0)/1e9))
# Create a sparse array for the Nonlinear term
Nlin_sparse = sp.bsr_matrix(Nlin.reshape(nmodes, -1), blocksize=(n,n))


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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:56:34 2024

Coupled ROM for Rayleigh Benard convection
    - Generate a coupled ROM modal basis
    - Project the equations parallelizing the calculations
    - Save ROM coefficients using h5py
    
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
from functools       import partial
from scipy.integrate import solve_ivp
from FUN             import *
from coupled.RB_coupled_FUN import *


if __name__ == '__main__':
    # =============================================================================
    # Build a coupled orthonormal mode basis
    # =============================================================================
    # Xi = [ui, vi, Ti]
    # Vi = [ui, vi]
    # X = ∑ai xi
    plt.close('all')
    log = False
    # Number of modes per wavenumber 
    n       = 16
    # Number of wavenumbers
    n_alpha = 6
    # Number of modes
    nmodes  = n*n_alpha
    print('Wavenumbers in x = ', n_alpha)
    print('Number of modes in y per wavenumber {:4.0f}'.format(n))
    # Domain size
    Lx, Ly  = 2,1
    # Fundamental wavenumber in the x direction
    alpha   = 2*np.pi/Lx
    beta    = 2*np.pi/Ly

    # Wavenumber in the x direction
    kx      = alpha*np.arange(n_alpha)
    ky      = beta

    # Prandtl and Rayleigh numbers
    Pr      = 1
    Ra      = 1
    g2      = 1

    print('Prandtl number = {:6.2f} '.format(Pr))
    print('Rayleigh number = {:6.2f} '.format(Ra))
    # X and Y axis distretization
    
    nx, ny  = 4*(n_alpha-1) + 2, 64
    
    print('Discretization: nx = {:4.0f}\t ny = {:4.0f}'.format(nx,ny))
    
    # Grid points

    X       = np.linspace(0, Lx, nx, endpoint = False)
    Y, W    = clenshaw_curtis_compute(ny, Ly, 0)
    Y, DY   = cheb(ny, Ly, 0)
    DY2     = DY@DY
    
    # 2D grid
    xx, yy  = np.meshgrid(X, Y)
    
    # Baseline temperature profile
    T0      = (1 - Y)


    # Velocity and Temperature modes
    u       = np.zeros([nmodes, ny, nx])
    v       = np.zeros([nmodes, ny, nx])
    theta   = np.zeros([nmodes, ny, nx])
    print('Extracting the modal basis ...')
    
    # Iterate over the wavenumbers and compute the modes
    for i in range(n_alpha):
        init_index  = i*n
        end_index   = (i+1)*n
        
        if kx[i] == 0:
            theta[init_index+1:end_index+1:2] = temp_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n//2, log)
            u[init_index:end_index:2], \
            v[init_index:end_index:2] = vel_modes(nx, ny, Lx, Ly,  kx[i], T0, Pr, Ra, n//2, log)
        else:
            theta[init_index:end_index], \
            u[init_index:end_index], \
            v[init_index:end_index] = coupled_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n, log, g2= g2) 
            
    # Store the modes in a single array object
    dim = 3
    base    = np.zeros([dim, nmodes, ny, nx])
    base[0] = u
    base[1] = v
    base[2] = theta

    #% Normalize the modal basis
    norm_mat                = np.ones([3,ny, nx])
    norm_mat[2]             = g2
    base, eig_inner_norm    = normalize_modes(base, X, Y, W, log=False, weight=norm_mat)

    t0 = time.time()
    
    print('Computing the Galerkin ROM')
    
    TT0     = np.ones([ny,nx])*T0.reshape([ny,1])
    TT0_ey  = np.zeros([3, ny, nx])
    TT0_et  = np.zeros([3, ny, nx])
    TT0_ey[1] = TT0
    TT0_et[2] = TT0

    For0    = np.zeros([nmodes])
    For1    = np.zeros([nmodes, nmodes])
    u_Diff  = np.zeros([nmodes, nmodes])
    T_Diff  = np.zeros([nmodes, nmodes])
    Line    = np.zeros([nmodes, nmodes])
    Nlin    = np.zeros([nmodes, nmodes, nmodes])
    
    # Create a pool of workers
    num_cores = mp.cpu_count()
    num_cores = 4
    
    print('Parallelization over {:4.0f} cores'.format(num_cores))
    pool = mp.Pool(num_cores)
    
    print('Processing modes...')
    # Prepare partial function
    
    process_mode_partial = partial(process_mode, n=n, nmodes=nmodes, base=base,  
                                   TT0_ey=TT0_ey, TT0_et=TT0_et, X=X, Y=Y, W=W, DY2=DY2, DY=DY, g2=g2)
    
    # Run computations in parallel
    results = pool.map(process_mode_partial, range(nmodes))
    
    NCM = 0
    # Process results
    for i, For0_i, For1_i, u_Diff_i, Nlin_i, Line_i, T_Diff_i, NCM_i in results:
        print("Processing mode = {:2.0f}/{:2.0f}".format(i + 1, nmodes))
        For0[i]     = For0_i
        For1[i]     = For1_i
        u_Diff[i]   = u_Diff_i
        Nlin[i]     = Nlin_i
        Line[i]     = Line_i
        T_Diff[i]   = T_Diff_i
        NCM         = NCM + NCM_i
    
    print('Sum over ijk in Nonlinear term = {:7.6e}'.format(Nlin.sum()))
    print('Number of elements in Nlin = {}'.format(nmodes**3))
    print('Number of elements in Nlin computed = {}'.format(NCM))
    print('Fraction of projections saved = {:6.4f} %'.format(100*(1 - NCM/nmodes**3)))
    
    
    # Close the pool
    pool.close()
    pool.join()
    
    t1 = time.time()
    
    C_time = t1 - t0
    
    print('Nmodes = {:5.0f}\t\t time = {:7.4f} s'.format(nmodes, C_time))
    
    
    print('Storing file in h5')
    
    Pr_text = str(Pr).replace('.', 'p')
    Ra_text = str(Ra).replace('.', 'p')
    g2_text = str(g2).replace('.', 'p')
    name    = 'RB_coupled_nx{:03.0f}_ny{:03.0f}_mX{:02.0f}_mY{:02.0f}_N{:03.0f}_Pr{}_Ra{}_g2_{}.h5'.format(nx, ny, n_alpha, n, nmodes, Pr_text, Ra_text, g2_text)
    path    = 'coupled/ROM'
    output  = os.path.join(path, name)
    print('File name = {}'.format(name))
    # Create an HDF5 file
    with h5.File(output, 'w') as h5file:
        # Create a dataset in the file
        h5file.create_dataset('X',      data=X)
        h5file.create_dataset('Y',      data=Y)
        h5file.create_dataset('TT0',    data=TT0)
        h5file.create_dataset('base',   data=base)
        h5file.create_dataset('For0',   data=For0)
        h5file.create_dataset('For1',   data=For1)
        h5file.create_dataset('u_Diff', data=u_Diff)
        h5file.create_dataset('Nlin',   data=Nlin)
        h5file.create_dataset('Line',   data=Line)
        h5file.create_dataset('T_Diff', data=T_Diff)
    print('File successfully saved to ', output)
    
    # Show normalization matrix
    if log == True:
        plt.figure(constrained_layout = True)
        plt.imshow(eig_inner_norm)
        plt.xlabel('$a_i$')
        plt.ylabel('$a_j$')
        plt.title('$\\langle \\chi_i, \\chi_j \\rangle$')
        plt.show()

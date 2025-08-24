#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:56:34 2024

Uncoupled ROM for Rayleigh Benard convection
    - Generate uncoupled ROM modal bases
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
from uncoupled.RB_uncoupled_FUN import *



if __name__ == '__main__':

    plt.close('all')
    log = False
    
    # Wavenumber in the x direction
    n_alpha = 6
    # Number of modes per wavenumber 
    n       = 8
    # Number of modes
    nmodes  = n*n_alpha
    print('Number of modes in y per wavenumber {:4.0f}'.format(n))
    print('Wavenumbers in x = ', n_alpha)
    # Domain size
    Lx, Ly  = 2, 1
    # Fundamental wavenumber in the x direction
    alpha   = 2*np.pi/Lx
    beta    = 2*np.pi/Ly

    kx      = alpha*np.arange(n_alpha)
    ky      = beta

    # Prandtl and Rayleigh numbers
    Pr      = 1
    Ra      = 1
    
    # X and Y axis distretization
    nx, ny  = 4*(n_alpha-1) + 2, 64
    # nx, ny  = 64, 64
    
    print('Discretization: nx = {:4.0f}\t ny = {:4.0f}'.format(nx,ny))
    
    # Grid points
    X       = np.flip(np.linspace(Lx, 0, nx, endpoint = False))
    X       = np.linspace(0, Lx, nx, endpoint = False)
    Y, W    = clenshaw_curtis_compute(ny, Ly, 0)
    Y, DY   = cheb(ny, Ly, 0)
    DY2     = DY@DY

    # 2D grid
    xx, yy  = np.meshgrid(X, Y)
    # Baseline temperature profile
    T0      = (1 -Y)
    
    # Velocity and Temperature modes
    u       = np.zeros([nmodes, ny, nx])
    v       = np.zeros([nmodes, ny, nx])
    theta   = np.zeros([nmodes, ny, nx])
    
    print('Extracting the modal basis ...')
    # Iterate over the wavenumbers and compute the modes
    for i in range(n_alpha):
        init_index  = i*n
        end_index   = (i+1)*n
        
        theta[init_index:end_index] = temp_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n, log)
        u[init_index:end_index], \
        v[init_index:end_index] = vel_modes(nx, ny, Lx, Ly, kx[i], T0, Pr, Ra, n, log)
    
    
    # Store the modes in a single array object
    dim = 2
    Ubase    = np.zeros([dim, nmodes, ny, nx])
    Ubase[0] = u
    Ubase[1] = v
    
    Tbase    = theta.reshape([1, nmodes, ny, nx])
    
    
    
    print('Normalizing the modal basis ...')
    # Normalize the modal basis
    Ubase, V_eig_inner_norm = normalize_modes(Ubase, X, Y, W)
    Tbase, T_eig_inner_norm = normalize_modes(Tbase, X, Y, W)
    
    
    
    t0 = time.time()
    
    print('Computing the Galerkin ROM')
    
    TT0     = np.ones([ny,nx])*T0.reshape([ny,1])
    
    u_Nlin    = np.zeros([nmodes, nmodes, nmodes])
    u_For1    = np.zeros([nmodes, nmodes])
    u_Diff    = np.zeros([nmodes, nmodes])
    u_For0    = np.zeros([nmodes])
    
    T_Nlin    = np.zeros([nmodes, nmodes, nmodes])
    T_Line    = np.zeros([nmodes, nmodes])
    T_Diff    = np.zeros([nmodes, nmodes])
    
    TT0_ey      = np.zeros([2, ny, nx])
    TT0_ey[1]   = TT0
    Tj_ey       = np.zeros([2, ny, nx])
    
    
    u_For0 = np.zeros(nmodes)
    u_For1 = np.zeros((nmodes, nmodes))
    u_Diff = np.zeros((nmodes, nmodes))
    u_Nlin = np.zeros((nmodes, nmodes, nmodes))
    
    T_Nlin = np.zeros((nmodes, nmodes, nmodes))
    T_Line = np.zeros((nmodes, nmodes))
    T_Diff = np.zeros((nmodes, nmodes))
    
    # Create a pool of workers
    num_cores = mp.cpu_count()
    num_cores = 4
    
    print('Parallelization over {:4.0f} cores'.format(num_cores))
    pool = mp.Pool(num_cores)
    

    
    print('Processing modes...')
    # Prepare partial function
    process_mode_partial = partial(process_mode, n=n, nmodes=nmodes, Ubase=Ubase, Tbase=Tbase, 
                                   TT0=TT0, TT0_ey=TT0_ey, X=X, Y=Y, W=W, DY2=DY2, DY=DY)
    
    # Run computations in parallel
    results = pool.map(process_mode_partial, range(nmodes))
    
    NCM = 0
    # Process results
    for i, u_For0_i, u_For1_i, u_Diff_i, u_Nlin_i, T_Line_i, T_Diff_i, T_Nlin_i, NCM_i in results:
        print("Processing mode = {:2.0f}/{:2.0f}".format(i + 1, nmodes))
        u_For0[i] = u_For0_i
        u_For1[i] = u_For1_i
        u_Diff[i] = u_Diff_i
        u_Nlin[i] = u_Nlin_i
        T_Line[i] = T_Line_i
        T_Diff[i] = T_Diff_i
        T_Nlin[i] = T_Nlin_i
        NCM       = NCM + NCM_i  
    
    print('Sum over ijk in Nonlinear term = {:7.6e}'.format(u_Nlin.sum()))
    print('Number of elements in Nlin = {}'.format(nmodes**3))
    print('Number of elements in Nlin computed = {}'.format(NCM))
    print('Fraction of projections saved = {:6.4f} %'.format(100*(1 - NCM/nmodes**3)))
    
    
    # Close the pool
    pool.close()
    pool.join()
    
    t1 = time.time()
    
    C_time = t1 - t0
    
    print('Nmodes = {:5.0f}\t\t time = {:7.4f} s'.format(nmodes, C_time))

    # print('Storing file in h5')
    
    name    = 'RB_uncoupled_nx{:03.0f}_ny{:03.0f}_mX{:02.0f}_mY{:02.0f}_n{:03.0f}.h5'.format(nx, ny, n_alpha, n, 2*nmodes)
    path    = 'uncoupled/ROM'
    output  = os.path.join(path, name)
    print('File name = {}'.format(name))
    # Create an HDF5 file
    with h5.File(output, 'w') as h5file:
        # Create a dataset in the file
        h5file.create_dataset('Y',      data= Y)
        h5file.create_dataset('X',      data= X)
        h5file.create_dataset('TT0',    data=TT0)
        h5file.create_dataset('Ubase',  data=Ubase)
        h5file.create_dataset('Tbase',  data=Tbase)
        h5file.create_dataset('u_For0', data=u_For0)
        h5file.create_dataset('u_For1', data=u_For1)
        h5file.create_dataset('u_Diff', data=u_Diff)
        h5file.create_dataset('u_Nlin', data=u_Nlin)
        h5file.create_dataset('T_Line', data=T_Line)
        h5file.create_dataset('T_Diff', data=T_Diff)
        h5file.create_dataset('T_Nlin', data=T_Nlin)
    print('File successfully saved to ', output)
    
    # Show normalization matrix
    if log == True:
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

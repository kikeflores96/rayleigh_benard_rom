#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:52:52 2024

Uncoupled ROM for Rayleigh Benard convection
    - Functions file
@author: efloresm
"""

import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
import os                as os
import h5py              as h5
import scipy.sparse      as sp
from scipy.sparse        import linalg
from scipy.linalg        import solve_sylvester
from FUN                 import cheb, Inner_prod, Lapl_2D, grad, CONV

def vel_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n, log = False):
    """
    Compute the velocity modes for an input wavenumber
        - nx: number of nodes in x
        - nt: number of Chebyshev points in y
        - Lx: domain length in x
        - Ly: domain length in y
        - kx: wavenumber
        - T0: baseline temperature profile
        - Pr: Prandtl number
        - Ra: Rayleigh number
        - n: number of modes to compute
        - log: log flag
        
    """
    n_2     = n//2
    Y, DY   = cheb(ny)
    
    X       = np.flip(np.linspace(Lx, 0, nx, endpoint = False))
    dT0     = DY@T0*(2/Ly)
    ddT0    = DY@dT0*(2/Ly)
    y       = Y[1:-1]
    N       = ny-2
    II      = np.eye(N)
    ZZ      = np.zeros([N,N])
    
    Theta   = T0[1:-1]*II
    dTheta  = dT0[1:-1]*II
    ddTheta = ddT0[1:-1]*II
    
    s       = (1/(1 - y**2))*II
    
    Dy      = DY[1:-1,1:-1]
    Dy2     = np.linalg.matrix_power(DY, 2)[1:-1,1:-1]
    DY3     = np.linalg.matrix_power(DY, 3)[1:-1,1:-1]
    DY4     = np.linalg.matrix_power(DY, 4)[1:-1,1:-1]
    Dy4     = ((1 - y**2)*II@DY4 - 8*y*II@DY3 - 12*Dy2)@s
    
    
    Dy  = Dy*2/Ly
    Dy2 = Dy2*(2/Ly)**2
    Dy4 = Dy4*(2/Ly)**4   
    
    Dx      = 1j*kx*II
    Dx2     = -kx**2*II
    Dx4     = kx**4*II
    
    Lapla   = Dx2 + Dy2
    Harmo   = Dx4 -2*Dx2@Dy2 + Dy4
    iLapl   = np.linalg.inv(Lapla)
        
    # Velocity modes
    A       = iLapl@Harmo
    A_adj   = iLapl@Harmo
    BBadj   = II
    
    Gramian         = solve_sylvester(A, A_adj, -BBadj)
    eigval, eigfun  = np.linalg.eig(Gramian)
    order           = np.flip(np.argsort(eigval))
    eigfun          = eigfun[:, order]
    
    if log == True:    
        f, ax = plt.subplots(1,1, figsize  =(6,3), constrained_layout = True)
        ax.plot(y*Ly/2, np.real(eigfun[:N,:n_2]))
        ax.set_xlabel('$y$')
        ax.set_ylabel('$\\psi(y)$')
        plt.figure()
        plt.title('Gramian')
        plt.pcolormesh(np.abs(Gramian))
        
        

    xx, yy = np.meshgrid(X, Y)
    
    Phi     = np.zeros([n, ny, nx], dtype = complex)
    u       = np.zeros([n, ny, nx])
    v       = np.zeros([n, ny, nx])
    
    if kx == 0:
        for i in range(n):
            Phi_ymode           = np.zeros(ny, dtype = 'complex')
            Phi_ymode[1:-1]     = eigfun[:N,i]
            U_ymode             = DY@Phi_ymode*(2/Ly)
            V_ymode             = -1j*kx*Phi_ymode
            Phi_ymode           = np.reshape(Phi_ymode, [ny,1])
            U_ymode             = np.reshape(U_ymode, [ny,1])
            V_ymode             = np.reshape(V_ymode, [ny,1])
            Phi[i]      = np.real(np.exp(1j*kx*xx)*Phi_ymode)
            u[i]        = np.real(np.exp(1j*kx*xx)*U_ymode)
            v[i]        = np.real(np.exp(1j*kx*xx)*V_ymode)
    else:
        for i in range(n_2):
            Phi_ymode           = np.zeros(ny, dtype = 'complex')
            Phi_ymode[1:-1]     = eigfun[:N,i]
            U_ymode             = DY@Phi_ymode*(2/Ly)
            V_ymode             = -1j*kx*Phi_ymode
            Phi_ymode           = np.reshape(Phi_ymode, [ny,1])
            U_ymode             = np.reshape(U_ymode, [ny,1])
            V_ymode             = np.reshape(V_ymode, [ny,1])
            
            
            Phi[2*i]      = np.real(np.exp(1j*kx*xx)*Phi_ymode)
            u[2*i]        = np.real(np.exp(1j*kx*xx)*U_ymode)
            v[2*i]        = np.real(np.exp(1j*kx*xx)*V_ymode)
            
            Phi[2*i + 1]      = np.imag(np.exp(1j*kx*xx)*Phi_ymode)
            u[2*i + 1]        = np.imag(np.exp(1j*kx*xx)*U_ymode)
            v[2*i + 1]        = np.imag(np.exp(1j*kx*xx)*V_ymode)
            
    return u, v

def temp_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n, log = False):
    """
    Compute the temperature modes for an input wavenumber
        - nx: number of nodes in x
        - nt: number of Chebyshev points in y
        - Lx: domain length in x
        - Ly: domain length in y
        - kx: wavenumber
        - T0: baseline temperature profile
        - Pr: Prandtl number
        - Ra: Rayleigh number
        - n: number of modes to compute
        - log: log flag
        
    """
    n_2 = n//2
    
    Y, DY   = cheb(ny)
    X       = np.flip(np.linspace(Lx, 0, nx, endpoint = False))
    dT0     = DY@T0*2/Ly
    ddT0    = DY@dT0*2/Ly
    
    y = Y[1:-1]
    
    N = ny-2
    
    II = np.eye(N)
    ZZ = np.zeros([N,N])
    
    Theta   = T0[1:-1]*II
    dTheta  = dT0[1:-1]*II
    ddTheta = ddT0[1:-1]*II
    
    s   = (1/(1 - y**2))*II
    DY2 = np.linalg.matrix_power(DY, 2)[1:-1,1:-1]
    DY3 = np.linalg.matrix_power(DY, 3)[1:-1,1:-1]
    DY4 = np.linalg.matrix_power(DY, 4)[1:-1,1:-1]
    DY4 = ((1 - y**2)*II@DY4 - 8*y*II@DY3 - 12*DY2)@s
    
    Dy  = DY[1:-1,1:-1]*2/Ly
    Dy2 = DY2*(2/Ly)**2
    Dy4 = DY4*(2/Ly)**4    
       
    Dx  = 1j*kx*II
    Dx2 = -kx**2*II
    Dx4 = kx**4*II
    
    Lapla = Dx2 + Dy2
    Harmo = Dx4 -2*Dx2@Dy2 + Dy4
    iLapl = np.linalg.inv(Lapla)
        
    # Temperature modes
    A       = Lapla
    A_adj   = Lapla
    BBadj   = II
    
    Gramian         = solve_sylvester(A, A_adj, -BBadj)
    eigval, eigfun  = np.linalg.eig(Gramian)
    order           = np.flip(np.argsort(eigval))
    eigfun          = eigfun[:, order]
    
    if log == True:    
        f, ax = plt.subplots(1, figsize  =(6,3), constrained_layout = True)
        ax.plot(y*Ly/2, np.real(eigfun[:N,:n_2]))
        ax.set_xlabel('$y$')
        ax.set_ylabel('$\\theta(y)$')
        plt.figure()
        plt.title('Gramian')
        plt.pcolormesh(np.abs(Gramian))

    xx, yy  = np.meshgrid(X, Y)
    theta   = np.zeros([n, ny, nx])
    
    if kx == 0:
        for i in range(n):
            Theta_ymode         = np.zeros(ny, dtype = 'complex')
            Theta_ymode[1:-1]   = eigfun[:N,i]
            Theta_ymode         = np.reshape(Theta_ymode, [ny,1])
            theta[i]            = np.real(np.exp(1j*kx*xx)*Theta_ymode)
    else:
        
        for i in range(n_2):
            Theta_ymode         = np.zeros(ny, dtype = 'complex')
            Theta_ymode[1:-1]   = eigfun[:N,i]
            Theta_ymode         = np.reshape(Theta_ymode, [ny,1])
            theta[2*i]          = np.real(np.exp(1j*kx*xx)*Theta_ymode)
            theta[2*i + 1]      = np.imag(np.exp(1j*kx*xx)*Theta_ymode)
            
    return theta

def process_mode(i, n, nmodes, Ubase, Tbase, TT0, TT0_ey, X, Y, W, DY2, DY):
    
    """
    Project the problem equations over mode i
        - n: number of modes in the y direction
        - nmodes: number of modes of each basis
        - Ubase: basis of velocity modes
        - Tbase: basis of temperature modes
        - TT0: baseline temperature field
        - TT0_ey: baseline temperature field with state vector structure
        - X: grid points in x
        - Y: grid points in y
        - W: quadrature weights for integration in y
        - D2Y: 2nd order derivative Chebyshev differentiation matrix
        - DY: 1st order derivative Chebyshev differentiation matrix
        
    """
    
    print("Processing mode = {:2.0f}/{:2.0f}".format(i + 1, nmodes))

    Vi = Ubase[:,i]
    Ti = Tbase[:,i]
    
    u_For0_i = Inner_prod(Vi, TT0_ey, X, Y, W)
    u_For1_i = np.zeros(nmodes)
    u_Diff_i = np.zeros(nmodes)
    u_Nlin_i = np.zeros((nmodes, nmodes))
    
    T_Line_i = np.zeros(nmodes)
    T_Diff_i = np.zeros(nmodes)
    T_Nlin_i = np.zeros((nmodes, nmodes))
    
    NCM_i    = 0

    for j in range(nmodes):
        Tj_ey       = np.zeros([2, len(Y), len(X)])
        Tj_ey[1]    = Tbase[:,j]
        Tj          = Tbase[:,j]
        Vj          = Ubase[:,j]
        
        lapl_Vj     = Lapl_2D(Vj, X, Y, DY2)
        
        u_For1_i[j] = Inner_prod(Vi, Tj_ey, X, Y, W)
        u_Diff_i[j] = Inner_prod(Vi, lapl_Vj, X, Y, W)
        
        lapl_Tj     = Lapl_2D(Tj, X, Y, DY2)
        Line_CONV   = CONV(TT0.reshape([1, len(Y), len(X)]), Vj, X, Y, DY)
        
        T_Line_i[j] = Inner_prod(Ti, Line_CONV, X, Y, W)
        T_Diff_i[j] = Inner_prod(Ti, lapl_Tj, X, Y, W)
        
        for k in range(nmodes):

            i_wav = i//n
            j_wav = j//n
            k_wav = k//n            
            
            triad_it = (i_wav + j_wav == k_wav) | (j_wav + k_wav == i_wav) | (k_wav + i_wav == j_wav)
            
            if triad_it:
                NCM_i           = NCM_i + 1
                Vk              = Ubase[:2,k]
                gradVU          = CONV(Vj, Vk, X, Y, DY)
                gradTU          = CONV(Tj, Vk, X, Y, DY)
                u_Nlin_i[j,k]   = Inner_prod(Vi, gradVU, X, Y, W)
                T_Nlin_i[j,k]   = Inner_prod(Ti, gradTU, X, Y, W)
            
    return i, u_For0_i, u_For1_i, u_Diff_i, u_Nlin_i, T_Line_i, T_Diff_i, T_Nlin_i, NCM_i


def ROM_sparse(t, ci, Pr, Ra, u_For0, u_For1, u_Diff, u_Nlin, T_Line, T_Diff, T_Nlin, nmodes):
    
    """
    Compute RHS of Galerkin ROM dynamics
        
    """
    
    aj = ci[:nmodes]
    bj = ci[nmodes:]
    
    aj_outer    = np.outer(aj, aj).ravel()
    bj_outer    = np.outer(bj, aj).ravel()

    Nlin_meq = u_Nlin.dot(aj_outer)
    Nlin_eeq = T_Nlin.dot(bj_outer)
    
    daidt   = Pr*(u_For0 + u_For1@bj) + Pr/np.sqrt(Ra)*u_Diff@aj - Nlin_meq
    dbidt   = T_Diff@bj/np.sqrt(Ra) - T_Line@aj  - Nlin_eeq
    
    dcidt   = np.hstack([daidt, dbidt])

    return dcidt


def jac_uncoupled(t, ci, Pr, Ra, u_For1, u_Diff, u_Nlin, T_Line, T_Diff, T_Nlin, nmodes):
    """
    Compute the Jacobian of the system
        
    """
    J11   = Pr/np.sqrt(Ra)*u_Diff - u_Nlin@ci[:nmodes] - np.transpose(u_Nlin, [0,2,1])@ci[:nmodes]
    J12   = Pr*u_For1
    J21   = -np.transpose(T_Nlin, [0, 2, 1])@ci[nmodes:] - T_Line
    J22   = T_Diff/np.sqrt(Ra)  - T_Nlin@ci[:nmodes]
    jac   = np.block([[J11,  J12],
                      [ J21, J22]])
    return jac





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:52:52 2024

Coupled ROM for Rayleigh Benard convection
    - Functions file
@author: efloresm
"""
import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
from scipy.sparse        import linalg
from scipy.linalg        import solve_sylvester
from FUN                 import *


def coupled_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n, log = False, g2 = 1):
    """
    Compute the controllability modes for an input wavenumber
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
        - g2: scaling factor for the temperature field
        
    """
    # Number of eigenvalues to be computed
    n_2     = n//2
    # Chebyshev grid and 1st order derivation matrix
    Y, DY   = cheb(ny)
    # Fourier grid
    X       = np.linspace(0, Lx, nx, endpoint = False)
    # Take the 1st derivative of mean temperature field
    dT0     = DY@T0*(2/Ly)
    # Number of grid points with Dirichlet BC
    N       = ny-2
    # Remove end points
    y       = Y[1:-1]
    # Unitary matrix
    II      = np.eye(N)
    # Zero matrix
    ZZ      = np.zeros([N,N])
    # Convert to diagonal matrix
    # Baseline temperature field
    Theta   = T0[1:-1]*II
    # Baseline temperature gradient
    dTheta  = dT0[1:-1]*II

    # Compute high order Chebyshev derivation matrices
    s       = (1/(1 - y**2))*II
    Dy      = DY[1:-1,1:-1]
    Dy2     = np.linalg.matrix_power(DY, 2)[1:-1,1:-1]
    DY3     = np.linalg.matrix_power(DY, 3)[1:-1,1:-1]
    DY4     = np.linalg.matrix_power(DY, 4)[1:-1,1:-1]
    Dy4     = ((1 - y**2)*II@DY4 - 8*y*II@DY3 - 12*Dy2)@s
    # Reescale derivation matrices to domain length
    Dy      = Dy*2/Ly
    Dy2     = Dy2*(2/Ly)**2
    Dy4     = Dy4*(2/Ly)**4   
    # Derivation matrices in the x direction
    Dx      = 1j*kx*II
    Dx2     = -kx**2*II
    Dx4     = kx**4*II
    # Compute 2D operators
    Lapla   = Dx2 + Dy2
    Harmo   = Dx4 -2*Dx2@Dy2 + Dy4
    iLapl   = np.linalg.inv(Lapla)
    # Define state matrix terms
    A11 = Pr/np.sqrt(Ra)*iLapl@Harmo
    A12 = -1j*kx*Pr*iLapl
    A21 = 1j*kx*dTheta
    A22 = 1/np.sqrt(Ra)*Lapla
    # Build direct problem matrices
    A = np.block([[A11, A12],[A21, A22]])
    B = np.block([[iLapl@Dy, -1j*kx*iLapl, ZZ],[ZZ, ZZ, II]])
    C = np.block([[Dy, ZZ],[-1j*kx*II, ZZ],[ZZ, II]])
    # Compute adjoint state matrix terms
    A11_adj = Pr/np.sqrt(Ra)*iLapl@Harmo
    A12_adj = 1j*kx*g2*iLapl@dTheta
    A21_adj = -1j*kx*Pr/g2*II
    A22_adj = 1/np.sqrt(Ra)*Lapla
    # Build adjoint problem matrices
    A_adj = np.block([[A11_adj, A12_adj],[A21_adj, A22_adj]])
    B_adj = np.block([[       Dy,    ZZ],
                      [-1j*kx*II,    ZZ], 
                      [       ZZ, g2*II]])
    
    # Compute BB+ product
    BBadj = np.block([[II,    ZZ],
                      [ZZ, g2*II]])
    # Solve sylvester equation to obtain the Gramian
    Gramian         = solve_sylvester(A, A_adj, -BBadj)
    # Compute Gramian eigenpairs
    eigval, eigfun  = np.linalg.eig(Gramian)

    # Sort eigenvalues
    order           = np.flip(np.argsort(np.abs(eigval)))
    # order           = (np.argsort(eigval))
    # Sort eigenfunctions
    eigfun          = eigfun[:, order]
    
    
    linestyle = ['o-', '>-.', 's:', '<--', 'x-', 'X-.', 'D:', '*--']
    
    while len(linestyle)<n_2:
        linestyle=linestyle + linestyle
    
    
    if log == True:    
        f, ax = plt.subplots(2,2, figsize  =(8.5,4.5), constrained_layout = True)
        
        for i in range(n_2):
            ax[0,0].plot(y, np.real(eigfun[:N,i]), linestyle[i], markevery = 10)
            ax[1,0].plot(y, np.imag(eigfun[:N,i]), linestyle[i], markevery = 10)
            ax[0,1].plot(y, np.real(eigfun[N:,i]), linestyle[i], markevery = 10)
            ax[1,1].plot(y, np.imag(eigfun[N:,i]), linestyle[i], markevery = 10)
            
        ax[0,0].set_xlabel('$y$')
        ax[0,0].set_ylabel('$\\psi_r(y)$')
        ax[0,0].set_xlim([-1,1])
        ax[0,0].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax[0,0].grid(True)
        

        ax[0,1].set_xlabel('$y$')
        ax[0,1].set_ylabel('$\\theta_r(y)$')
        ax[0,1].set_xlim([-1,1])
        ax[0,1].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax[0,1].grid(True)
        
        
        ax[1,0].set_xlabel('$y$')
        ax[1,0].set_ylabel('$\\psi_i(y)$')
        ax[1,0].set_xlim([-1,1])
        ax[1,0].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax[1,0].grid(True)
        
         
        ax[1,1].set_xlabel('$y$')
        ax[1,1].set_ylabel('$\\theta_i(y)$')
        ax[1,1].set_xlim([-1,1])
        ax[1,1].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax[1,1].grid(True)
        
        
        plt.figure()
        plt.title('Gramian')
        plt.pcolormesh(np.abs(Gramian))
        

    
    xx, yy = np.meshgrid(X, Y)
    
    
    Phi     = np.zeros([n, ny, nx], dtype = complex)
    theta   = np.zeros([n, ny, nx])
    u       = np.zeros([n, ny, nx])
    v       = np.zeros([n, ny, nx])
    
    for i in range(n_2):
        Phi_ymode           = np.zeros(ny, dtype = 'complex')
        Theta_ymode         = np.zeros(ny, dtype = 'complex')
        Theta_ymode[1:-1]   = eigfun[N:,i]
        Phi_ymode[1:-1]     = eigfun[:N,i]
        U_ymode             = DY@Phi_ymode*(2/Ly)
        V_ymode             = -1j*kx*Phi_ymode
        Phi_ymode           = np.reshape(Phi_ymode, [ny,1])
        Theta_ymode         = np.reshape(Theta_ymode, [ny,1])
        U_ymode             = np.reshape(U_ymode, [ny,1])
        V_ymode             = np.reshape(V_ymode, [ny,1])
        
        
        Phi[2*i]      = np.real(np.exp(1j*kx*xx)*Phi_ymode)
        theta[2*i]    = np.real(np.exp(1j*kx*xx)*Theta_ymode)
        u[2*i]        = np.real(np.exp(1j*kx*xx)*U_ymode)
        v[2*i]        = np.real(np.exp(1j*kx*xx)*V_ymode)
        
        Phi[2*i + 1]      = np.imag(np.exp(1j*kx*xx)*Phi_ymode)
        theta[2*i + 1]    = np.imag(np.exp(1j*kx*xx)*Theta_ymode)
        u[2*i + 1]        = np.imag(np.exp(1j*kx*xx)*U_ymode)
        v[2*i + 1]        = np.imag(np.exp(1j*kx*xx)*V_ymode)
        
    return theta, u, v

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
    
    Dy      = Dy*2/Ly
    Dy2     = Dy2*(2/Ly)**2
    Dy4     = Dy4*(2/Ly)**4   

    Dx      = 1j*kx*II
    Dx2     = -kx**2*II
    Dx4     = kx**4*II
    
    # Define 2D operators
    Lapla = Dx2 + Dy2
    Harmo = Dx4 -2*Dx2@Dy2 + Dy4
    iLapl = np.linalg.inv(Lapla)
        
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
        ax.plot(y, np.real(eigfun[:N,:n_2]))
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
    
    Dy      = Dy*2/Ly
    Dy2     = Dy2*(2/Ly)**2
    Dy4     = Dy4*(2/Ly)**4   

    Dx      = 1j*kx*II
    Dx2     = -kx**2*II
    Dx4     = kx**4*II
    
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
        ax.plot(y, np.real(eigfun[:N,:n_2]))
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
        
            theta[i]          = np.real(np.exp(1j*kx*xx)*Theta_ymode)
    else:
        
        for i in range(n_2):
    
            Theta_ymode         = np.zeros(ny, dtype = 'complex')
            Theta_ymode[1:-1]   = eigfun[:N,i]
            Theta_ymode         = np.reshape(Theta_ymode, [ny,1])
        
            theta[2*i]          = np.real(np.exp(1j*kx*xx)*Theta_ymode)
            theta[2*i + 1]      = np.imag(np.exp(1j*kx*xx)*Theta_ymode)


    return theta

def process_mode(i, n, nmodes, base, TT0_ey, TT0_et, X, Y, W, DY2, DY, g2):
    """
    Project the problem equations over mode i
        - n: number of modes in the y direction
        - nmodes: number of modes of each basis
        - base: basis of coupled modes
        - TT0_ey: baseline temperature field in the vertical velocity component
        - TT0_et: baseline temperature field in the temperature component
        - X: grid points in x
        - Y: grid points in y
        - W: quadrature weights for integration in y
        - D2Y: 2nd order derivative Chebyshev differentiation matrix
        - DY: 1st order derivative Chebyshev differentiation matrix
        - g2: scaling factor for the temperature field
    """
    
    print("Processing mode = {:2.0f}/{:2.0f}".format(i + 1, nmodes))
    
    Ti = base[2:,i]
    Vi = base[:2,i]
    Xi = base[:,i]
    
    Tj_ey   = np.zeros_like(TT0_ey)
    
    in_prod_weight     = np.ones_like(Xi)
    in_prod_weight[2]  = g2
    
    For1_i      = np.zeros(nmodes)
    u_Diff_i    = np.zeros(nmodes)
    Line_i      = np.zeros(nmodes)
    T_Diff_i    = np.zeros(nmodes)
    Nlin_i      = np.zeros((nmodes, nmodes))
    
    For0_i = Inner_prod(Xi[1:2], TT0_ey[1:2], X, Y, W)
    
    NCM_i    = 0
    
    for j in range(nmodes):
        Tj_ey[1]    = base[2, j]
        Tj          = base[2:,j]
        Vj          = base[:2,j]
        Xj          = base[:,j]

        lapl_Vj     = Lapl_2D(Vj, X, Y, DY2)
        lapl_Tj     = Lapl_2D(Tj, X, Y, DY2)
        Line_CONV   = CONV(TT0_et, Vj, X, Y, DY)
        
        For1_i[j]       = Inner_prod(Xi[1:2], Tj_ey[1:2], X, Y, W)
        u_Diff_i[j]     = Inner_prod(Vi, lapl_Vj, X, Y, W)
        T_Diff_i[j]     = Inner_prod(Ti, lapl_Tj, X, Y, W, weight = in_prod_weight[2:])
        Line_i[j]       = Inner_prod(Xi[2:], Line_CONV[2:], X, Y, W, weight = in_prod_weight[2:])
        for k in range(nmodes):
            # Vk          = base[:2,k]
            # gradXU      = CONV(Xj, Vk, X, Y, DY)
            # Nlin_i[j,k] = Inner_prod(Xi, gradXU, X, Y, W)
            i_wav = i//n
            j_wav = j//n
            k_wav = k//n            
            
            triad_it = (i_wav + j_wav == k_wav) | (j_wav + k_wav == i_wav) | (k_wav + i_wav == j_wav)
            
            if triad_it:
                NCM_i           = NCM_i + 1
                Vk              = base[:2,k]
                gradXU      = CONV(Xj, Vk, X, Y, DY)
                Nlin_i[j,k] = Inner_prod(Xi, gradXU, X, Y, W, weight = in_prod_weight)
            

    return i, For0_i, For1_i, u_Diff_i, Nlin_i, Line_i, T_Diff_i, NCM_i



def ROM_sparse(t, Xi, Pr, Ra, For0, For1, u_Diff, T_Diff, Line, Nlin):
    """
    Compute RHS of Galerkin ROM dynamics
        
    """
    Xi_outer    = np.outer(Xi, Xi).ravel()
    Nlin_eq     = Nlin.dot(Xi_outer)
    dXidt       = (Pr*(For0 + For1@Xi) +
                   Pr/np.sqrt(Ra)*u_Diff@Xi + 
                   1/np.sqrt(Ra)*T_Diff@Xi 
                   - Line@Xi - Nlin_eq)
    return dXidt

def jac_coupled(t, y, Pr, Ra, For1, u_Diff, T_Diff, Line, Nlin):
    """
    Compute the Jacobian of the system
        
    """
    Jac       = (Pr*For1 + Pr/np.sqrt(Ra)*u_Diff + 1/np.sqrt(Ra)*T_Diff 
                 - Line - Nlin@y - np.transpose(Nlin, [0,2,1])@y)
    return Jac





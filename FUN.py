#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:52:52 2024

@author: efloresm
"""

import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
from scipy.sparse        import linalg
from scipy.linalg        import solve_sylvester


def cheb(nn, Lx=2, x0=-1):
    """
    Compute 1st order chebyshev differentiation matrix
        
    """
    N   = nn-1
    n   = np.arange(0, N+1)
    x   = np.cos(np.pi*n/N).reshape(N+1,1)
    c   = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
    X   = np.tile(x,(1,N+1))
    dX  = X - X.T
    D   = np.dot(c,1./c.T)/(dX+np.eye(N+1))
    D   = D - np.diag(np.sum(D.T,axis=0))
    
    x   = Lx*(x.reshape(N+1) + 1)/2 + x0
    D   = D*2/Lx
    return x, D

def clenshaw_curtis_compute(n, Lx=2, x0 = -1):
    """
    Compute clenshaw curtis quadrature weights for integration
    over Chebyshev grid
        
    """
    i = np.arange(n)
    theta = (n-1-i)*np.pi/(n-1)
    x = np.cos (theta)
    w = np.ones( n )
    jhi = ( n - 1 ) // 2
    for i in range(n):
      for j in range (jhi):
        if ( 2 * ( j + 1 ) == ( n - 1 ) ):
          b = 1.0
        else:
          b = 2.0
        w[i] = w[i] - b * np.cos ( 2.0 * ( j + 1 ) * theta[i] )/( 4 * j * ( j + 2 ) + 3 )
    w[0] = w[0] /( n - 1 )
    for i in range (1, n - 1):
      w[i] = 2.0 * w[i] /( n - 1 )
    w[n-1] = w[n-1] /( n - 1 )
    
    x   = Lx*(x + 1)/2 - x0
    w   = w*Lx/2
    return x, w




def Inner_prod(f, g, x, y, W, weight = 1):
    """
    Compute the inner product between two vector fields
        - f: vector field
        - g: vector field
        - x: grid (Fourier)
        - y: grid (Chebyshev)
        - W: Clenshaw-Curtis quadrature weights
        - weight: weight matrix for the inner produc components
        
    """
    dim, ny, nx     = f.shape
    
    dx = x[1] - x[0]
    Lx = nx*dx
    
    Ly = np.max(y) - np.min(y)

    prod        = np.sum(f*g*weight, axis = 0)
    w           = np.reshape(W, [ny, 1])
    Y_int_prod  = np.sum(prod*w, axis = 0)
    X_int_Y_int = Lx/(nx)*np.sum(Y_int_prod)
    Inner_prod  = X_int_Y_int/Lx/Ly
    return Inner_prod



def normalize_modes(base, X, Y, W, log= False, weight = 1):
    """
    Normalize a modal basis
        - base: modal basis
        - X: grid points (Fourier)
        - Y: grid points (Chebyshev)
        - W: Clenshaw-Curtis quadrature weights
        - weight: weight matrix to perform the inner product operations
        
    """
    dim, nmodes, ny, nx = base.shape
    eig_inner_norm   = np.zeros([nmodes,nmodes])
    
    base_normalized = np.zeros_like(base)
    for i in range(nmodes):
        mod = np.sqrt(Inner_prod(base[:,i], base[:,i], X, Y, W, weight))
        base_normalized[:,i] = base[:,i]/mod
    
    for i in range(nmodes):
        for j in range(nmodes):
            
            f = base_normalized[:,i]
            g = base_normalized[:,j]
            
            eig_inner_norm[i,j] = Inner_prod(f, g, X, Y, W, weight)
    
    if log == True:
        plt.figure()
        plt.imshow(eig_inner_norm)
    max_nonortho = np.max(np.abs(eig_inner_norm - np.eye(nmodes)))
    print("Maximum non-orthogonality = {:5.4e}".format(max_nonortho))
        
    return base_normalized,  eig_inner_norm

def Lapl_2D(f, X, Y, DY2):
    """
    Compute the Laplacian of a vector field
        
    """
    dim, ny, nx = f.shape
    dx          = np.abs(X[1] - X[0])
    
    f_fft       = np.fft.rfft(f, axis = 2)
    nf          = f_fft.shape[2]
    kx          = np.fft.rfftfreq(nx, dx)
    kx          = 1j*2*np.pi*kx.reshape([1, 1, nf])
    D2FDX2_fft  = kx**2*f_fft
    D2DDX2      = np.fft.irfft(D2FDX2_fft, axis=2)

    D2DDY2      = np.zeros_like(f)
    for i in range(dim):
        D2DDY2[i]   = DY2@f[i]
    lapl = D2DDX2 + D2DDY2

    return lapl

def grad(f, X, Y, DY):
    """
    Compute the gradient of a vector field
        
    """
    ny, nx      = f.shape
    dx          = np.abs(X[1] - X[0])
    # dfdx        = np.zeros_like(f)
    
    f_fft       = np.fft.rfft(f, axis = 1)
    nf          = f_fft.shape[1]
    kx          = np.fft.rfftfreq(nx, dx)
    kx          = 1j*2*np.pi*kx.reshape([1, nf])
    dfdx_fft    = kx*f_fft
    dfdx        = np.fft.irfft(dfdx_fft, axis=1)
    

    dfdy        = DY@f
    return dfdx, dfdy

def CONV(Xj, Vk, X, Y, DY_2D):
    """
    Compute the convective term for field Xj
        - Xj: vector field that is advected
        - Vk: velocity field
        
    """
    dim, ny, nx = Xj.shape

    gradXU      = np.zeros_like(Xj)
    for l in range(dim):
        dXldx, dXldy    = grad(Xj[l], X, Y, DY_2D)
        gradXU[l]       = Vk[0]*dXldx + Vk[1]*dXldy
    return gradXU



def detA(Ra, k):
    q0 = 1j * k * (-1 + (Ra/k**4)**(1/3))**(1/2)
    q1 = k * (1 + (Ra/k**4)**(1/3) * (1/2 + 1j * np.sqrt(3)/2))**(1/2)
    q2 = k * (1 + (Ra/k**4)**(1/3) * (1/2 - 1j * np.sqrt(3)/2))**(1/2)

    A = np.array([
        [1, 1, 1, 1, 1, 1],
        [np.exp(q0), np.exp(-q0), np.exp(q1), np.exp(-q1), np.exp(q2), np.exp(-q2)],
        [q0, -q0, q1, -q1, q2, -q2],
        [q0*np.exp(q0), -q0*np.exp(-q0), q1*np.exp(q1), -q1*np.exp(-q1), q2*np.exp(q2), -q2*np.exp(-q2)],
        [(q0**2-k**2)**2, (q0**2-k**2)**2, (q1**2-k**2)**2, (q1**2-k**2)**2, (q2**2-k**2)**2, (q2**2-k**2)**2],
        [(q0**2-k**2)**2*np.exp(q0), (q0**2-k**2)**2*np.exp(-q0), (q1**2-k**2)**2*np.exp(q1), 
         (q1**2-k**2)**2*np.exp(-q1), (q2**2-k**2)**2*np.exp(q2), (q2**2-k**2)**2*np.exp(-q2)]])
    return np.linalg.det(A)

def linear_analysis():
    """
    Linear analysis of RB convection
        
    """
    ks          = np.linspace(1, 8, 701)# range of ks to solve for
    Ra_c        = np.zeros_like(ks, dtype='complex')
    Ra          = 7000  # start guess for k = 8
    delta_Ra    = 0.001  # step size for derivative approximation
    tolerance   = 0.001  # tolerance for the absolute error of Ra
    
    for i in range(len(ks)):
        k   = ks[i]
        dR  = 1  
        # Solve with Newton's method until the absolute value error
        # approximation dR reaches a value below the tolerance
        while abs(dR) > tolerance:
            f  = detA(Ra, k)
            df  = (detA(Ra + delta_Ra, k) - detA(Ra - delta_Ra, k)) / (2 * delta_Ra)
            dR  = -f / df
            Ra  += dR
        Ra_c[i] = Ra
    crit_Ra     = np.min(np.real(Ra_c))
    return Ra_c, ks, crit_Ra
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Kalman Filter (EKF) for Rayleigh-Bénard ROM state estimation.

This module provides the core building blocks for the EKF described in
§III–IV of the accompanying paper.  The main components are:

EKF predict / update
--------------------
    estimate  — prediction step: integrate ROM with RK45 and propagate
                covariance via the first-order (Euler) linearisation
                (paper eqs. 24–26).
    update    — correction step: Joseph-form covariance update
                (paper eqs. 28–32).

DNS data import
---------------
    import_DNS — reads Dedalus 3 HDF5 segment files.

    Grid-point note:  Dedalus uses the ChebyshevT basis, whose collocation
    points are the Gauss–Chebyshev (root) points
        y_j = cos(π(2j+1)/(2N)),  j = 0, …, N-1,
    whereas the ROM is built on the Gauss–Lobatto (extrema) points returned
    by cheb(), which include both wall boundaries.  The two grids are
    different and fields must be re-interpolated before projecting onto the
    ROM basis.  The exact transformation is a DCT-II followed by an IDCT-I:
        f_GL = IDCT-I( DCT-II(f_GC) ) / (2 N)
    This identity holds because both sets of points are nodes of Chebyshev
    polynomials, so the DCT pair performs an exact Chebyshev expansion and
    re-evaluation.  See Burns et al. (Dedalus paper, ref. [47]) for details
    of the Dedalus spectral representation.

Mode upsampling (paper eq. 38)
-------------------------------
    upsample_modes — the ROM basis is computed on a coarser x grid
    (nx_ROM) than the DNS (nx_DNS).  To evaluate basis functions at DNS
    probe locations the modes are upsampled via zero-padding in Fourier
    space and amplitude-corrected by the factor nx_DNS/nx_ROM:
        χ̂_us[k] = (nx_DNS/nx_ROM) χ̂[k],  k = 0, …, nx_ROM//2
        χ_us = IRFFT( χ̂_us )
    This is equivalent to spectral interpolation onto the finer grid.

Observation matrix (paper eqs. 21–22)
--------------------------------------
    build_obs_mat — constructs the linear map H from ROM modal coefficients
    to physical-space probe measurements.  For a set of probes at locations
    {x_ℓ}, {x_p}, {x_q} the block observation matrix is
        H = [ H^u ]       H^u_{ℓj} = U_j(x_ℓ)
            [ H^v ]   ,   H^v_{pj} = V_j(x_p)
            [ H^θ ]       H^θ_{qj} = Θ_j(x_q)
    where U_j, V_j, Θ_j are the j-th ROM basis functions evaluated on the
    upsampled grid (after upsample_modes).

Run scripts in state_estimation/ from the repo root, e.g.:
    python state_estimation/RB_EKF_coarse_grid.py

@author: efloresm
"""

import sys
import os

# Add repo root to path so FUN.py and coupled/ are importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy             as np
import scipy             as scipy
import scipy.linalg      as scla
import matplotlib.pyplot as plt
import time              as tm
import h5py              as h5
import scipy.sparse      as sp
import jax               as jax
import jax.numpy         as jnp
from functools           import partial
from scipy.integrate     import solve_ivp
from scipy.fftpack       import dct, idct
from FUN                 import (cheb, clenshaw_curtis_compute, Inner_prod,
                                  normalize_modes, grad, linear_analysis)


# =============================================================================
# DNS DATA IMPORT
# =============================================================================

def import_DNS(Ra, Pr, path=None):
    """
    Import and concatenate all Dedalus 3 DNS segment files for a given case.

    Reads merged segment HDF5 files (_s1.h5, _s2.h5, ...) produced by
    DNS_kalman/RB_DNS_dedalus3.py.

    Dedalus stores fields on Chebyshev-T (Gauss-Chebyshev root) points, while
    the ROM uses the Gauss-Lobatto grid produced by cheb(). A DCT-II / IDCT-I
    pair re-interpolates the y direction onto the Gauss-Lobatto grid so that
    fields are consistent with the ROM inner products and differentiation matrix.

    Parameters
    ----------
    Ra   : float      — Rayleigh number (unused; kept for API compatibility)
    Pr   : float      — Prandtl number  (unused; kept for API compatibility)
    path : str | None — path to case folder (relative to repo root), e.g.
                        'DNS_kalman/simulations/DNS_Pr10p0_R080_rseed001'.
                        If None, falls back to legacy path construction.

    Returns
    -------
    X   : (nx,)          x grid (uniform Fourier)
    Y   : (ny,)          y grid (Gauss-Lobatto, as returned by cheb())
    U   : (nt, ny, nx)   x-velocity
    V   : (nt, ny, nx)   y-velocity
    T   : (nt, ny, nx)   full temperature b = T0 + theta'
    W   : (nt, ny, nx)   vorticity
    t   : (nt,)          simulation time
    """
    if path is None:
        problem   = 'RB_DNS'
        Pr_folder = 'Pr_{:08.3f}'.format(Pr).replace('.', 'p')
        case_name = 'Ra_{:08.0f}'.format(Ra)
        case_path = os.path.join(problem, Pr_folder, case_name)
    else:
        case_path = path

    file_list = os.listdir(case_path)
    h5_list   = sorted([f for f in file_list if f.endswith('.h5')])

    # Accumulate fields across segment files
    theta = np.empty(shape=[0, 128, 64])
    vor   = np.empty(shape=[0, 128, 64])
    ux    = np.empty(shape=[0, 128, 64])
    uy    = np.empty(shape=[0, 128, 64])
    t     = np.empty(shape=[0])

    for h5_file in h5_list:
        h5_path = os.path.join(case_path, h5_file)
        file    = h5.File(h5_path, mode='r')
        theta = np.concatenate([theta, np.copy(file['tasks']['buoyancy'])],    axis=0)
        vor   = np.concatenate([vor,   np.copy(file['tasks']['vorticity'])],   axis=0)
        ux    = np.concatenate([ux,    np.copy(file['tasks']['u'][:, 0])],     axis=0)
        uy    = np.concatenate([uy,    np.copy(file['tasks']['u'][:, 1])],     axis=0)
        t     = np.concatenate([t, np.copy(
                    file['tasks']['buoyancy'].dims[0]['sim_time'])])

    # Grid from the last opened file (same for all segments)
    X  = np.copy(file['tasks']['buoyancy'].dims[1][0])   # (nx,) uniform x grid
    Ny = file['tasks']['buoyancy'].dims[2][0].shape[0]

    Lx, Ly = 2, 1

    # Dedalus layout: (nt, nx, ny) with y from bottom (y=0) to top (y=1).
    # Step 1 — flip y axis so it runs top → bottom, then transpose to (nt, ny, nx).
    u_ft = np.transpose(np.flip(ux,    axis=2), [0, 2, 1])
    v_ft = np.transpose(np.flip(uy,    axis=2), [0, 2, 1])
    T_ft = np.transpose(np.flip(theta, axis=2), [0, 2, 1])
    w_ft = np.transpose(np.flip(vor,   axis=2), [0, 2, 1])

    # Step 2 — convert Chebyshev-T (Gauss) grid → Gauss-Lobatto grid used by cheb().
    # idct(dct(f, type=2), type=1) / (2*Ny) is an exact re-interpolation between
    # the two standard Chebyshev point sets.
    U = idct(dct(u_ft, type=2, axis=1), type=1, axis=1) / (2 * Ny)
    V = idct(dct(v_ft, type=2, axis=1), type=1, axis=1) / (2 * Ny)
    T = idct(dct(T_ft, type=2, axis=1), type=1, axis=1) / (2 * Ny)
    W = idct(dct(w_ft, type=2, axis=1), type=1, axis=1) / (2 * Ny)

    # Step 3 — Gauss-Lobatto y grid matching the ROM convention
    Y, _ = cheb(Ny, Ly, 0)

    return X, Y, U, V, T, W, t


# =============================================================================
# EKF PREDICT / UPDATE
# =============================================================================

def estimate(ci_k, Pk, Q, dt, Ra, Pr, ROM, jac_partial):
    """EKF prediction step: propagate state and covariance forward by dt.

    Implements eqs. (24)–(26) of the paper:
        ĉ⁻_k  = Φ(ĉ_{k-1})          (nonlinear ROM integration, eq. 24)
        F_k   = I + J(ĉ_{k-1}) Δt   (first-order linearisation,  eq. 26)
        P⁻_k  = F_k P_{k-1} F_k^T + Q (covariance propagation,   eq. 25)
    """
    # Propagate state with RK45 integration of the ROM ODE (eq. 24)
    out    = solve_ivp(ROM,
                       [0, dt],
                       ci_k,
                       args=(Pr, Ra),
                       method='RK45')
    ci_hat = out.y[:, -1]

    # Linearised transition matrix F = I + J Δt  (eq. 26)
    J      = jac_partial(0, ci_k, Pr, Ra)
    n      = len(ci_k)
    F      = np.eye(n) + J * dt

    # Predicted covariance  P⁻ = F P F^T + Q  (eq. 25)
    Pk_hat = F @ Pk @ F.T + Q
    return ci_hat, Pk_hat


def update(ci_hat, Pk_hat, yk, Hk, R):
    """EKF update step: correct state estimate with observations yk.

    Implements eqs. (28)–(32) of the paper:
        z_k = y_k - H ĉ⁻_k                       (innovation,            eq. 28)
        S_k = H P⁻_k H^T + R                      (innovation covariance, eq. 29)
        K_k = P⁻_k H^T S_k^{-1}                   (Kalman gain,           eq. 30)
        ĉ_k = ĉ⁻_k + K_k z_k                      (posterior mean,        eq. 31)
        P_k = (I-K_kH) P⁻_k (I-K_kH)^T + K_kRK_k^T  (Joseph form,       eq. 32)
    """
    # Innovation (measurement residual)  eq. (28)
    zk   = yk - Hk @ ci_hat

    # Innovation covariance  S = H P⁻ H^T + R   eq. (29)
    S    = Hk @ Pk_hat @ Hk.T + R

    # Kalman gain  K = P⁻ H^T S^{-1}   eq. (30)
    K    = Pk_hat @ Hk.T @ np.linalg.inv(S)

    # Updated state estimate  ĉ = ĉ⁻ + K z   eq. (31)
    ci_k = ci_hat + K @ zk

    # Updated covariance — Joseph form for numerical stability   eq. (32)
    # P = (I - K H) P⁻ (I - K H)^T + K R K^T
    n    = len(ci_hat)
    IKH  = np.eye(n) - K @ Hk
    Pk   = IKH @ Pk_hat @ IKH.T + K @ R @ K.T
    return ci_k, Pk, K


# =============================================================================
# ROM IMPORT
# =============================================================================

def import_ROM(n_alpha, n, nx, ny, nmodes,
               ROM_Pr, ROM_Ra, ROM_g2, ROM_modes='con'):
    """
    Load a precomputed coupled ROM from HDF5 and return JAX-compiled functions.

    Parameters
    ----------
    n_alpha   : int   — number of Fourier wavenumbers
    n         : int   — number of Chebyshev modes per wavenumber
    nx        : int   — number of x grid points
    ny        : int   — number of y grid points
    nmodes    : int   — total number of ROM modes (n_alpha * n)
    ROM_Pr    : float — Prandtl number used to build ROM
    ROM_Ra    : float — Rayleigh number used to build ROM
    ROM_g2    : float — coupling weight γ² used to build ROM
    ROM_modes : str   — 'con' (controllability, default)

    Returns
    -------
    jaxROM, jaxJac, base, X, Y, DY, W, xx, yy, TT0
    """
    Pr_text = str(ROM_Pr).replace('.', 'p')
    Ra_text = str(ROM_Ra).replace('.', 'p')
    g2_text = str(ROM_g2).replace('.', 'p')
    name    = ('RB_coupled_nx{:03.0f}_ny{:03.0f}_mX{:02.0f}_mY{:02.0f}'
               '_N{:03.0f}_Pr{}_Ra{}_g2_{}.h5').format(
                   nx, ny, n_alpha, n, nmodes, Pr_text, Ra_text, g2_text)

    path       = 'coupled/ROM/'
    input_file = os.path.join(path, name)
    print('File name = {}'.format(name))

    with h5.File(input_file, 'r') as h5file:
        X      = h5file['X'][:]
        Y      = h5file['Y'][:]
        TT0    = h5file['TT0'][:]
        base   = h5file['base'][:]
        For0   = h5file['For0'][:]
        For1   = h5file['For1'][:]
        u_Diff = h5file['u_Diff'][:]
        Nlin   = h5file['Nlin'][:]
        Line   = h5file['Line'][:]
        T_Diff = h5file['T_Diff'][:]
    print('File successfully read from', input_file)

    # Domain
    Lx = 2;  Ly = 1;  Y0 = 0
    X      = np.linspace(0, Lx, nx, endpoint=False)
    Y, W   = clenshaw_curtis_compute(ny, Ly, Y0)
    Y, DY  = cheb(ny, Ly, Y0)
    xx, yy = np.meshgrid(X, Y)

    # Jacobian partial function
    jac_partial = partial(jac_coupled,
                          For1=For1, u_Diff=u_Diff,
                          T_Diff=T_Diff, Line=Line,
                          Nlin=Nlin)

    # ROM partial function
    ROM_partial = partial(ROM,
                          For0=For0, For1=For1,
                          u_Diff=u_Diff, T_Diff=T_Diff,
                          Line=Line, Nlin=Nlin)

    jaxROM = jax.jit(ROM_partial)
    jaxJac = jax.jit(jac_partial)

    return jaxROM, jaxJac, base, X, Y, DY, W, xx, yy, TT0


# =============================================================================
# DNS DATA LOADER (case-level)
# =============================================================================

def get_dns_data(r, RaS, PrS, rseed):
    """
    Load DNS data for a given case and trim away the initial transient
    (first third of the simulation).

    Parameters
    ----------
    r     : int/float — ratio Ra/Ra_c used to label the case
    RaS   : float     — Rayleigh number (passed to import_DNS)
    PrS   : float     — Prandtl number
    rseed : int       — random seed used in the DNS

    Returns
    -------
    t_dns, nx_DNS, ny_DNS, X_DNS, Y_DNS,
    ux_dns, uy_dns, T_dns, T0_dns, T1_dns, nt
    """
    Pr_text   = f'{PrS:03.1f}'.replace('.', 'p')
    r_text    = f'{r:03.0f}'.replace('.', 'p')
    case_path = f'DNS_kalman/simulations/DNS_Pr{Pr_text}_R{r_text}_rseed{rseed:03.0f}'

    X_DNS, Y_DNS, \
    ux_DNS, uy_DNS, theta_DNS, \
    vor_DNS, t_DNS         = import_DNS(RaS, PrS, path=case_path)
    nt_DNS, ny_DNS, nx_DNS = ux_DNS.shape
    T0_DNS  = (1 - Y_DNS)
    TT0_DNS = np.ones([ny_DNS, nx_DNS]) * T0_DNS.reshape([ny_DNS, 1])

    # Discard first third as transient
    init_time = t_DNS[-1] / 3
    t_filt    = t_DNS > init_time

    t_dns  = t_DNS[t_filt];       t_dns = t_dns - t_dns[0]
    ux_dns = ux_DNS[t_filt]
    uy_dns = uy_DNS[t_filt]
    T_dns  = theta_DNS[t_filt]
    T0_dns = TT0_DNS.reshape([1, ny_DNS, nx_DNS])
    T1_dns = theta_DNS[t_filt] - TT0_DNS
    nt     = len(t_dns)

    return t_dns, nx_DNS, ny_DNS, X_DNS, Y_DNS, ux_dns, uy_dns, T_dns, T0_dns, T1_dns, nt


# =============================================================================
# MODE UPSAMPLING
# =============================================================================

def upsample_modes(base, nx_DNS):
    """
    Upsample ROM basis from ROM x-resolution to DNS x-resolution via
    zero-padding in Fourier space (paper eq. 38).

    The ROM basis is computed on a coarse Fourier grid (nx_ROM points).
    To evaluate basis functions at DNS probe locations the modes are
    spectrally interpolated to nx_DNS points by zero-padding the real FFT
    and rescaling amplitudes by nx_DNS / nx_ROM.

    Parameters
    ----------
    base   : (3, nmodes, ny, nx_ROM)  — ROM basis (u, v, θ components)
    nx_DNS : int                       — target x resolution (DNS grid)

    Returns
    -------
    base_us : (3, nmodes, ny, nx_DNS)  — upsampled basis
    """
    # UPSAMPLE MODES TO DNS RESOLUTION
    _, nmodes, ny, nx = base.shape

    # Upsampling ratio between DNS and ROM grids
    x_ds         = nx_DNS / nx

    # Length of the UPSAMPLED real-FFT frequency dimension
    kx_rfft      = nx_DNS // 2 + 1

    # Allocate upsampled complex spectrum (zero-filled = zero-padding)
    base_us_rfft = np.zeros((3, nmodes, ny, kx_rfft), dtype='complex')

    # Downsampled real FFT of the ROM bases
    base_rfft    = np.fft.rfft(base, axis=-1)

    # Length of the DOWNSAMPLED real-FFT frequency dimension
    kx_ds_rfft   = base_rfft.shape[-1]

    # ZERO-PADDING: copy low-wavenumber coefficients and correct amplitude
    base_us_rfft[:, :, :, :kx_ds_rfft] = base_rfft * x_ds

    # IFFT to physical space on the DNS grid
    base_us = np.fft.irfft(base_us_rfft, axis=-1)
    return base_us


# =============================================================================
# OBSERVATION MATRIX
# =============================================================================

def coarse_grid(nrow, ncol, nx, ny):
    """
    Build coarse-grid probe indices on a (ny, nx) domain.

    Parameters
    ----------
    nrow : int — number of interior probe rows
    ncol : int — number of probes per row
    nx   : int — x grid size
    ny   : int — y grid size

    Returns
    -------
    indices_U, indices_V, indices_T : flat index arrays into ny*nx
    nUp, nVp, nTp                  : number of probes per field
    """
    spacing   = nx // ncol
    row_index = np.linspace(1, ny, nrow + 2, dtype='int')[1:-1]
    indices   = np.array([], dtype='int')
    for i in range(nrow):
        indi_line = (np.arange((row_index[i] - 1) * nx, row_index[i] * nx, spacing)
                     + spacing // 2)
        indices   = np.concatenate([indices, indi_line])

    indices_U = indices
    indices_V = indices
    indices_T = indices
    nUp = len(indices_U)
    nVp = len(indices_V)
    nTp = len(indices_T)
    return indices_U, indices_V, indices_T, nUp, nVp, nTp


def build_obs_mat(base_us, indices_U, indices_V, indices_T):
    """
    Build the linear observation matrix H mapping ROM modal coefficients to
    physical-space probe measurements (paper eqs. 21–22).

    The ROM approximation of each physical field is linear in the modal
    coefficients c:
        u(x) ≈ Σ_j c_j U_j(x)
        v(x) ≈ Σ_j c_j V_j(x)
        θ(x) ≈ Σ_j c_j Θ_j(x)

    The block observation matrix H (eq. 22) is therefore:
        H = [ H^u ]       H^u_{ℓj} = U_j(x_ℓ)   (velocity-x probes)
            [ H^v ]   ,   H^v_{pj} = V_j(x_p)   (velocity-y probes)
            [ H^θ ]       H^θ_{qj} = Θ_j(x_q)   (temperature probes)

    where x_ℓ, x_p, x_q are the probe locations for each field.
    The basis functions are evaluated on the upsampled DNS grid (after
    upsample_modes) so that probe indices are consistent with DNS resolution.

    Parameters
    ----------
    base_us  : (3, nmodes, ny, nx) — upsampled basis (u, v, θ components)
    indices_U : flat index array into ny*nx for u probes
    indices_V : flat index array into ny*nx for v probes
    indices_T : flat index array into ny*nx for θ probes

    Returns
    -------
    Hk      : (out_dim, nmodes)    observation matrix
    out_dim : int                  total number of scalar observations
    h_ci_u, h_ci_v, h_ci_T : (ny*nx, nmodes) full basis evaluation matrices
    """
    _, nmodes, ny, nx = base_us.shape
    h_ci_u = base_us[0].transpose([1, 2, 0]).reshape([ny * nx, nmodes])
    h_ci_v = base_us[1].transpose([1, 2, 0]).reshape([ny * nx, nmodes])
    h_ci_T = base_us[2].transpose([1, 2, 0]).reshape([ny * nx, nmodes])

    Hk_ci_u = h_ci_u[indices_U, :nmodes]
    Hk_ci_v = h_ci_v[indices_V, :nmodes]
    Hk_ci_T = h_ci_T[indices_T, :nmodes]

    Hk = np.block([[Hk_ci_u],
                   [Hk_ci_v],
                   [Hk_ci_T]])
    out_dim, _ = Hk.shape
    return Hk, out_dim, h_ci_u, h_ci_v, h_ci_T


# =============================================================================
# UTILITIES
# =============================================================================

def L2norm(field, W, Lx=2, Ly=1):
    """Weighted L2 integral of a 2D field (Clenshaw-Curtis in y, uniform in x)."""
    ny, nx   = field.shape
    integral = np.sum(Lx / nx * W.reshape([ny, 1]) * field)
    return integral


# =============================================================================
# ROM DYNAMICS
# =============================================================================

def ROM(t, Xi, Pr, Ra, For0, For1, u_Diff, T_Diff, Line, Nlin):
    """Coupled ROM RHS (dense tensor, compatible with JAX jit)."""
    dXidt = (Pr * (For0 + For1 @ Xi) +
             Pr / (Ra**0.5) * u_Diff @ Xi +
             1  / (Ra**0.5) * T_Diff @ Xi -
             Line @ Xi - Nlin @ Xi @ Xi)
    return dXidt


def ROM_sparse(t, Xi, Pr, Ra, For0, For1, u_Diff, T_Diff, Line, Nlin):
    """Coupled ROM RHS with sparse nonlinear contraction."""
    Xi_outer = np.outer(Xi, Xi).ravel()
    Nlin_eq  = Nlin.dot(Xi_outer)
    dXidt    = (Pr * (For0 + For1 @ Xi) +
                Pr / np.sqrt(Ra) * u_Diff @ Xi +
                1  / np.sqrt(Ra) * T_Diff @ Xi -
                Line @ Xi - Nlin_eq)
    return dXidt


def jac_coupled(t, y, Pr, Ra, For1, u_Diff, T_Diff, Line, Nlin):
    """Jacobian of the coupled ROM used in the EKF linearisation."""
    Jac = (Pr * For1 +
           Pr / (Ra**0.5) * u_Diff +
           1  / (Ra**0.5) * T_Diff -
           Line - Nlin @ y - np.transpose(Nlin, [0, 2, 1]) @ y)
    return Jac

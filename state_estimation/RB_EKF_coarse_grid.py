#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coarse-grid Extended Kalman Filter (EKF) for Rayleigh-Bénard state estimation.

Uses a coupled Galerkin ROM as the forecast model and assimilates sparse
velocity or temperature measurements at a coarse sensor grid to reconstruct
the full 2D flow field.

Run from the repo root:
    python state_estimation/RB_EKF_coarse_grid.py

Outputs are written to:
    state_estimation/filter_runs/

@author: efloresm
"""

import sys
import os

# Add repo root and this directory to path
_dir  = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_dir)
for _p in [_repo, _dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from FUN                 import cheb, clenshaw_curtis_compute, Inner_prod, linear_analysis
from RB_EKF_FUN          import (estimate, update, import_ROM, import_DNS,
                                  get_dns_data, upsample_modes, coarse_grid,
                                  build_obs_mat, L2norm)


# =============================================================================
# IMPORT ROM
# =============================================================================
n_alpha, n      = 6, 16
nx, ny          = 4 * (n_alpha - 1) + 2, 64
ROM_Pr, ROM_Ra  = 1, 1
ROM_g2          = 1.24
nmodes          = n_alpha * n
ndim            = nmodes
ROMmodes        = 'con'

jaxROM, jaxJac, base, \
X, Y, DY, W, xx, yy, TT0 = import_ROM(n_alpha, n, nx, ny, nmodes,
                                        ROM_Pr, ROM_Ra, ROM_g2, ROMmodes)

# =============================================================================
# IMPORT LINEAR ANALYSIS (Ra_c)
# =============================================================================
_, _, Ra_c = linear_analysis()

# =============================================================================
# IMPORT DNS DATA
# =============================================================================
r       = 120
Ra      = float(r * Ra_c)
Pr      = float(10)
rseed   = 1

t_dns, nx_DNS, ny_DNS,  \
X_DNS, Y_DNS,           \
ux_dns, uy_dns, T_dns,  \
T0_dns, T1_dns, nt      = get_dns_data(r, Ra, Pr, rseed)

# Time-step vector
dt           = np.diff(t_dns)
xx_DNS, yy_DNS = np.meshgrid(X_DNS, Y_DNS)

# =============================================================================
# UPSAMPLE MODES TO DNS RESOLUTION
# =============================================================================
base_us = upsample_modes(base, nx_DNS)

# =============================================================================
# PROJECT DNS DATA ONTO UPSAMPLED MODES
# =============================================================================
ci_DNS      = np.zeros([nmodes, nt])
norm_mat    = np.ones([3, ny_DNS, nx_DNS])
norm_mat[2] = ROM_g2
Xi          = np.zeros([3, ny_DNS, nx_DNS])

for i in range(nt):
    Xi[0] = ux_dns[i]
    Xi[1] = uy_dns[i]
    Xi[2] = T1_dns[i]
    for j in range(nmodes):
        ci_DNS[j, i] = Inner_prod(Xi, base_us[:, j], X_DNS, Y, W, norm_mat)

#%% =============================================================================
# KALMAN FILTER SETUP
# =============================================================================

# Observation strategy: 'UVgrid' (velocity probes) or 'Tgrid' (temperature probes)
obs_strat = 'grid'

indices_U, indices_V, \
indices_T, nUp, nVp, nTp = coarse_grid(4, 4, nx_DNS, ny)

if obs_strat == 'Tgrid':
    indices_U = []
    indices_V = []
    nUp = 0
    nVp = 0

if obs_strat == 'UVgrid':
    indices_T = []
    nTp = 0

# Observation matrix
Hk, out_dim, _, _, _ = build_obs_mat(base_us, indices_U, indices_V, indices_T)

# Initialise estimates
ci_hat = np.zeros_like(ci_DNS)   # a priori estimate
ci_k   = np.zeros_like(ci_DNS)   # a posteriori estimate

# Initialise covariance matrices
Pk_hat = 1e-3 * np.eye(ndim)
Pk     = 1e-3 * np.eye(ndim)

# Noise covariances
QR_ratio    = 0.01
Q           = QR_ratio * np.eye(ndim)
R           = 1.0      * np.eye(out_dim)

# =============================================================================
# PERFORMANCE QUANTIFICATION
# =============================================================================
yk               = np.zeros([nt, out_dim])
filter_error     = np.zeros(nt)
prediction_error = np.zeros(nt)
cov_trace        = np.zeros(nt)
L                = np.zeros([nt, ndim, out_dim])

prediction_error[0] = (np.linalg.norm(ci_hat[:, 0] - ci_DNS[:, 0]) /
                       np.linalg.norm(ci_DNS[:, 0]))
filter_error[0]     = (np.linalg.norm(ci_k[:, 0] - ci_DNS[:, 0]) /
                       np.linalg.norm(ci_DNS[:, 0]))

# Analysis mode: 'direct' (local measurements) or 'projection' (modal projection)
mode = 'direct'

# =============================================================================
# RUN FILTER
# =============================================================================
for i in range(1, nt):
    if mode == 'direct':
        observed_u          = (ux_dns[i].ravel())[indices_U]
        observed_v          = (uy_dns[i].ravel())[indices_V]
        observed_theta      = (T1_dns[i].ravel())[indices_T]
        yk[i]               = np.hstack([observed_u, observed_v, observed_theta])
    elif mode == 'projection':
        yk[i] = Hk @ ci_DNS[:, i]

    ci_hat[:, i], Pk_hat    = estimate(ci_k[:, i-1], Pk, Q, dt[i-1], Ra, Pr, jaxROM, jaxJac)
    ci_k[:, i], Pk, L[i]   = update(ci_hat[:, i], Pk_hat, yk[i], Hk, R)

    prediction_error[i] = (np.linalg.norm(ci_hat[:, i] - ci_DNS[:, i]) /
                           np.linalg.norm(ci_DNS[:, i]))
    filter_error[i]     = (np.linalg.norm(ci_k[:, i] - ci_DNS[:, i]) /
                           np.linalg.norm(ci_DNS[:, i]))
    cov_trace[i]        = np.trace(np.abs(Pk))

    print(f'It {i}: Pred error = {prediction_error[i]:7.2f}, '
          f'Filter error = {filter_error[i]:7.2f}, tr = {cov_trace[i]:<7.2f}')

# =============================================================================
# RECONSTRUCT FLOW FIELDS FROM FILTERED COEFFICIENTS
# =============================================================================
u_filter = np.zeros([nt, ny_DNS, nx_DNS])
v_filter = np.zeros([nt, ny_DNS, nx_DNS])
t_filter = np.ones( [nt, ny_DNS, nx_DNS]) * T0_dns

for it in range(nt):
    for j in range(nmodes):
        u_filter[it] += ci_k[j, it] * base_us[0, j]
        v_filter[it] += ci_k[j, it] * base_us[1, j]
        t_filter[it] += ci_k[j, it] * base_us[2, j]

# =============================================================================
# FIELD-LEVEL ERRORS
# =============================================================================
e_U = np.zeros(nt)
e_T = np.zeros(nt)

for it in range(nt):
    u_integrand = (ux_dns[it] - u_filter[it])**2 + (uy_dns[it] - v_filter[it])**2
    t_integrand = (T_dns[it]  - t_filter[it])**2

    e_U[it] = (L2norm(u_integrand, W)**0.5 /
               L2norm(ux_dns[it]**2 + uy_dns[it]**2, W)**0.5)
    e_T[it] = (L2norm(t_integrand, W)**0.5 /
               L2norm(T_dns[it]**2, W)**0.5)
    print(f'Vel error = {e_U[it]:.3f}  T error = {e_T[it]:.3f}')

# Time-averaged errors
E_prediction = np.trapz(prediction_error, t_dns) / t_dns[-1]
E_filter     = np.trapz(filter_error,     t_dns) / t_dns[-1]
E_U          = np.trapz(e_U,              t_dns) / t_dns[-1]
E_T          = np.trapz(e_T,              t_dns) / t_dns[-1]

#%% =============================================================================
# SAVE FILTER RESULTS
# =============================================================================
output_path = 'state_estimation/filter_runs/'
os.makedirs(output_path, exist_ok=True)
case_name   = (f'ROM{ROMmodes}_Ra{ROM_Ra:03.0f}_Pr{ROM_Pr:02.0f}_g2_{ROM_g2*100:3.0f}'
               f'_n{nmodes:03.0f}_'
               f'DNS_r{r:03.0f}_Pr{Pr:02.0f}_rseed{rseed:02.0f}_'
               f'filter_{mode}_m{out_dim:03.0f}_obs_{obs_strat}_QR{QR_ratio*1e6:09.0f}')
case_path_out = output_path + case_name
print(f'Saving results to = {case_path_out}')

with h5.File(case_path_out, 'w') as h5file:
    h5file.create_dataset('ROM_Ra',          data=ROM_Ra)
    h5file.create_dataset('ROM_Pr',          data=ROM_Pr)
    h5file.create_dataset('ROM_g2',          data=ROM_g2)
    h5file.create_dataset('nx',              data=nx)
    h5file.create_dataset('ny',              data=ny)
    h5file.create_dataset('nmodes',          data=nmodes)
    h5file.create_dataset('ndim',            data=ndim)
    h5file.create_dataset('ROMmodes',        data=ROMmodes)
    h5file.create_dataset('Ra_c',            data=Ra_c)
    h5file.create_dataset('r',               data=r)
    h5file.create_dataset('Ra',              data=Ra)
    h5file.create_dataset('Pr',              data=Pr)
    h5file.create_dataset('rseed',           data=rseed)
    h5file.create_dataset('nx_DNS',          data=nx_DNS)
    h5file.create_dataset('ny_DNS',          data=ny_DNS)
    h5file.create_dataset('nt',              data=nt)
    h5file.create_dataset('t_dns',           data=t_dns)
    h5file.create_dataset('ci_DNS',          data=ci_DNS)
    h5file.create_dataset('obs_strat',       data=obs_strat)
    h5file.create_dataset('indices_U',       data=indices_U)
    h5file.create_dataset('indices_V',       data=indices_V)
    h5file.create_dataset('indices_T',       data=indices_T)
    h5file.create_dataset('nUp',             data=nUp)
    h5file.create_dataset('nVp',             data=nVp)
    h5file.create_dataset('nTp',             data=nTp)
    h5file.create_dataset('out_dim',         data=out_dim)
    h5file.create_dataset('Hk',             data=Hk)
    h5file.create_dataset('QR_ratio',        data=QR_ratio)
    h5file.create_dataset('ci_hat',          data=ci_hat)
    h5file.create_dataset('ci_k',            data=ci_k)
    h5file.create_dataset('yk',              data=yk)
    h5file.create_dataset('cov_trace',       data=cov_trace)
    h5file.create_dataset('prediction_error',data=prediction_error)
    h5file.create_dataset('filter_error',    data=filter_error)
    h5file.create_dataset('vel_error',       data=e_U)
    h5file.create_dataset('tem_error',       data=e_T)
    h5file.create_dataset('E_prediction',    data=E_prediction)
    h5file.create_dataset('E_filter',        data=E_filter)
    h5file.create_dataset('E_vel',           data=E_U)
    h5file.create_dataset('E_tem',           data=E_T)

print('File successfully saved to ', case_path_out)


#%% =============================================================================
# PLOTTING
# =============================================================================
import matplotlib.gridspec as gridspec

params = {'backend': 'ps',
          'axes.labelsize': 20,
          'font.size': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True}
plt.rcParams.update(params)

# --- Probe locations ---
plt.close('all')
plt.figure(figsize=(7.5, 4.5), constrained_layout=True, rasterized=True)
plt.contourf(xx_DNS, yy_DNS, T_dns[-1], 51, cmap='RdBu_r')
plt.colorbar(shrink=0.7, ticks=[0, 0.5, 1], label='$\\theta$')
plt.plot(xx_DNS.ravel()[indices_U], yy_DNS.ravel()[indices_U], 'ko',
         mfc='none', markersize=12, label='$u_x$')
plt.plot(xx_DNS.ravel()[indices_V], yy_DNS.ravel()[indices_V], 'ks',
         mfc='none', markersize=8,  label='$u_y$')
plt.plot(xx_DNS.ravel()[indices_T], yy_DNS.ravel()[indices_T], 'kx',
         mfc='none', markersize=14, label="$\\theta'$")
plt.legend(ncols=3, bbox_to_anchor=(0.5, 1), loc='center', borderaxespad=0)
plt.axis('equal')
plt.xlabel('$x$');  plt.ylabel('$y$')
plt.xticks([0, 1, 2]);  plt.yticks([0, 0.5, 1])
plt.gca().set_frame_on(False)
print('Output dimension = {:5.0f}'.format(out_dim))


# --- Modal coefficients: broken-axis time series ---
def broken_ax(fig, gs_cell, t_start1, t_end1, t_start2, t_end2):
    """Return (ax_l, ax_r) broken-axis pair."""
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell,
                                             width_ratios=[1, 2], wspace=0.05)
    ax_l = fig.add_subplot(inner[0])
    ax_r = fig.add_subplot(inner[1], sharey=ax_l)
    ax_l.set_xlim([t_start1, t_end1])
    ax_r.set_xlim([t_start2, t_end2])
    ax_l.spines['right'].set_visible(False)
    ax_r.spines['left'].set_visible(False)
    ax_r.yaxis.set_visible(False)
    d = 0.015
    kw = dict(color='k', clip_on=False, lw=1)
    ax_l.plot((1-d, 1+d), (-d, +d),   transform=ax_l.transAxes, **kw)
    ax_l.plot((1-d, 1+d), (1-d, 1+d), transform=ax_l.transAxes, **kw)
    ax_r.plot((-d, +d),   (-d, +d),   transform=ax_r.transAxes, **kw)
    ax_r.plot((-d, +d),   (1-d, 1+d), transform=ax_r.transAxes, **kw)
    return ax_l, ax_r


c_sort   = np.flip(np.argsort(np.abs(ci_DNS[:nmodes, :]).mean(axis=1)))
modes2plot = 6
t_start1, t_end1 = t_dns[0],       t_dns[100]
t_start2, t_end2 = t_dns[nt-200],  t_dns[nt-1]

fig = plt.figure(figsize=(10.66, 4), constrained_layout=True)
gs  = gridspec.GridSpec(modes2plot // 2, 4, figure=fig,
                        width_ratios=[1, 2, 1, 2], wspace=0.05)

for j in range(modes2plot // 2):
    for k in range(2):
        idx   = c_sort[j * 2 + k]
        col_l = k * 2
        col_r = k * 2 + 1
        ax_l  = fig.add_subplot(gs[j, col_l])
        ax_r  = fig.add_subplot(gs[j, col_r], sharey=ax_l)

        for ax_, t0, t1 in [(ax_l, t_start1, t_end1), (ax_r, t_start2, t_end2)]:
            ax_.plot(t_dns, ci_DNS[idx, :].T, 'k-s', mfc='none', markevery=50,
                     label='$c_i$')
            ax_.plot(t_dns, ci_k[idx, :].T, '--o', mfc='none', color='tab:blue',
                     markevery=50, label='$\\hat{c}_i$')
            ax_.set_xlim([t0, t1])
            ax_.grid(True)

        ax_l.spines['right'].set_visible(False)
        ax_r.spines['left'].set_visible(False)
        ax_r.yaxis.set_visible(False)
        d = 0.03
        kw = dict(transform=ax_l.transAxes, color='k', clip_on=False, lw=1)
        ax_l.plot((1-d, 1+d), (-d, +d), **kw)
        ax_l.plot((1-d, 1+d), (1-d, 1+d), **kw)
        kw.update(transform=ax_r.transAxes)
        ax_r.plot((-d, +d), (-d, +d), **kw)
        ax_r.plot((-d, +d), (1-d, 1+d), **kw)

        ax_l.set_ylabel(f'$c_{{{idx:.0f}}}$')
        if j == 0 and k == 0:
            ax_l.legend(fontsize=20)
        if j == modes2plot // 2 - 1:
            ax_r.set_xlabel('$t$')
        else:
            ax_l.set_xticklabels([])
            ax_r.set_xticklabels([])

fig.suptitle(f'$Ra/Ra_c = {Ra/Ra_c:g}\\quad Pr = {Pr:g}$')

# --- Filter and prediction errors ---
fig1 = plt.figure(figsize=(10.66, 4), constrained_layout=True)
gs1  = gridspec.GridSpec(1, 1, figure=fig1)
ax_l, ax_r = broken_ax(fig1, gs1[0], t_start1, t_end1, t_start2, t_end2)
for ax_ in (ax_l, ax_r):
    ax_.plot(t_dns, filter_error,      'k-o',  mfc='none', markersize=10,
             markevery=50, label='$|c_k - \\hat{c}_k|/|c_k|$')
    ax_.plot(t_dns, prediction_error,  '--s',  mfc='none', markersize=10,
             markevery=50, label='$|c_k - \\hat{c}^-_k|/|c_k|$')
    ax_.plot(t_dns, e_U,               ':<',   mfc='none', markersize=10,
             markevery=50, label='$e_\\mathbf{u}$')
    ax_.plot(t_dns, e_T,               '-.>',  mfc='none', markersize=10,
             markevery=50, label='$e_\\theta$')
    ax_.set_ylim([0, 0.3])
    ax_.grid(True)
ax_l.set_ylabel('$e_k$')
ax_r.set_xlabel('$t$')
ax_r.legend(fontsize=20, ncols=2)
fig1.suptitle(f'$Ra/Ra_c = {Ra/Ra_c:g}\\quad Pr = {Pr:g}$')

# --- Covariance trace ---
fig2 = plt.figure(figsize=(10.66 / 2, 4), constrained_layout=True)
gs2  = gridspec.GridSpec(1, 1, figure=fig2)
ax_l, ax_r = broken_ax(fig2, gs2[0], t_start1, t_end1, t_start2, t_end2)
for ax_ in (ax_l, ax_r):
    ax_.plot(t_dns[:nt-3], cov_trace[:nt-3], 'k-o', mfc='none',
             markersize=10, markevery=50)
    ax_.grid(True)
ax_l.set_ylabel('$\\mathrm{trace}(P_k)$')
ax_r.set_xlabel('$t$')
fig2.suptitle(f'$Ra/Ra_c = {Ra/Ra_c:g}\\quad Pr = {Pr:g}$')

plt.show()

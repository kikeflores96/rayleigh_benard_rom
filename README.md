# Galerkin Reduced-Order Model for Two-Dimensional Rayleigh-Bénard Convection

> **Disclaimer:** This documentation may contain errors or inconsistencies with the original papers and the code. In case of doubt, always refer to the original papers and contact the authors. AI assistance has been used to create this documentation.

This repository contains the code and precomputed ROM data accompanying the papers:

> **Galerkin reduced-order model for two-dimensional Rayleigh-Bénard convection**  
> Enrique Flores-Montoya and André V. G. Cavalieri  
> *Submitted to Physical Review Fluids*, 2026

> **State estimation of Rayleigh-Bénard convection with reduced-order models**  
> Enrique Flores-Montoya, André F. C. da Silva, and André V. G. Cavalieri  
> *Submitted to Physical Review Fluids*, 2026

This repository implements two Galerkin Reduced-Order Models (ROMs) for 2D Rayleigh-Bénard (RB) convection with **no-slip walls**: an **uncoupled** approach that uses separate orthonormal bases for velocity and temperature, and a **coupled** approach that projects the equations onto a single basis combining velocity and temperature components. Orthonormal bases are obtained as eigenfunctions of the controllability Gramian of the linearized RB equations — no DNS snapshot data are required. Four models with different degrees of freedom are provided: **C96**, **C192**, **U96**, and **U192**.

---

## Table of Contents

1. [Physical problem](#1-physical-problem)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Quickstart: simulate from precomputed ROM data](#4-quickstart-simulate-from-precomputed-rom-data)
5. [Build the ROM from scratch](#5-build-the-rom-from-scratch)
6. [Direct numerical simulation (DNS)](#6-direct-numerical-simulation-dns)
7. [State estimation with the EKF](#7-state-estimation-with-the-ekf)
8. [Code overview](#8-code-overview)
9. [Precomputed ROM files](#9-precomputed-rom-files)
10. [Key parameters](#10-key-parameters)
11. [Citation](#11-citation)

---

## 1. Physical problem

A two-dimensional fluid layer bounded by two parallel **no-slip** walls is considered. Distances are normalized by the channel height $h$, so that the wall-normal direction spans $y \in [0, 1]$. The fluid is heated from below ($y=0$) and cooled from above ($y=1$), and the domain is periodic in $x$ with period $L_x = 2$. The dimensionless mass, momentum, and energy conservation equations for a Boussinesq fluid are:

$$\nabla \cdot \mathbf{u} = 0,$$

$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\nabla p + Pr\,\theta\,\mathbf{e}_y + \frac{Pr}{\sqrt{Ra}}\,\nabla^2 \mathbf{u},$$

$$\frac{\partial \theta}{\partial t} + \mathbf{u} \cdot \nabla \theta = \frac{1}{\sqrt{Ra}}\,\nabla^2 \theta,$$

where $\mathbf{u}$, $\theta$, and $p$ are the dimensionless velocity, temperature perturbation, and driven pressure. The velocity is normalized by $\kappa\sqrt{Ra}/h$, where $\kappa$ is the thermal diffusivity. The temperature is nondimensionalized as $\theta = (T - T_0)/\Delta T$, where $\Delta T = T_1 - T_0$, with $T_1$ and $T_0$ the temperatures imposed at the bottom and top walls, respectively. The Rayleigh number is defined as

$$Ra = \frac{\sigma \Delta T\, g\, h^3}{\nu \kappa},$$

where $\sigma$ is the thermal expansion coefficient, $g$ is gravitational acceleration, and $\nu$ is the kinematic viscosity. The term $Pr\,\theta\,\mathbf{e}_y$ represents the upward buoyancy force that couples momentum and energy equations.

**Boundary conditions** at $y = 0$ and $y = 1$:
- No-slip velocity: $\mathbf{u} = 0$
- Fixed temperature perturbation: $\theta = 0$

**Governing parameters.** Two nondimensional parameters govern the dynamics: the Prandtl number $Pr = \nu/\kappa$ and the Rayleigh number $Ra$. For no-slip boundary conditions, the critical Rayleigh number is $Ra_c = 1707.8$ at a critical wavenumber $k_c \approx 3.117$. Because $k_c \approx \pi$, setting $L_x = 2$ gives a domain whose fundamental wavenumber is $\alpha = \pi$, yielding a critical Rayleigh number that closely matches the theoretical prediction.

The baseline conductive temperature profile is $\theta_0(y) = 1 - y$, and temperature modes represent perturbations $\theta'$ about this base state: $\theta(t,\mathbf{x}) = \theta_0(y) + \theta'(t,\mathbf{x})$.

---

## 2. Repository structure

```
rayleigh_benard_ROM/
│
├── FUN.py                              # Shared spectral utilities (Chebyshev, quadrature, inner products, operators)
├── RB_ROM_environment.yaml             # Conda environment for ROM and EKF scripts
├── dns_dedalus3_environment.yaml       # Separate Conda environment for Dedalus 3 DNS
│
├── uncoupled/                          # Uncoupled ROM — separate velocity & temperature bases
│   ├── RB_uncoupled_FUN.py             # Uncoupled ROM functions (modes, projection, ODE RHS, Jacobian)
│   ├── RB_uncoupled_ROM.py             # Build ROM from scratch and run a test simulation
│   ├── RB_uncoupled_ROM_parallel.py    # Parallel ROM build (multiprocessing)
│   ├── RB_uncoupled_simulate.py        # Load precomputed ROM → simulate → visualise
│   └── ROM/                            # Precomputed ROM coefficient files (.h5)
│
├── coupled/                            # Coupled ROM — joint velocity-temperature basis
│   ├── RB_coupled_FUN.py               # Coupled ROM functions (modes, projection, ODE RHS, Jacobian)
│   ├── RB_coupled_ROM.py               # Build ROM from scratch and run a test simulation
│   ├── RB_coupled_ROM_parallel.py      # Parallel ROM build (multiprocessing)
│   ├── RB_coupled_simulate.py          # Load precomputed ROM → simulate → visualise
│   └── ROM/                            # Precomputed ROM coefficient files (.h5)
│
├── DNS_kalman/                         # Dedalus 3 DNS (requires dns_dedalus3_environment.yaml)
│   ├── RB_DNS_dedalus3.py              # Run a DNS and save HDF5 output
│   └── simulations/                    # Precomputed DNS snapshots (HDF5 segment files)
│       ├── DNS_Pr10p0_R040_rseed001/
│       ├── DNS_Pr10p0_R080_rseed001/
│       ├── DNS_Pr10p0_R120_rseed001/
│       ├── DNS_Pr10p0_R120_rseed002/
│       ├── DNS_Pr10p0_R120_rseed003/
│       └── DNS_Pr10p0_R120_rseed004/
│
└── state_estimation/                   # Extended Kalman Filter state estimation
    ├── RB_EKF_FUN.py                   # EKF functions (predict, update, import_DNS, import_ROM, …)
    ├── RB_EKF_coarse_grid.py           # Run EKF with coarse-grid velocity/temperature sensors
    └── filter_runs/                    # EKF output files (.h5)
```

All scripts must be run **from the repository root** (`rayleigh_benard_ROM/`), so that `FUN.py` and all sub-packages are on the Python path. The `state_estimation/` scripts insert the repo root into `sys.path` automatically.

---

## 3. Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.13

### ROM and state estimation environment

```bash
conda env create -f RB_ROM_environment.yaml
conda activate galerkin
```

The environment installs all required packages. The key numerical dependencies are:

| Package | Role |
|---------|------|
| `numpy` | Array operations and FFT |
| `scipy` | ODE integration (`solve_ivp`), Sylvester equation (`solve_sylvester`), sparse matrices |
| `matplotlib` | Plotting and frame export |
| `h5py` | Reading and writing ROM and filter coefficient files |
| `jax` | JIT-compiled ROM evaluation and Jacobian in the EKF |
| `mpi4py` | Parallel ROM build (optional) |

### DNS environment (Dedalus 3)

Running new DNS simulations requires a **separate** Conda environment because Dedalus 3 has specific dependency requirements that conflict with the main environment.

```bash
conda env create -f dns_dedalus3_environment.yaml
conda activate dns_dedalus3
```

This environment is only needed to run `DNS_kalman/RB_DNS_dedalus3.py`. All ROM simulation and EKF scripts use the standard `galerkin` environment.

---

## 4. Quickstart: simulate from precomputed ROM data

The fastest way to get started is to load a precomputed ROM and run a simulation. All ROM tensor coefficients have been pre-projected and are stored in the `ROM/` directories. Four models are provided: **C96**, **C192** (coupled) and **U96**, **U192** (uncoupled) — see [§7](#7-precomputed-rom-files) for details.

> **Key property.** Because $Ra$ and $Pr$ appear only as scalar prefactors in the projected ODE, the same ROM tensor set can be used for any $(Ra, Pr)$ at no additional computational cost. The precomputed uncoupled tensors are independent of $Ra$ and $Pr$. The coupled tensors depend on the generation parameters $(Ra_\mathrm{ROM}, Pr_\mathrm{ROM}, \gamma^2)$ listed in the table below.

### Uncoupled ROM (U96 or U192)

```bash
cd rayleigh_benard_ROM
python uncoupled/RB_uncoupled_simulate.py
```

The script loads the precomputed U192 ROM ($n_\alpha = 8$ wavenumbers, $n_\beta = 12$ modes per wavenumber, 192 total degrees of freedom), integrates the ROM ODE for the chosen $(Ra, Pr)$, reconstructs the physical-space vorticity and temperature fields, and exports time snapshots.

Key parameters at the top of the script:

```python
n_alpha, n  = 8, 12            # wavenumbers × modes per wavenumber  →  U192
Pr          = 10               # Prandtl number
Ra          = 200 * 1707.8     # Rayleigh number  (Ra_c = 1707.8 for no-slip)
tend        = 500/np.sqrt(Pr)  # integration end time
```

To use **U96** instead, set `n_alpha, n = 6, 8` and point to the corresponding `.h5` file.

### Coupled ROM (C96 or C192)

```bash
cd rayleigh_benard_ROM
python coupled/RB_coupled_simulate.py
```

The coupled ROM uses a single joint basis $\{\boldsymbol{\chi}_i\}$ containing both velocity and temperature components, scaled by a common amplitude coefficient. The state vector is $\mathbf{X} = [u,\, v,\, \theta]^T$.

Key parameters at the top of the script:

```python
n_alpha, n      = 6, 16    # wavenumbers × modes per wavenumber  →  C96
ROM_Pr, ROM_Ra  = 1, 1     # Ra and Pr used to generate the ROM basis
ROM_g2          = 1.24     # temperature energy scaling factor γ²
Pr              = 1        # simulation Prandtl number
Ra              = 1500     # simulation Rayleigh number
```

To use **C192** instead, set `n_alpha, n = 8, 24` and point to the corresponding `.h5` file.

---

## 5. Build the ROM from scratch

If you want to recompute the ROM tensors for a different resolution or domain, use the build scripts.

### Uncoupled ROM

```bash
python uncoupled/RB_uncoupled_ROM.py
```

The script performs the following pipeline:

1. **Build the velocity basis** — for each horizontal wavenumber $k_x = j\alpha$ ($j = 0, 1, \ldots, n_\alpha - 1$, $\alpha = 2\pi/L_x$), compute Stokes modes as eigenfunctions of the controllability Gramian of the streamfunction equation $\nabla^{-2}\nabla^4 \Psi = \lambda \Psi$. These modes inherit the no-slip boundary conditions $\tilde\Psi(0) = \tilde\Psi(1) = 0$ and $\partial_y\tilde\Psi(0) = \partial_y\tilde\Psi(1) = 0$.
2. **Build the temperature basis** — compute thermal diffusion modes as eigenfunctions of the Gramian of the Laplacian, satisfying $\tilde\theta(0) = \tilde\theta(1) = 0$.
3. **Normalise** both bases to be orthonormal under the $L^2$ volume-averaged inner product.
4. **Project** the Navier-Stokes and energy equations to form the ROM tensor coefficients: baseline forcing ($f^u_i$, $F^u_{ij}$), diffusion ($D^u_{ij}$, $D^\theta_{ij}$), linear advection ($L^\theta_{ij}$), and the nonlinear triadic tensor ($N^u_{ijk}$, $N^\theta_{ijk}$). Only triads satisfying $k_i + k_j = k_k$ (and cyclic permutations) are computed, reducing the number of nonlinear projections by up to 83% for U192.
5. **Save** all ROM data to an `.h5` file in `uncoupled/ROM/`.

For parallel computation of the projection (recommended for $n > 100$):

```bash
python uncoupled/RB_uncoupled_ROM_parallel.py
```

### Coupled ROM

```bash
python coupled/RB_coupled_ROM.py
```

1. **Build the coupled basis** — for each non-zero wavenumber, solve the Sylvester equation $A\Phi + \Phi A^{+} + BB^{+} = 0$ for the controllability Gramian $\Phi$, where $A$ is the linearized RB state matrix and $A^{+}$ is its adjoint under the weighted inner product with weight $\gamma^2$. The state vector is $\mathbf{z} = [\tilde{\Psi},\ \tilde{\theta}]^{T}$. The direct state matrix is

$$A = \begin{bmatrix}
\dfrac{Pr}{\sqrt{Ra}}\nabla^{-2}\nabla^4 & -ik_x Pr\,\nabla^{-2} \\
ik_x \partial_y\theta_0 & \dfrac{1}{\sqrt{Ra}}\nabla^2
\end{bmatrix}$$

   and the adjoint state matrix is

$$A^{+} = \begin{bmatrix}
\dfrac{Pr}{\sqrt{Ra}}\nabla^{-2}\nabla^4 & i\gamma^2 k_x \nabla^{-2}\partial_y\theta_0 \\
-\dfrac{ik_x Pr}{\gamma^2} & \dfrac{1}{\sqrt{Ra}}\nabla^2
\end{bmatrix}.$$

Each eigenfunction $\mathbf{z}_\lambda$ of the Gramian is mapped to a physical mode via the observation matrix $C$, which extracts velocity and temperature from the streamfunction:

$$\tilde{\boldsymbol{\chi}}_\lambda = C\,\mathbf{z}_\lambda = \begin{bmatrix} \partial_y \tilde{\Psi}_\lambda \\ -ik_x \tilde{\Psi}_\lambda \\ \tilde{\theta}_\lambda \end{bmatrix} = \begin{bmatrix} \tilde{u} \\ \tilde{v} \\ \tilde{\theta} \end{bmatrix}.$$

This ensures that every mode satisfies the continuity equation and the no-slip boundary conditions by construction. For $k_x = 0$, the off-diagonal coupling terms in $A$ and $A^+$ vanish and decoupled Stokes/diffusion modes are used instead.

2. **Normalise** under the weighted inner product

$$\langle \boldsymbol{\chi}_i, \boldsymbol{\chi}_j \rangle_c = \frac{1}{L_x L_y}\int_0^{L_x}\int_0^{L_y}\left(u_i u_j + v_i v_j + \gamma^2 \theta_i \theta_j\right)dy\,dx,$$

   with $\gamma^2 = 1.24$ determined from DNS energy ratios.

3. **Project** and **save** as in the uncoupled case, but with the unified state vector $\boldsymbol{\chi}_i = [u_i,\, v_i,\, \theta_i]^T$.

---

## 6. Direct numerical simulation (DNS)

The folder `DNS_kalman/` contains the Dedalus 3 DNS script and a set of precomputed simulation snapshots used to validate the ROM and provide measurements for the state estimation. Dedalus 3 is not compatible with the main `galerkin` conda environment and requires a **separate environment**.

### Environment setup

```bash
conda env create -f dns_dedalus3_environment.yaml
conda activate dns_dedalus3
```

### Running a DNS

```bash
conda activate dns_dedalus3
python DNS_kalman/RB_DNS_dedalus3.py
```

Edit the parameters at the top of the script:

```python
Pr    = 10       # Prandtl number
r     = 80       # ratio Ra/Ra_c  (Ra_c = 1707.7651913068134 for no-slip)
Ra    = r*Ra_c
rseed = 1        # random seed for the initial perturbation
```

Output is written to `DNS_kalman/simulations/DNS_Pr{Pr}_R{r}_rseed{rseed}/` as a sequence of HDF5 segment files (`_s1.h5`, `_s2.h5`, …).

### Precomputed simulation data

Six DNS cases are included at $Pr = 10$ for different Rayleigh numbers and random seeds:

| Case folder | $r = Ra/Ra_c$ | rseed |
|-------------|:-------------:|:-----:|
| `DNS_Pr10p0_R040_rseed001` | 40 | 1 |
| `DNS_Pr10p0_R080_rseed001` | 80 | 1 |
| `DNS_Pr10p0_R120_rseed001` | 120 | 1 |
| `DNS_Pr10p0_R120_rseed002` | 120 | 2 |
| `DNS_Pr10p0_R120_rseed003` | 120 | 3 |
| `DNS_Pr10p0_R120_rseed004` | 120 | 4 |

Each case contains 6 segments, with 500 snapshots per segment ($n_t = 2500$ total over $t \in [0,\, 1500/\sqrt{Pr}]$). The domain is $L_x = 2$, $L_y = 1$ discretised with $N_x = 128$, $N_y = 64$ modes and $3/2$-dealiasing. The SBDF2 scheme is used for time integration.

---

## 7. State estimation with the EKF

The `state_estimation/` folder implements an Extended Kalman Filter (EKF) that combines the coupled Galerkin ROM as a forecast model with sparse physical-space measurements to reconstruct the full 2D flow field.

### Algorithm

The EKF alternates between a **prediction step** that advances the ROM state and covariance forward in time, and an **update step** that assimilates sensor measurements to correct the estimate.

**Prediction (ROM forecast):**

$$\hat{\mathbf{c}}^-_k = \mathcal{F}(\mathbf{c}_{k-1},\, \Delta t), \qquad P^-_k = F_k\,P_{k-1}\,F_k^T + Q,$$

where $\mathcal{F}$ integrates the coupled ROM ODE using RK45, $F_k = I + J(\mathbf{c}_{k-1})\,\Delta t$ is the linearised state transition matrix, $J$ is the analytical ROM Jacobian, and $Q$ is the model noise covariance.

**Update (sensor assimilation):**

$$K_k = P^-_k H^T(H P^-_k H^T + R)^{-1}, \qquad \mathbf{c}_k = \hat{\mathbf{c}}^-_k + K_k(\mathbf{y}_k - H\hat{\mathbf{c}}^-_k),$$

where $H$ is the observation matrix mapping ROM coefficients to physical-space sensor readings, $R$ is the measurement noise covariance, and $\mathbf{y}_k$ is the vector of observations. The Joseph form is used for the covariance update.

### Observation strategies

Two sensing strategies are available in `RB_EKF_coarse_grid.py`:

- **`UVgrid`** — assimilates velocity measurements ($u_x$, $u_y$) at a coarse $4 \times 4$ sensor grid.
- **`Tgrid`** — assimilates temperature measurements ($\theta'$) at the same grid.

### Running the EKF

```bash
conda activate galerkin
python state_estimation/RB_EKF_coarse_grid.py
```

Key parameters at the top of the script:

```python
n_alpha, n   = 6, 16    # coupled ROM (C96: 6 wavenumbers × 16 modes)
ROM_Pr, ROM_Ra, ROM_g2 = 1, 1, 1.24   # ROM generation parameters
r, Pr, rseed = 120, 10, 1             # DNS case to assimilate
obs_strat    = 'UVgrid'               # observation strategy
QR_ratio     = 0.01                   # model-to-sensor noise ratio Q/R
```

Results are written to `state_estimation/filter_runs/` as HDF5 files containing the filtered/predicted coefficients, error timeseries, and covariance trace.

Three precomputed demonstration runs are included in `state_estimation/filter_runs/`, covering different observation strategies (velocity-only, temperature-only, and combined) for the $r = 120$ DNS case.

---

## 8. Code overview

### `FUN.py` — shared spectral utilities

| Function | Description |
|----------|-------------|
| `cheb(n, Lx, x0)` | Chebyshev differentiation matrix and grid for $n$ points |
| `clenshaw_curtis_compute(n, Lx, x0)` | Clenshaw-Curtis quadrature weights for $L^2$ integration over the Chebyshev grid |
| `Inner_prod(f, g, x, y, W)` | Volume-averaged $L^2$ inner product $\langle f, g \rangle$ |
| `normalize_modes(base, X, Y, W)` | Orthonormalise a modal basis and report maximum non-orthogonality |
| `Lapl_2D(f, X, Y, DY2)` | 2D Laplacian $\nabla^2 f$ (Fourier in $x$, Chebyshev in $y$) |
| `grad(f, X, Y, DY)` | 2D gradient $(\partial_x f,\, \partial_y f)$ |
| `CONV(Xj, Vk, X, Y, DY)` | Advection $(\mathbf{V}_k \cdot \nabla)\mathbf{X}_j$ |
| `linear_analysis()` | Newton solver for the neutral stability curve $Ra_c(k)$ |

### `uncoupled/RB_uncoupled_FUN.py`

| Function | Description |
|----------|-------------|
| `vel_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n)` | Stokes velocity modes from the biharmonic Gramian |
| `temp_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n)` | Temperature modes from the Laplacian Gramian |
| `process_mode(i, n, nmodes, Ubase, Tbase, ...)` | Projects equations onto mode $i$; returns rows of all ROM tensors |
| `ROM_sparse(t, ci, Pr, Ra, ...)` | ODE right-hand side with sparse triadic nonlinear evaluation |
| `jac_uncoupled(t, ci, Pr, Ra, ...)` | Analytical Jacobian (for stiff LSODA/Radau integrators) |

The uncoupled ROM consists of the following coupled ODE system for velocity coefficients $a_i$ and temperature coefficients $b_i$:

$$\dot{a}_i + \sum_{j,k} N^u_{ijk}\,a_k\,a_j = Pr\left(F^u_i + \sum_j F^u_{ij}\,b_j\right) + \frac{Pr}{\sqrt{Ra}}\sum_j D^u_{ij}\,a_j$$

$$\dot{b}_i + \sum_j L^\theta_{ij}\,a_j + \sum_{j,k} N^\theta_{ijk}\,a_k\,b_j = \frac{1}{\sqrt{Ra}}\sum_j D^\theta_{ij}\,b_j$$

The total number of degrees of freedom is $n = 2N$, where $N = n_\alpha \times n_\beta$ is the size of each independent basis.

### `coupled/RB_coupled_FUN.py`

| Function | Description |
|----------|-------------|
| `coupled_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n, g2)` | Coupled velocity–temperature modes from the generalised Gramian |
| `vel_modes(...)` | Stokes modes (used for $k_x = 0$) |
| `temp_modes(...)` | Thermal diffusion modes (used for $k_x = 0$) |
| `process_mode(i, n, nmodes, base, ...)` | Projects equations onto mode $i$ |
| `ROM_sparse(t, Xi, Pr, Ra, ...)` | ODE right-hand side for the coupled system |
| `jac_coupled(t, y, Pr, Ra, ...)` | Analytical Jacobian |

The coupled ROM ODE for the unified amplitude coefficients $c_i$ (with $n = N$ degrees of freedom) reads:

$$\dot{c}_i + \sum_j L^\chi_{ij}\,c_j + \sum_{j,k} N^\chi_{ijk}\,c_k\,c_j = Pr\left(F^\chi_i + \sum_j F^\chi_{ij}\,c_j\right) + \frac{Pr}{\sqrt{Ra}}\sum_j D^V_{ij}\,c_j + \frac{1}{\sqrt{Ra}}\sum_j D^T_{ij}\,c_j$$

### `state_estimation/RB_EKF_FUN.py`

| Function | Description |
|----------|-------------|
| `import_DNS(Ra, Pr, path)` | Load and concatenate Dedalus 3 HDF5 segment files; returns grids and field arrays in `(nt, ny, nx)` layout. Re-interpolates from Gauss–Chebyshev (Dedalus ChebyshevT) to Gauss–Lobatto points via a DCT-II / IDCT-I pair |
| `estimate(ci_k, Pk, Q, dt, Ra, Pr, ROM, jac)` | EKF prediction step: RK45 forecast + linearised covariance propagation |
| `update(ci_hat, Pk_hat, yk, Hk, R)` | EKF update step: Kalman gain, state correction, Joseph-form covariance update |
| `import_ROM(n_alpha, n, nx, ny, nmodes, …)` | Load coupled ROM from `coupled/ROM/`, return JAX-compiled ROM and Jacobian functions |
| `get_dns_data(r, Ra, Pr, rseed)` | Load DNS case and trim the initial transient (first third of simulation) |
| `upsample_modes(base, nx_DNS)` | Zero-pad ROM basis in Fourier space to upsample from ROM to DNS x-resolution |
| `coarse_grid(nrow, ncol, nx, ny)` | Generate flat probe indices for a regular coarse sensor grid |
| `build_obs_mat(base_us, …)` | Assemble the linear observation matrix $H$ from upsampled basis functions |
| `ROM(t, Xi, Pr, Ra, …)` | Coupled ROM RHS (dense tensor form, JAX-compatible) |
| `ROM_sparse(t, Xi, Pr, Ra, …)` | Coupled ROM RHS with sparse nonlinear contraction |
| `jac_coupled(t, y, Pr, Ra, …)` | Analytical Jacobian for the EKF linearisation |

---

## 9. Precomputed ROM files

Four ROMs are provided, following the nomenclature of Table I in the companion paper. The **degrees of freedom** (DoF) $n$ is the total dimension of the ODE system: $n = 2N$ for uncoupled ROMs (two independent bases) and $n = N$ for coupled ROMs (one joint basis).

| Name | Type | $n_\alpha$ | $n_\beta$ | $N$ | DoF $n$ | $n_x$ | $n_y$ | $Ra_\mathrm{ROM}$ | $Pr_\mathrm{ROM}$ | $\gamma^2$ |
|------|------|-----------|----------|-----|---------|--------|--------|----------|----------|--------|
| **U96** | Uncoupled | 6 | 8 | 48 | 96 | 22 | 64 | N/A | N/A | N/A |
| **U192** | Uncoupled | 8 | 12 | 96 | 192 | 30 | 64 | N/A | N/A | N/A |
| **C96** | Coupled | 6 | 16 | 96 | 96 | 22 | 64 | 1 | 1 | 1.24 |
| **C192** | Coupled | 8 | 24 | 192 | 192 | 30 | 64 | 1 | 1 | 1.24 |

**N/A** — uncoupled bases are independent of $Ra$ and $Pr$.

### File naming convention

```
# Uncoupled
RB_uncoupled_nx{NX}_ny{NY}_mX{n_alpha}_mY{n_beta}_N{N}.h5

# Coupled
RB_coupled_nx{NX}_ny{NY}_mX{n_alpha}_mY{n_beta}_N{N}_Pr{Pr}_Ra{Ra}_g2_{g2}.h5
```

Note: `N` in uncoupled file names refers to the size of each independent basis ($N = n_\alpha \times n_\beta$), so the total DoF is $n = 2N$.

### HDF5 file contents

Each `.h5` file stores:

| Dataset | Uncoupled | Coupled | Description |
|---------|-----------|---------|-------------|
| `X`, `Y` | ✓ | ✓ | Grid points |
| `TT0` | ✓ | ✓ | Baseline temperature field $\theta_0(y)$ on the 2D grid |
| `Ubase`, `Tbase` | ✓ | — | Velocity and temperature modal bases |
| `base` | — | ✓ | Coupled modal basis $[u, v, \theta]$ |
| `u_For0`, `u_For1` | ✓ | — | Forcing tensors $F^u_i$, $F^u_{ij}$ |
| `u_Diff`, `T_Diff` | ✓ | ✓ | Diffusion tensors $D^u_{ij}$, $D^\theta_{ij}$ |
| `T_Line` | ✓ | — | Linear advection tensor $L^\theta_{ij}$ |
| `u_Nlin`, `T_Nlin` | ✓ | — | Nonlinear tensors $N^u_{ijk}$, $N^\theta_{ijk}$ |
| `For0`, `For1` | — | ✓ | Forcing tensors $F^\chi_i$, $F^\chi_{ij}$ |
| `Line` | — | ✓ | Linear advection tensor $L^\chi_{ij}$ |
| `Nlin` | — | ✓ | Nonlinear tensor $N^\chi_{ijk}$ |

---

## 10. Key parameters

| Parameter | Symbol | Values used | Description |
|-----------|--------|-------------|-------------|
| `n_alpha` | $n_\alpha$ | 6, 8 | Number of retained horizontal wavenumbers |
| `n` (per wn) | $n_\beta$ | 8, 12 (uncoupled); 16, 24 (coupled) | Modes per wavenumber |
| DoF | $n$ | 96, 192 | Total ODE system dimension |
| `Pr` | $Pr$ | 1, 10 | Prandtl number |
| `Ra` | $Ra$ | $Ra_c$–$500\,Ra_c$ | Rayleigh number ($Ra_c = 1707.8$ for no-slip) |
| `g2` | $\gamma^2$ | 1.24 | Temperature energy weight in coupled inner product |
| `nx`, `ny` | $n_x$, $n_y$ | 22/30, 64 | Physical grid size (Fourier × Chebyshev) |

The Chebyshev grid uses $n_y = 64$ collocation points in $y \in [0, 1]$. No-slip and fixed-temperature boundary conditions are enforced by removing the boundary rows from the discretized operators, so that all modes automatically satisfy $\tilde\Psi(0) = \tilde\Psi(1) = \partial_y\tilde\Psi(0) = \partial_y\tilde\Psi(1) = \tilde\theta(0) = \tilde\theta(1) = 0$.

The number of Fourier points in $x$ is set to $n_x = 4(n_\alpha - 1) + 2$ to resolve the nonlinear convolutions of the highest-wavenumber modes without aliasing.

---

## 11. Citation

If you use this code or data, please cite:

```bibtex
@article{FloresMontoya2026_ROM,
  author  = {Flores-Montoya, Enrique and Cavalieri, André V. G.},
  title   = {Galerkin reduced-order model for two-dimensional {Rayleigh-B\'enard} convection},
  journal = {Physical Review Fluids},
  note    = {Submitted},
  year    = {2026}
}

@article{FloresMontoya2026_EKF,
  author  = {Flores-Montoya, Enrique and da Silva, André F. C. and Cavalieri, André V. G.},
  title   = {State estimation of {Rayleigh-B\'enard} convection with reduced-order models},
  journal = {Physical Review Fluids},
  note    = {Submitted},
  year    = {2026}
}
```

---

## Contact

Enrique Flores-Montoya — efloresm.ca@gmail.com

# Galerkin Reduced-Order Model for Rayleigh-Bénard Convection

This repository contains the code and precomputed ROM data accompanying the paper:

> **[Paper title]**  
> Enrique Flores-Montoya et al.  
> *[Journal name]*, [Year]

It implements two Galerkin Reduced-Order Models (ROMs) for 2D Rayleigh-Bénard (RB) convection: an **uncoupled** approach that uses separate orthonormal bases for velocity and temperature, and a **coupled** approach that uses a joint basis derived from a controllability Gramian. Both ROMs share a common spectral discretisation built on Fourier modes in the horizontal direction and Chebyshev polynomials in the vertical direction.

---

## Table of Contents

1. [Physical problem](#1-physical-problem)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Quickstart: simulate from precomputed ROM data](#4-quickstart-simulate-from-precomputed-rom-data)
5. [Build the ROM from scratch](#5-build-the-rom-from-scratch)
6. [Code overview](#6-code-overview)
7. [Precomputed ROM files](#7-precomputed-rom-files)
8. [Key parameters](#8-key-parameters)
9. [Citation](#9-citation)

---

## 1. Physical problem

We study 2D Rayleigh-Bénard convection: a fluid layer of height $L_y = 1$ and horizontal period $L_x = 2$, heated from below and cooled from above. The governing equations are nondimensionalised with the convective velocity scale $U_c = \sqrt{Ra}\,\kappa / L_y$, giving

$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + \frac{Pr}{\sqrt{Ra}}\,\nabla^2\mathbf{u} + Pr\,\theta\,\hat{\mathbf{e}}_y, \qquad \nabla\cdot\mathbf{u}=0,$$

$$\frac{\partial \theta}{\partial t} + \mathbf{u}\cdot\nabla\theta = \frac{1}{\sqrt{Ra}}\,\nabla^2\theta - v\,\frac{dT_0}{dy},$$

where $Ra$ is the Rayleigh number, $Pr$ is the Prandtl number, $\theta$ is the temperature perturbation about the conductive base state $T_0(y)=1-y$, and $\hat{\mathbf{e}}_y$ is the unit vertical vector. The critical Rayleigh number for this geometry is $Ra_c \approx 657.5$.

---

## 2. Repository structure

```
rayleigh_benard_ROM/
│
├── FUN.py                           # Shared spectral utilities (Chebyshev, quadrature, inner products, operators)
├── RB_ROM_environment.yaml          # Conda environment specification
│
├── uncoupled/                       # Uncoupled ROM (separate velocity & temperature bases)
│   ├── RB_uncoupled_FUN.py          # Uncoupled ROM functions (modes, projection, ODE RHS, Jacobian)
│   ├── RB_uncoupled_ROM.py          # Build ROM from scratch and run a test simulation
│   ├── RB_uncoupled_ROM_parallel.py # Parallel ROM build (multiprocessing)
│   ├── RB_uncoupled_simulate.py     # Load precomputed ROM → simulate → visualise
│   └── ROM/                         # Precomputed ROM coefficient files (.h5)
│
└── coupled/                         # Coupled ROM (joint velocity-temperature basis)
    ├── RB_coupled_FUN.py            # Coupled ROM functions (modes, projection, ODE RHS, Jacobian)
    ├── RB_coupled_ROM.py            # Build ROM from scratch and run a test simulation
    ├── RB_coupled_ROM_parallel.py   # Parallel ROM build (multiprocessing)
    ├── RB_coupled_simulate.py       # Load precomputed ROM → simulate → visualise
    └── ROM/                         # Precomputed ROM coefficient files (.h5)
```

All scripts must be run **from the repository root** (`rayleigh_benard_ROM/`), so that `FUN.py` and the `uncoupled/` and `coupled/` sub-packages are on the Python path.

---

## 3. Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.13

### Create the environment

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
| `h5py` | Reading and writing ROM coefficient files |
| `mpi4py` | Parallel ROM build (optional) |

---

## 4. Quickstart: simulate from precomputed ROM data

The fastest way to get started is to load a precomputed ROM and run a simulation. All ROM tensor coefficients have been pre-projected and stored in the `ROM/` directories.

### Uncoupled ROM

```bash
cd rayleigh_benard_ROM
python uncoupled/RB_uncoupled_simulate.py
```

The script loads the precomputed ROM for $N_\alpha = 8$ wavenumbers and $n = 12$ modes per wavenumber (192 total modes), integrates the ROM ODE for a chosen $(Ra, Pr)$ pair, reconstructs the physical-space vorticity and temperature fields, and exports time snapshots to `uncoupled/frames/`.

Key parameters at the top of the script:

```python
n_alpha, n  = 8, 12            # wavenumbers × modes per wavenumber
Pr          = 10               # Prandtl number
Ra          = 200 * Ra_c       # Rayleigh number  (Ra_c ≈ 657.5)
tend        = 500/np.sqrt(Pr)  # integration end time
```

### Coupled ROM

```bash
cd rayleigh_benard_ROM
python coupled/RB_coupled_simulate.py
```

The coupled ROM uses a single joint basis $\{\boldsymbol{\chi}_i\}$ that simultaneously represents velocity and temperature fluctuations. The script loads the precomputed ROM for $N_\alpha = 6$ wavenumbers and $n = 16$ modes per wavenumber (96 total modes).

Key parameters at the top of the script:

```python
n_alpha, n      = 6, 16    # wavenumbers × modes per wavenumber
ROM_Pr, ROM_Ra  = 1, 1     # Ra and Pr at which the ROM was built
ROM_g2          = 1.24     # temperature energy scaling factor g²
Pr              = 1        # simulation Prandtl number
Ra              = 1500     # simulation Rayleigh number
```

> **Generalisation across parameters.** The ROM tensors are computed at a reference $(Ra_\mathrm{ROM}, Pr_\mathrm{ROM})$. Because $Ra$ and $Pr$ appear only as scalar prefactors in the projected ODE, the same tensor set can be used for any $(Ra, Pr)$ at no additional computational cost.

---

## 5. Build the ROM from scratch

If you want to recompute the ROM tensors for a different resolution or domain, use the build scripts. This step is computationally expensive for large mode counts and is not required to reproduce the paper results.

### Uncoupled ROM

```bash
python uncoupled/RB_uncoupled_ROM.py
```

The script performs the following pipeline:

1. **Build the velocity basis** — for each horizontal wavenumber $\alpha_i = i \cdot 2\pi/L_x$, compute Stokes modes by solving a controllability Gramian for the biharmonic operator $\nabla^{-2}\nabla^4\psi = \lambda\psi$.
2. **Build the temperature basis** — compute thermal diffusion modes by solving a controllability Gramian for the Laplacian $\nabla^2\theta = \lambda\theta$.
3. **Normalise** both bases to be orthonormal under the $L^2$ volume-averaged inner product.
4. **Project** the Navier-Stokes and energy equations onto each basis function to extract the ROM tensors: forcing ($\mathbf{f}_0$, $F_{1}$), diffusion ($D_u$, $D_T$), linear advection ($L$), and the nonlinear triadic tensor ($N_{ijk}$). A **triad interaction rule** is enforced, keeping only wavenumber-compatible triads and reducing the nominal $O(N^3)$ cost.
5. **Save** all ROM data to an `.h5` file in `uncoupled/ROM/`.

For parallel computation of the projection (recommended for $N > 100$):

```bash
python uncoupled/RB_uncoupled_ROM_parallel.py
```

### Coupled ROM

```bash
python coupled/RB_coupled_ROM.py
```

1. **Build the coupled basis** — for each non-zero wavenumber, solve a generalised Sylvester equation for the coupled velocity–temperature state matrix, whose direct and adjoint operators include the buoyancy coupling terms $A_{12}$ and $A_{21}$. Zero-wavenumber modes use the decoupled Stokes/thermal basis.
2. **Normalise** under a weighted inner product that accounts for the temperature energy scaling $g^2$.
3. **Project** and **save** as in the uncoupled case, but with the unified state vector $\boldsymbol{\chi}_i = [u_i, v_i, \theta_i]^T$.

---

## 6. Code overview

### `FUN.py` — shared spectral utilities

| Function | Description |
|----------|-------------|
| `cheb(n, Lx, x0)` | Chebyshev differentiation matrix and grid for $n$ points |
| `clenshaw_curtis_compute(n, Lx, x0)` | Clenshaw-Curtis quadrature weights for integration over the Chebyshev grid |
| `Inner_prod(f, g, x, y, W)` | Volume-averaged $L^2$ inner product $\langle f, g \rangle$ |
| `normalize_modes(base, X, Y, W)` | Orthonormalise a modal basis and report the maximum non-orthogonality |
| `Lapl_2D(f, X, Y, DY2)` | 2D Laplacian (Fourier in $x$, Chebyshev in $y$) |
| `grad(f, X, Y, DY)` | 2D gradient $(f_x, f_y)$ |
| `CONV(Xj, Vk, X, Y, DY)` | Advection $(\mathbf{V}_k \cdot \nabla)\mathbf{X}_j$ |
| `linear_analysis()` | Newton solver for the critical Rayleigh number $Ra_c(k)$ |

### `uncoupled/RB_uncoupled_FUN.py`

| Function | Description |
|----------|-------------|
| `vel_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n)` | Stokes velocity modes from the biharmonic Gramian |
| `temp_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n)` | Temperature modes from the Laplacian Gramian |
| `process_mode(i, n, nmodes, Ubase, Tbase, ...)` | Projects equations onto mode $i$; returns rows of all ROM tensors |
| `ROM_sparse(t, ci, Pr, Ra, ...)` | ODE right-hand side using sparse nonlinear tensor evaluation |
| `jac_uncoupled(t, ci, Pr, Ra, ...)` | Analytical Jacobian (for stiff LSODA/Radau integrators) |

The uncoupled ROM ODE reads:

$$\dot{a}_i = Pr\!\left(f_{0,i} + F_{1,ij}\,b_j\right) + \frac{Pr}{\sqrt{Ra}}\,D_{u,ij}\,a_j - N^u_{ijk}\,a_j\,a_k$$

$$\dot{b}_i = \frac{1}{\sqrt{Ra}}\,D_{T,ij}\,b_j - L_{ij}\,a_j - N^T_{ijk}\,b_j\,a_k$$

where $a_i$ are velocity coefficients and $b_i$ are temperature coefficients.

### `coupled/RB_coupled_FUN.py`

| Function | Description |
|----------|-------------|
| `coupled_modes(nx, ny, Lx, Ly, kx, T0, Pr, Ra, n, g2)` | Coupled velocity–temperature modes from the generalised Gramian |
| `vel_modes(...)` | Stokes modes (used for the $k_x = 0$ subspace) |
| `temp_modes(...)` | Thermal modes (used for the $k_x = 0$ subspace) |
| `process_mode(i, n, nmodes, base, ...)` | Projects equations onto mode $i$ |
| `ROM_sparse(t, Xi, Pr, Ra, ...)` | ODE right-hand side for the coupled system |
| `jac_coupled(t, y, Pr, Ra, ...)` | Analytical Jacobian |

The coupled ROM ODE for the unified state $\chi_i$ reads:

$$\dot{\chi}_i = Pr\!\left(F_{0,i} + F_{1,ij}\,\chi_j\right) + \frac{Pr}{\sqrt{Ra}}\,D_{u,ij}\,\chi_j + \frac{1}{\sqrt{Ra}}\,D_{T,ij}\,\chi_j - L_{ij}\,\chi_j - N_{ijk}\,\chi_j\,\chi_k$$

---

## 7. Precomputed ROM files

The `ROM/` directories contain HDF5 files with precomputed ROM tensors. File names encode the discretisation parameters:

```
RB_uncoupled_nx{NX}_ny{NY}_mX{Nalpha}_mY{n}_N{ndim}.h5
RB_coupled_nx{NX}_ny{NY}_mX{Nalpha}_mY{n}_N{nmodes}_Pr{Pr}_Ra{Ra}_g2_{g2}.h5
```

### Uncoupled ROM files

| File | $N_\alpha$ | $n$ | Total modes $N$ |
|------|-----------|-----|-----------------|
| `..._nx030_ny064_mX08_mY12_N096.h5` | 8 | 12 | 96 |
| `..._nx046_ny064_mX12_mY16_N192.h5` | 12 | 16 | 192 |

### Coupled ROM files

| File | $N_\alpha$ | $n$ | $N$ | $Pr_\mathrm{ROM}$ | $Ra_\mathrm{ROM}$ | $g^2$ |
|------|-----------|-----|-----|---------|---------|-------|
| `..._N096_Pr1_Ra1_g2_1.h5` | 6 | 16 | 96 | 1 | 1 | 1.00 |
| `..._N096_Pr1_Ra1_g2_1p24.h5` | 6 | 16 | 96 | 1 | 1 | 1.24 |
| `..._N192_Pr1_Ra1_g2_1p24.h5` | 8 | 24 | 192 | 1 | 1 | 1.24 |
| `..._N384_Pr1_Ra1_g2_1p24.h5` | 12 | 32 | 384 | 1 | 1 | 1.24 |
| `..._N096_Pr1_Ra1500_g2_1p24.h5` | 6 | 16 | 96 | 1 | 1500 | 1.24 |
| `..._N096_Pr1_Ra5000_g2_1p24.h5` | 6 | 16 | 96 | 1 | 5000 | 1.24 |

Each `.h5` file stores the full ROM state: the mode arrays (`Ubase`, `Tbase` or `base`), the grid (`X`, `Y`), the mean profile (`TT0`), and all projected tensor coefficients.

---

## 8. Key parameters

| Parameter | Symbol | Typical values | Description |
|-----------|--------|----------------|-------------|
| `n_alpha` | $N_\alpha$ | 6, 8, 12 | Number of retained horizontal wavenumbers |
| `n` | $n$ | 8, 12, 16, 32 | Modes per wavenumber |
| `nmodes` | $N$ | 96–384 | Total modal basis size |
| `Pr` | $Pr$ | 1, 10 | Prandtl number |
| `Ra` | $Ra$ | $Ra_c$–$5000\,Ra_c$ | Rayleigh number |
| `g2` | $g^2$ | 1.00, 1.24 | Temperature energy weight in coupled basis |
| `nx`, `ny` | — | 30/46, 64 | Physical grid size (Fourier × Chebyshev) |

The Chebyshev grid uses $n_y = 64$ collocation points in $y \in [0,1]$ with free-slip, fixed-temperature boundary conditions enforced as Dirichlet conditions on the streamfunction.

---

## 9. Citation

If you use this code or data, please cite:

```bibtex
@article{FloresMontoya2025,
  author  = {Flores-Montoya, Enrique and [co-authors]},
  title   = {[Paper title]},
  journal = {[Journal name]},
  year    = {2025},
  volume  = {},
  pages   = {},
  doi     = {}
}
```

---

## Contact

Enrique Flores-Montoya — enriquefloresmontoya1996@gmail.com

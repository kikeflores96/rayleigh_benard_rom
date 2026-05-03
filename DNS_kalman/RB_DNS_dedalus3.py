#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rayleigh-Bénard convection DNS using Dedalus 3

Solves the 2D Boussinesq equations in buoyancy form with no-slip,
fixed-temperature boundary conditions. Requires the separate Dedalus 3
environment (see dns_dedalus3_environment.yaml at repo root).

Run from the repo root:
    conda activate dns_dedalus3
    python DNS_kalman/RB_DNS_dedalus3.py

Output is written to:
    DNS_kalman/simulations/DNS_Pr{Pr}_R{r}_rseed{rseed}/

@author: efloresm
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Critical Rayleigh number for no-slip boundary conditions
Ra_c = 1707.7651913068134

# Parameters
Pr    = 10
r     = 80
Ra    = r * Ra_c
rseed = 1

# Case labelling
Pr_text   = f'{Pr:03.1f}'.replace('.', 'p')
r_text    = f'{r:03.0f}'.replace('.', 'p')
case_path = f'DNS_kalman/simulations/DNS_Pr{Pr_text}_R{r_text}_rseed{rseed:03.0f}'

# Domain dimensions
Lx, Ly          = 2, 1

# Numerical parameters
nt              = 2500
Nx, Ny          = 128, 64
dealias         = 3/2
stop_sim_time   = 1500/Pr**0.5
framerate       = stop_sim_time/nt
timestepper     = d3.SBDF2
max_timestep    = 1e-2/Pr**0.5
init_timestep   = 1e-6
dtype           = np.float64

# Bases
coords  = d3.CartesianCoordinates('x', 'y')
dist    = d3.Distributor(coords, dtype=dtype)
xbasis  = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis  = d3.ChebyshevT(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
p       = dist.Field(name='p', bases=(xbasis,ybasis))
b       = dist.Field(name='b', bases=(xbasis,ybasis))
u       = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))
tau_p   = dist.Field(name='tau_p')
tau_b1  = dist.Field(name='tau_b1', bases=xbasis)
tau_b2  = dist.Field(name='tau_b2', bases=xbasis)
tau_u1  = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2  = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
x, y        = dist.local_grids(xbasis, ybasis)
ex, ey      = coords.unit_vector_fields(dist)
lift_basis  = ybasis.derivative_basis(1)
lift        = lambda A: d3.Lift(A, lift_basis, -1)
grad_u      = d3.grad(u) + ey*lift(tau_u1) # First-order reduction
grad_b      = d3.grad(b) + ey*lift(tau_b1) # First-order reduction

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - div(grad_b)/Ra**0.5 + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - Pr*div(grad_u)/Ra**0.5 + grad(p) - Pr*b*ey + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(y=0) = 1")
problem.add_equation("u(y=0) = 0")
problem.add_equation("b(y=1) = 0")
problem.add_equation("u(y=1) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver                  = problem.build_solver(timestepper)
solver.stop_sim_time    = stop_sim_time

# Initial conditions
# Setup random seed for repeatability
b.fill_random('g', seed=rseed, distribution='normal', scale=1e-1) # Random noise
b['g'] *= (y - 1) * (y)     # Damp noise at walls
b['g'] += (1 - y)              # Add linear background



# Analysis
solution    = solver.evaluator.add_file_handler(case_path, 
                                                sim_dt=framerate, 
                                                max_writes=500)
solution.add_task(b, name='buoyancy')
solution.add_task(-d3.div(d3.skew(u)), name='vorticity')
solution.add_tasks(solver.state, layout='g')


# CFL
CFL = d3.CFL(solver, initial_dt=init_timestep, 
             cadence=5, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, 
             max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(Ra)*b*u@ey -d3.Gradient(b)@ey , name='Nu')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            Nu = flow.volume_integral('Nu')/Lx
            logger.info('Iteration=%i, Time=%e, dt=%e, Nu=%f' %(solver.iteration, solver.sim_time, timestep, Nu))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

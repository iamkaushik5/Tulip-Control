#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
This example is an extension of robot_discrete.py by including continuous
dynamics with disturbances.

Petter Nilsson (pettni@kth.se)
August 14, 2011

NO, system and cont. prop definitions based on TuLiP 1.x
2 Jul, 2013
NO, TuLiP 1.x discretization
17 Jul, 2013
"""
#
# Note: This code is commented to allow components to be extracted into
# the tutorial that is part of the users manual.  Comments containing
# strings of the form @label@ are used for this purpose.

#import logging
#logging.basicConfig(level=logging.INFO)

# @import_section@
import numpy as np

from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
# @import_section_end@

show = False

# @dynamics_section@
# Problem parameters
input_bound = 14.0
uncertainty = 0.01

# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])

# Continuous dynamics
A = np.array([[1.0, 0.], [ 0., 1.0]])
B = np.array([[0.2, 0.], [ 0., 0.2]])
E = np.array([[1,0], [0,1]])

# Available control, possible disturbances
U = input_bound *np.array([[-1., 1.], [-1., 1.]])
W = uncertainty *np.array([[-1., 1.], [-1., 1.]])

# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# @dynamics_section_end@

# @partition_section@
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None
# @partition_section_end@

# @discretize_section@
# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=False,
    N=1, min_cell_volume=0.1, plotit=show
)
# @discretize_section_end@

"""Visualize transitions in continuous domain (optional)"""
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts) if show else None

"""Specifications"""
# Environment variables and assumptions
env_vars = {'park'}
env_init = set()                # empty set
env_prog = '!park'
env_safe = set()                # empty set

# System variables and requirements
sys_vars = {'X0reach'}
sys_init = {'X0reach'}          
sys_prog = {'home'}               # []<>home
sys_safe = {'(X(X0reach) <-> lot) || (X0reach && !park)'}
sys_prog |= {'X0reach'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# @synthesize_section@
"""Synthesize"""
ctrl = synth.synthesize('jtlv', specs,
                        sys=disc_dynamics.ts, ignore_sys_init=True)

# Generate a graphical representation of the controller for viewing
#if not ctrl.save('continuous.png'):
#    print(ctrl)
# @synthesize_section_end@

# Simulation

from tulip.abstract.find_controller import *
x = [1.25]
y = [.25]
s0_part = find_discrete_state([x[0],y[0]],disc_dynamics.ppp)
s0_loc = disc_dynamics.ppp2ts[s0_part]

u = get_input(np.array([x[0],y[0]]), sys_dyn, disc_dynamics, s0_part, disc_dynamics.ppp2ts.index(disc_dynamics.ts[s0_loc].keys()[0]))

"""s0_loc = disc_dynamics.ppp2ts[s0_part]
mach = synth.determinize_machine_init(ctrl, {'loc':s0_loc})
sim_hor = 130

(s1, dum) = mach.reaction('Sinit', {'level': 1})
(s1, dum) = mach.reaction(s1, {'level': 1})
for sim_time in range(sim_hor):
    u = get_input(
            np.array([x[sim_time*N],y[sim_time*N]]),
            sys_dyn,
            disc_dynamics,
            s0_part,
            disc_dynamics.ppp2ts.index(dum['loc']),
            mid_weight=100.0,
            test_result=True)
    #u = get_input_helper_special(
    #        np.array([Tc[sim_time*N],Th[sim_time*N],P[sim_time*N]]),
    #        cont_sys.list_subsys[sysnow],
    #        disc_dynamics.ppp[disc_dynamics.ppp2ts.index(s0_loc)][0],
    #        disc_dynamics.ppp[disc_dynamics.ppp2ts.index(dum['loc'])][0],
    #        N=N)
    #u = u.reshape(N, 3)
    for ind in range(N):
        snow = np.dot(
                sys_dyn.A, [x[-1],y[-1]]
                ) + np.dot(sys_dyn.B,u[ind]) + sys_dyn.K.flatten()
        x.append(snow[0])
        y.append(snow[1])
    
    s0_part = find_discrete_state([Tc[-1],Th[-1],P[-1]],disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]
    print s0_loc, dum['loc']
    if pc.is_inside(disc_dynamics.ppp[disc_dynamics.ppp2ts.index(dum['loc'])],[x[-1],y[-1]]):
        s0_part = disc_dynamics.ppp2ts.index(dum['loc'])
    if sim_time <= 10:
        (s1, dum) = mach.reaction(s1, {'park': 1})
    elif sim_time <= 50:
        (s1, dum) = mach.reaction(s1, {'park': 0})
    else:
        (s1, dum) = mach.reaction(s1, {'park': 1})"""



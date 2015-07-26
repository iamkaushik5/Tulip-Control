#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
Simplified AMS example from Mickelin et al., ACC 2014

Necmiye Ozay, August 24, 2014
"""
import numpy as np

from tulip import spec, synth
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_strategy

plotting = False

# Problem parameters
input_bound = 30.
uncertainty = 0.0015
epsilon = .01
bound = 1-uncertainty/epsilon
beta = 1.

# Continuous state space
cont_state_space = box2poly([[-10., 10.], [-20., 20.], [-0.33, 1.39]])


# @subsystem0@
def subsys0():
    A = np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])
    B = beta*np.array([[10.1,0.1787, 0.000194], [30.46,0., 0.5361], [5.21, -2.311, 0.]])
    E = np.eye(3)
    K = np.array([[-0.00237196], [-0.159101], [-1.530935]])
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [-0.33, 0.01]])
    
    sys_dyn = LtiSysDyn(A, B, E, K, U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn
# @subsystem0_end@

# @subsystem1@
def subsys1():
    A = np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])
    B = beta*np.array([[10.11,0.1779, 0.000194], [30.58,-0.04497, 0.5361], [5.232, -2.318, 0.]])
    E = np.eye(3)
    K = np.array([[-0.00237196], [-0.159101], [-1.85701]])
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [0.01, 0.70]])
    
    sys_dyn = LtiSysDyn(A, B, E, K, U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn
# @subsystem1_end@

def subsys2():
    A = np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])
    B = beta*np.array([[10.13,0.1892, 0.000194], [30.61,-0.06435, 0.5361], [6.738, -3.34, 0.]])
    E = np.eye(3)
    K = np.array([[0.000058], [-0.1591], [-2.483662]])
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [0.70, 1.39]])
    
    sys_dyn = LtiSysDyn(A, B, E, K, U, W, dom)
    #sys_dyn.plot()
    
    return sys_dyn

# @pwasystem@
subsystems = [subsys0(), subsys1(), subsys2()]

# Build piecewise affine system from its subsystems
sys_dyn = PwaSysDyn(subsystems, cont_state_space)
# @pwasystem_end@

# Continuous proposition
cont_props = {}
cont_props['tempGood'] = box2poly([[-1.5, 1.5], [-20., 20.], [-0.33, 1.39]])
cont_props['tempCold'] = box2poly([[-6, -3], [-20., 20.], [-0.33, 1.39]])
cont_props['tempHot'] = box2poly([[3, 6], [-20., 20.], [-0.33, 1.39]])
cont_props['noHXfreeze'] = box2poly([[-10., 10.], [5., 20.], [-0.33, 1.39]])

# Compute the proposition preserving partition
# of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=0.5, plotit=plotting,
    cont_props=cont_props
)

# Specifications

env_vars ={}
env_vars['level'] = (0,2)
sys_vars ={'trackfl':(0,3)}

env_init = {'level=1'}
env_prog = set()
env_safe = set() 

sys_init = {'tempGood'}
sys_safe = {
    '(trackfl=0) -> tempHot',
    '(trackfl=1) -> tempGood',
    '(trackfl=2) -> tempCold',
    '(trackfl=0) -> X ((trackfl=0) || (level!=0))',
    '(trackfl=1) -> X ((trackfl=1) || (level!=1))',
    '(trackfl=2) -> X ((trackfl=2) || (level!=2))'
}

sys_prog = {
    '(trackfl=0) || (level!=0)',
    '(trackfl=1) || (level!=1)',
    '(trackfl=2) || (level!=2)',
    'noHXfreeze'
}


# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# Synthesize
ctrl = synth.synthesize('gr1c', specs,
                        sys=disc_dynamics.ts, ignore_sys_init=True)
if plotting:
    ax = plot_strategy(disc_dynamics, ctrl)
    ax.figure.savefig('pwa_proj_mealy.pdf')

# Save graphical representation of controller for viewing
if not ctrl.save('pwa.png'):
    print(ctrl)

# Simulation

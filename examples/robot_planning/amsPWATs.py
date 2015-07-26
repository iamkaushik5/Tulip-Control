#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
Simplified AMS example from Mickelin et al., ACC 2014

Necmiye Ozay, August 24, 2014
"""
#import logging
#logging.basicConfig(filename='ams.log', level=logging.DEBUG, filemode='w')
#logger = logging.getLogger(__name__)

import numpy as np

from tulip import spec, synth
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_strategy

plotting = False

# Problem parameters
input_bound = 1.
uncertainty = 0.0015
epsilon = .01
bound = 1-uncertainty/epsilon
beta = 1.

scale = 1.
ss = np.eye(3);
ss[2,2] = scale
invss = np.linalg.inv(ss)

# Continuous state space
cont_state_space = box2poly([[-10., 10.], [-20., 20.], [-0.33*scale, 1.39*scale]])


# @subsystem0@
def subsys0():
    A = np.dot(np.dot(ss, np.array([[0.2743, 0.6562, 0.08471], [0.,0.7813, 0.], [0.,0.,0.]])),invss)
    B = np.dot(ss, beta*np.array([[-185.3, 179.4, 2.28], [272.8,0., 4.801], [7.826, -4.391, 0.]]))
    E = np.dot(ss, np.eye(3))
    K = np.dot(ss, np.array([[-3.4381], [-0.0004], [6.2555]]))
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [-0.33*scale, 0.01*scale]])
    
    sys_dyn = LtiSysDyn(A, B, None, K, U, None, dom)
    #sys_dyn.plot()
    
    return sys_dyn
# @subsystem0_end@

# @subsystem1@
def subsys1():
    A = np.dot(np.dot(ss, np.array([[ 0.2743, 0.6562, 0.06744], [0.,0.7813, -0.01034], [0.,0.,0.]])),invss)
    B = np.dot(ss,beta*np.array([[-237.1, 209.6, 2.28], [243.1, 15.43, 4.801], [6.973, -3.947, 0.]]))
    E = np.dot(ss,np.eye(3))
    K = np.dot(ss,np.array([[-3.4381], [-0.0004], [2.9948]]))
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [0.01*scale, 0.70*scale]])
    
    sys_dyn = LtiSysDyn(A, B, None, K, U, None, dom)
    #sys_dyn.plot()
    
    return sys_dyn
# @subsystem1_end@

def subsys2():
    A = np.dot(np.dot(ss, np.array([[0.2723, 0.6543, 0.07406], [0.,0.7813, -0.01159], [0.,0.,0.]])),invss)
    B = np.dot(ss,beta*np.array([[-199.1, 193.7, 2.276], [239.8, 17.82, 4.801], [7.816, -4.56, 0.]]))
    E = np.dot(ss,np.eye(3))
    K = np.dot(ss,np.array([[0.0031], [-0.0004], [--3.2718]]))
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [0.70*scale, 1.39*scale]])
    
    sys_dyn = LtiSysDyn(A, B, None, K, U, None, dom)
    #sys_dyn.plot()
    
    return sys_dyn

# A = np.dot(np.dot(ss, np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])),invss)
# B = np.dot(ss, beta*np.array([[10.1,0.1787, 0.000194], [30.46,0., 0.5361], [5.21, -2.311, 0.]]))
# E = np.dot(ss, np.eye(3))
# K = np.dot(ss, np.array([[-0.00237196], [-0.159101], [-1.530935]]))
    
# U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
# U.scale(input_bound)
    
# W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
# W.scale(uncertainty)
    
# sys_dyn = LtiSysDyn(A, B, None, K, U, None, cont_state_space)

subsystems = [subsys0(), subsys1(), subsys2()]
# Build piecewise affine system from its subsystems
sys_dyn = PwaSysDyn(subsystems, cont_state_space)

# Continuous proposition
cont_props = {}
cont_props['tempGood'] = box2poly([[-1.5, 1.5], [-20., 20.], [-0.33*scale, 1.39*scale]])
cont_props['tempCold'] = box2poly([[-10, -3], [-20., 20.], [-0.33*scale, 1.39*scale]])
cont_props['tempHot'] = box2poly([[3, 10], [-20., 20.], [-0.33*scale, 1.39*scale]])
cont_props['noHXfreeze'] = box2poly([[-10., 10.], [5., 20.], [-0.33*scale, 1.39*scale]])

# Compute the proposition preserving partition
# of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)


disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=1, min_cell_volume=5, plotit=plotting,
    cont_props=cont_props, abs_tol=0.1
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
    '(trackfl=0) -> X ((trackfl=0) || !(level=0))',
    '(trackfl=1) -> X ((trackfl=1) || !(level=1))',
    '(trackfl=2) -> X ((trackfl=2) || !(level=2))'
}

sys_prog = {
    '(trackfl=0) || !(level=0)',
    '(trackfl=1) || !(level=1)',
    '(trackfl=2) || !(level=2)',
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

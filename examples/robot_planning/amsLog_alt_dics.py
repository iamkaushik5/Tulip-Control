#!/usr/bin/env python
#
# WARNING: This example may not yet be working.  Please check again in
#          the upcoming release.
#
"""
Simplified AMS example from Mickelin et al., ACC 2014

Necmiye Ozay, August 24, 2014

N.O. September 20, 2014 Alternative discretization added
"""
#import logging
#logging.basicConfig(filename='ams2.log', level=logging.DEBUG, filemode='w')
#logger = logging.getLogger(__name__)

import numpy as np

from tulip import spec, synth
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from polytope import box2poly
from tulip.abstract import prop2part, add_grid, is_feasible_alternative, pwa_partition
from tulip.abstract.plot import plot_strategy
from scipy import sparse as sp
from tulip import transys as trs
from tulip.abstract.discretization import AbstractPwa

plotting = False

# Problem parameters
input_bound = 4.
uncertainty = 0.0015
epsilon = .01
bound = 1-uncertainty/epsilon
beta = 1.

scale = 10.
ss = np.eye(3);
ss[2,2] = scale
invss = np.linalg.inv(ss)

# Continuous state space
cont_state_space = box2poly([[-10., 10.], [-20., 20.], [-0.33*scale, 1.39*scale]])

# first state temp, second HX, third P

# @subsystem0@
def subsys0():
    A = np.dot(np.dot(ss, np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])),invss)
    B = np.dot(ss, beta*np.array([[10.1,0.1787, 0.000194], [30.46,0., 0.5361], [5.21, -2.311, 0.]]))
    E = np.dot(ss, np.eye(3))
    K = np.dot(ss, np.array([[-0.00237196], [-0.159101], [-1.530935]]))
    
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
    A = np.dot(np.dot(ss, np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])),invss)
    B = np.dot(ss,beta*np.array([[10.11,0.1779, 0.000194], [30.58,-0.04497, 0.5361], [5.232, -2.318, 0.]]))
    E = np.dot(ss,np.eye(3))
    K = np.dot(ss,np.array([[-0.00237196], [-0.159101], [-1.85701]]))
    
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
    A = np.dot(np.dot(ss, np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])),invss)
    B = np.dot(ss,beta*np.array([[10.13,0.1892, 0.000194], [30.61,-0.06435, 0.5361], [6.738, -3.34, 0.]]))
    E = np.dot(ss,np.eye(3))
    K = np.dot(ss,np.array([[0.000058], [-0.1591], [-2.483662]]))
    
    U = box2poly([[-0.155, 0.845], [-0.155, 0.845], [-2.49, 5.826]])
    U.scale(input_bound)
    
    W = box2poly([[-1., 1.], [-1., 1.], [-1., 1.]])
    W.scale(uncertainty)
    
    dom = box2poly([[-10., 10.], [-20., 20.], [0.70*scale, 1.39*scale]])
    
    sys_dyn = LtiSysDyn(A, B, None, K, U, None, dom)
    #sys_dyn.plot()
    
    return sys_dyn

# @pwasystem@
subsystems = [subsys0(), subsys1(), subsys2()]

# Build piecewise affine system from its subsystems
sys_dyn = PwaSysDyn(subsystems, cont_state_space)
# @pwasystem_end@

# Continuous proposition
cont_props = {}
cont_props['tempGood'] = box2poly([[-1.5, 1.5], [-20., 20.], [-0.33*scale, 1.39*scale]])
cont_props['tempCold'] = box2poly([[-6, -3], [-20., 20.], [-0.33*scale, 1.39*scale]])
cont_props['tempHot'] = box2poly([[3, 6], [-20., 20.], [-0.33*scale, 1.39*scale]])
cont_props['noHXfreeze'] = box2poly([[-10., 10.], [5., 20.], [-0.33*scale, 1.39*scale]])

# Compute the proposition preserving partition
# of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
cont_par = add_grid(cont_partition, num_grid_pnts=[80, 2, 2])

cont_par, subsys_list, parents = pwa_partition(sys_dyn, cont_par)

adj = np.zeros(cont_par.adj.shape, dtype=bool)

N=5

for in1 in range(cont_par.adj.shape[0]):
    for in2 in range(cont_par.adj.shape[1]):
        if cont_par.adj[in1,in2]==1:
            #transmat[in1,in2]=is_feasible(cont_par2[in1], cont_par2[in2], 
            #                      sys_dyn.list_subsys[subsys_list[in1]], N=4)
            adj[in1,in2] = is_feasible_alternative(cont_par[in1], cont_par[in2], 
                               sys_dyn.list_subsys[subsys_list[in1]], N=N)

ofts = trs.FTS()
adj = sp.lil_matrix(adj)
n = adj.shape[0]
ofts_states = range(n)
ofts_states = trs.prepend_with(ofts_states, 's')
ofts.states.add_from(ofts_states)
ofts.transitions.add_adj(adj, ofts_states)
# Decorate TS with state labels
atomic_propositions = set(cont_par.prop_regions)
ofts.atomic_propositions.add_from(atomic_propositions)
for state, region in zip(ofts_states, cont_par.regions):
    state_prop = region.props.copy()
    ofts.states.add(state, ap=state_prop)

param = {'N':N, 'closed_loop':False, 'conservative':True}
#ppp2orig = [part2orig[x] for x in orig]
#end_time = os.times()[0]
#msg = 'Total abstraction time: ' +\
#str(end_time - start_time) + '[sec]'

disc_dynamics = AbstractPwa(ppp=cont_par,
                    ts=ofts,
                    ppp2ts=ofts_states,
                    pwa=sys_dyn,
                    pwa_ppp=cont_par,
                    disc_params=param)


#disc_dynamics = discretize(
#    cont_partition, sys_dyn, closed_loop=True,
#    N=8, min_cell_volume=0.5, plotit=plotting,
#    cont_props=cont_props, abs_tol=0.01
#)

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

fname = 'int_state_'
import pickle
pickle.dump(specs, open(fname+'specs', 'wb') )
pickle.dump(ctrl, open(fname+'ctrl', 'wb') )
pickle.dump(disc_dynamics, open(fname+'disc_dyn', 'wb') )
pickle.dump(sys_dyn, open(fname+'sys_dyn', 'wb') )

# Save graphical representation of controller for viewing
#if not ctrl.save('pwa.png'):
#    print(ctrl)

# Simulation

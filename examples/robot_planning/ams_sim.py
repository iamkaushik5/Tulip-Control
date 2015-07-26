#!/usr/bin/env python
#
# PWA simulation trial

import pickle
import numpy as np

from tulip import spec, synth
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_strategy
from tulip.abstract.find_controller import *
from tulip.abstract.feasible import exists_input as get_input_helper_special
import polytope as pc

lp_solver = 'mosek'

#fname = 'scl10N1_pwa_'
#fname = 'new3_'
fname = 'int_state_'

disc_dynamics = pickle.load(open(fname+'disc_dyn', 'r') )

cont_sys = pickle.load(open(fname+'sys_dyn', 'r') )

#sys_ts = pickle.load(open(fname+'ts', 'r') )

specs = pickle.load(open(fname+'specs', 'r') )

ctrl = pickle.load(open(fname+'ctrl', 'r') )

N=disc_dynamics.disc_params['N']
#N=2

#disc_dynamics.disc_params['N']=N
#disc_dynamics.disc_params['conservative']=True
#disc_dynamics.disc_params['closed_loop']=False


Tc = [1.3]
Th = [-1.]
P = [-1.3]

s0_part = find_discrete_state([Tc[0],Th[0],P[0]],disc_dynamics.ppp)
s0_loc = disc_dynamics.ppp2ts[s0_part]
mach = synth.determinize_machine_init(ctrl, {'loc':s0_loc})
sim_hor = 130

(s1, dum) = mach.reaction('Sinit', {'level': 1})
(s1, dum) = mach.reaction(s1, {'level': 1})
for sim_time in range(sim_hor):
    for i in range(3):
        if np.array([Tc[sim_time*N],Th[sim_time*N],P[sim_time*N]]) in cont_sys.list_subsys[i].domain:
            sysnow=i
    u = get_input(
            np.array([Tc[sim_time*N],Th[sim_time*N],P[sim_time*N]]),
            cont_sys.list_subsys[sysnow],
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
        x = np.dot(
                cont_sys.list_subsys[sysnow].A, [Tc[-1],Th[-1],P[-1]]
                ) + np.dot(cont_sys.list_subsys[sysnow].B,u[ind]) + cont_sys.list_subsys[sysnow].K.flatten()
        Tc.append(x[0])
        Th.append(x[1])
        P.append(x[2])
    
    s0_part = find_discrete_state([Tc[-1],Th[-1],P[-1]],disc_dynamics.ppp)
    s0_loc = disc_dynamics.ppp2ts[s0_part]
    print s0_loc, dum['loc']
    if pc.is_inside(disc_dynamics.ppp[disc_dynamics.ppp2ts.index(dum['loc'])],[Tc[-1],Th[-1],P[-1]]):
        s0_part = disc_dynamics.ppp2ts.index(dum['loc'])
    if sim_time <= 10:
        (s1, dum) = mach.reaction(s1, {'level': 1})
    elif sim_time <= 50:
        (s1, dum) = mach.reaction(s1, {'level': 0})
    else:
        (s1, dum) = mach.reaction(s1, {'level': 2})



import switchingFunctions
import numpy as np
from tulip import *
import tulip.polytope as pc
from tulip.polytope.plot import *
import matplotlib.pyplot as plt
from tank_functionsNEW import *
import os,sys
from random import shuffle
from generateFilter import *
from cvxopt import matrix
from shrinkPoly import *
import mergingTest
import pickle

"""The example controls a simplified
AMS system with piecewise affined dynamics
and partial state information.

Oscar Mickelin (oscarmi@kth.se)
August 5, 2013

This file contains a non-switched system without PWA dynamics
as a simple template.

OM, August 24 2014
"""

### Problem parameters
testfile = 'AMSN5Right'
path = os.path.abspath(os.path.dirname(sys.argv[0]))
smvfile = os.path.join(path, 'specs', testfile+'.smv')
spcfile = os.path.join(path, 'specs', testfile+'.spc')
autfile = os.path.join(path, 'specs', testfile+'.aut')


N=1
uncertainty = 0.003
epsilon = .02
bound = 1-uncertainty/epsilon
beta = 5.
print(bound)
domain = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [1.39], [0.33]]))


##Definition of dynamics
############# FIRST SET OF ENGINE PARAMETERS

What = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
    epsilon*np.array([[1.], [1.], [1.], [1.], [1.], [1.]]))


ANW11 = np.array([[1.005, 0.0007144, 0.002231], [0.,0.979, 0.], [0.,0.,0.]])
BNW11 = beta*np.array([[10.1,0.1787, 0.000194], [30.46,0., 0.5361], [5.21, -2.311, 0.]])
ENW11 = np.eye(3)
CNW11 = matrix(np.array([[1.,0., 0.], [0.,1., 0.], [0.,0.,0.]]))
print(CNW11)
LNW11 = generateFilter(matrix(ANW11), matrix(CNW11), bound)
KNW11 = np.array([[-0.00237196], [-0.159101], [-1.530935]])
UNW11 = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
    np.array([[0.845], [0.155], [0.845], [0.155], [5.826], [2.49]]))
WNW11 = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
    uncertainty*np.array([[1.], [1.], [1.], [1.], [1.], [1.]]))
domNW11 = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [0.01], [0.33]]))
sysDynHat11 = hybrid.LtiSysDyn(A=ANW11, B=BNW11, E=np.dot(LNW11, CNW11), K = KNW11, Uset=UNW11, Wset=What, sub_domain=domNW11)


### Filters
Ls1 = LNW11

### Continuous propositions
cont_props = {}
cont_props['tempOK'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[5.], [5.], [100.], [20.], [1.16], [0.58]]))
cont_props['tempGood'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[1.5], [1.5], [20.], [20.], [1.39], [0.33]]))
cont_props['tempCold'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[-1.5], [10.], [20.], [20.], [1.39], [0.33]]))
cont_props['tempHot'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [-1.5], [20.], [20.], [1.39], [0.33]]))
cont_props['pressureGood'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [0.82], [0.58]]))
cont_props['pressureBad'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [1.16], [-0.82]]))
cont_props['noHXfreeze'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [-5.], [1.39], [0.33]]))
cont_props['1chokedFlow'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [1.39], [-0.7]]))
cont_props['2chokedFlow'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [0.01], [0.33]]))
cont_props['chokedFlow'] = pc.Polytope.from_box(np.array([[1., 0., 0.], [-1., 0., 0.], [0.,1., 0.], [0.,-1., 0.], [0.,0.,1.], [0.,0.,-1.]]), \
        np.array([[10.], [10.], [20.], [20.], [0.77], [-0.01]]))


### Discretization procedure
print("Matrices synthesized")
propPres = prop2part.prop2part2(domain, cont_props)
print("Proposition preserving partition done")
#propPres = prop2part.prop2partconvex(propPres)
print("Convex proposition preserving partition done")
lengthie = len(propPres.list_region)
for i in range(lengthie):
   propPres.list_region[i] = shrinkRegion(propPres.list_region[i], epsilon)
print("Shrinking done")
#propPres = prop2part.prop2partconvex(propPres)
print("Convex proposition preserving partition of shrunk polytopes done")


disc = discretize.discretize(propPres, sysDynHat11, N=N, conservative=False, use_all_horizon=False, closed_loop=True, min_cell_volume=0.5, verbose=3)

f = open('AMSDisc'+testfile, 'w')
pickle.dump(disc, f)
f.close()


f=open('AMSDisc'+testfile, 'r+')
disc = pickle.load(f)
f.close()

# Specifications

# Environment variables and assumptions
env_vars ={'mode':[0,1], 'level':[0,1,2]}
env_init = 'level=1'                # empty set
env_prog = set()
env_safe = set()                # empty set

# System variables and requirements
sys_vars ={'trackfl':[0,1,2,3]}
sys_init = {'trackfl=0'}          
sys_prog = {'noHXfreeze && ((trackfl=0) || (level!=0)) && ((trackfl=1) || (level!=1)) && (trackfl=2) | (level!=2)'}               # []<>home
sys_safe = {'((trackfl=0) -> tempHot) && ((trackfl=1) -> tempGood) && ((trackfl=2) -> tempCold) \
		&& ((trackfl=0) -> next((trackfl=0) | (level!=0))) && ((trackfl=1) -> next((trackfl=1) | (level!=1))) \
		&& ((trackfl=2) -> next((trackfl=2) | (level!=2)))'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# Synthesize
ctrl = synth.synthesize('jtlv', specs, disc_dynamics.ofts)


prob = jtlvint.generateJTLVInput(env_vars, sys_vars, [assumption, guarantee],
                                 {}, disc, smvfile, spcfile, verbose=3, file_exist_option='r')

# Check realizability
realizability = jtlvint.checkRealizability(smv_file=smvfile, spc_file=spcfile,
                                           aut_file=autfile, verbose=3, file_exist_option='r')

# Compute an automaton
jtlvint.computeStrategy(smv_file=smvfile, spc_file=spcfile, aut_file=autfile,
                        priority_kind=3, verbose=3, file_exist_option='r')
aut = automaton.Automaton(autfile, [], 3)



# Remove dead-end states from automaton
aut.trimDeadStates()
print(aut.states)

# Simulate
num_it = 100
env_states = {'level':1, 'mode':1}


states = grsim.grsim([aut],env_states=[env_states], 
             num_it=num_it,
                     deterministic_env=False)

# Store discrete trajectory in np array
cellid_arr = []
mode_arr = []
level_arr = []
for (autID, state) in states:
    print(state)
    mode_arr.append(state.state['mode'])
    cellid_arr.append(state.state['cellID'])
    level_arr.append(state.state['level'])
cellid_arr = np.array(cellid_arr)

# First continuous state is middle point of first cell
r, x = pc.cheby_ball(disc.list_region[cellid_arr[0]])
x = x.flatten().T
print(discretize.get_cellID(x, disc))
print(cellid_arr)

x_arr = x
x0_arr = x
x1_arr = x
x0_t = 0
x1_t = 0
print(x)
# Filter and observer
## The observed state is initialized in the
## same partition as the unknown state x
theta = 2*np.pi*np.random.random()
x1 = 2*np.random.random()-1
x2 = 2*np.random.random()-1
x3 = 2*np.random.random()-1
xh = x + epsilon*np.array([x1, x2, x3])
print(xh)
xh_arr = xh     #Storage
u_arr = np.zeros([N*num_it, BNW11.shape[1]])
d_arr = np.zeros([N*num_it, ENW11.shape[1]])
C = CNE11                         ###########<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<---------- DEFINITION OF C
y = np.dot(CNE11, x)
y_arr = y
t_11 = []
t_12 = []
t_21 = []
t_22 = []
print(cellid_arr)
print(disc.trans)
sysDyn = sysDyn11
sysDynHat = sysDynHat11
Ls = [LNW11, LSE11, LNE11]
for i in range(1, num_it):
    # For each step, calculate N input signals
    for j in range(N):
    #Simulate using the observed dynamics
        for index in range(len(sysDyn.list_subsys)):
            if pc.is_inside(sysDyn.list_subsys[index].sub_domain, xh):
                sys_dyn = sysDyn.list_subsys[index]
		sys_dyn_hat = sysDynHat.list_subsys[index]
                A = sysDyn.list_subsys[index].A
                B = sysDyn.list_subsys[index].B
                E = sysDyn.list_subsys[index].E
		K = sysDyn.list_subsys[index].K
		L = Ls[index]
                break
        u_seq = discretize.get_input(xh, sys_dyn_hat, disc, \
                cellid_arr[i-1], cellid_arr[i], N-j, mid_weight=3, Q=np.eye(3*(N-j)), \
                test_result=True, closed_loop=True, conservative = False)
        u0 = u_seq[0,:] # Only the first input should be used

	#u0 = np.random.random(2)-0.155
	#tmp = np.random.random()*1.74-0.58
	#u0 = np.hstack([u0,tmp])        
	#print(u0)
	print(x)
        u_arr[(i-1)*N + j,:] = u0   # Store input
        
        d =  uncertainty * 2 * (np.random.rand(3) - 0.5 )   # Simulate disturbance
        d_arr[(i-1)*N + j,:] = d    # Store disturbance
    
    #Measurement
        y = np.dot(C,x).flatten()
        y_arr = np.vstack([y_arr, y])
        x = np.dot(sys_dyn.A, x).flatten() + np.dot(sys_dyn.B, u0).flatten() + K.flatten() + np.dot(sys_dyn.E, d).flatten()

    #Estimate
        xh = np.dot(sys_dyn.A - np.multiply(L,C), xh).flatten() + K.flatten() + np.dot(B, u0).flatten() + np.dot(L,y).flatten()
        #print(x,xh) 
        xh_arr = np.vstack([xh_arr, xh])
        x_arr = np.vstack([x_arr, x])   # Store state

plt.plot(23+x_arr[:,0])
plt.savefig('AMSplots/noSwitchShrunk.png')
plt.show()

"""
# Remove dead-end states from automaton
aut.trimDeadStates()
print(aut.states)
print("========================")
print("Number of states:")
print(len(merged.list_region))
print(len(aut.states))
#raw_input()
# Simulate
N=5
num_it = 100
env_states = {'level':1, 'levelClock':0, 'mode':1}

#for i in range(6000):
#	env_states.append({'level':'0', 'mode':'1'})
#for i in range(1000):
#	env_states.append({'level':'1'})
#for i in range(1000):
#	env_states.append({'level':'2'})
#for i in range(1000):
#	env_states.append({'level':'1'})
#for i in range(1000):
#	env_states.append({'level':'0'})
#for i in range(1000):
#	env_states.append({'level':'2'})

states = grsim.grsim([aut],env_states=[env_states], 
             num_it=num_it,
                     deterministic_env=False)

# Store discrete trajectory in np array
cellid_arr = []
mode_arr = []
for (autID, state) in states:
    mode_arr.append(state.state['mode'])
    cellid_arr.append(state.state['cellID'])
    print(state)
#    raw_input()
cellid_arr = np.array(cellid_arr)

# First continuous state is middle point of first cell
r, x = pc.cheby_ball(merged.list_region[cellid_arr[0]])
x = x.flatten().T
print(discretize.get_cellID(x, merged))
#raw_input()
x_arr = x
x0_arr = x
x1_arr = x
x0_t = 0
x1_t = 0
print(x)
# Filter and observer
## The observed state is initialized in the
## same partition as the unknown state x
theta = 2*np.pi*np.random.random()
x1 = 2*np.random.random()-1
x2 = 2*np.random.random()-1
x3 = 2*np.random.random()-1
xh = x + epsilon*np.array([x1, x2, x3])
print(xh)
xh_arr = xh     #Storage
u_arr = np.zeros([N*num_it, BNW11.shape[1]])
d_arr = np.zeros([N*num_it, ENW11.shape[1]])
C = CNE11                         ###########<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<---------- DEFINITION OF C
y = np.dot(CNE11, x)
y_arr = y
t_11 = []
t_12 = []
t_21 = []
t_22 = []
print(cellid_arr)
notDone = True


for i in range(1, len(cellid_arr)):
    if notDone:
        # For each step, calculate N input signals
        for j in range(N):
            if mode_arr[i-1] == 0:
               dyn = sysDyn1
               env_state = 0
               filters = Ls1
               dynhat = sysDynHat1
            else:
               dyn = sysDyn2
               env_state = 1
               filters = Ls2
               dynhat = sysDynHat2
	    print(i,j)
            print("========================")
            print("Mode/env-state")
            print(mode_arr[i-1], env_state)
            print("========================")
            costs = []
            index_for_cost = []
            dynamicsList = []
            filtsList = []
            filtDynamics = []
            dynamics = []
            filts = []
            print(cellid_arr[i-1], discretize.get_cellID(x, merged))
	    costs = []
	    index_for_costs = []
            dynamics = []# Remove dead-end states from automaton
aut.trimDeadStates()
print(aut.states)

# Simulate
num_it = 20
init_state = {}

states = grsim.grsim([aut],env_states=[init_state], 
             num_it=num_it,
                     deterministic_env=False)

# Store discrete trajectory in np array
cellid_arr = []
mode_arr = []
level_arr = []
for (autID, state) in states:
    print(state)
    mode_arr.append(state.state['mode'])
    cellid_arr.append(state.state['cellID'])
    level_arr.append(state.state['level'])
cellid_arr = np.array(cellid_arr)

# First continuous state is middle point of first cell
r, x = pc.cheby_ball(disc.list_region[cellid_arr[0]])
x = x.flatten().T
print(discretize.get_cellID(x, disc))
print(cellid_arr)
raw_input()
x_arr = x
x0_arr = x
x1_arr = x
x0_t = 0
x1_t = 0
print(x)
# Filter and observer
## The observed state is initialized in the
## same partition as the unknown state x
theta = 2*np.pi*np.random.random()
x1 = 2*np.random.random()-1
x2 = 2*np.random.random()-1
x3 = 2*np.random.random()-1
xh = x + epsilon*np.array([x1, x2, x3])
print(xh)
xh_arr = xh     #Storage
u_arr = np.zeros([N*num_it, BNW11.shape[1]])
d_arr = np.zeros([N*num_it, ENW11.shape[1]])
C = CNE11                         ###########<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<---------- DEFINITION OF C
y = np.dot(CNE11, x)
y_arr = y
t_11 = []
t_12 = []
t_21 = []
t_22 = []
print(cellid_arr)
print(disc.trans)
for i in range(1, num_it):
    # For each step, calculate N input signals
    for j in range(N):
    #Simulate using the observed dynamics
        for index in range(len(sysDyn.list_subsys)):
            if pc.is_inside(sysDyn.list_subsys[index].sub_domain, xh):
                sys_dyn = sysDyn.list_subsys[index]
		sys_dyn_hat = sysDynHat.list_subsys[index]
                A = sysDyn.list_subsys[index].A
                B = sysDyn.list_subsys[index].B
                E = sysDyn.list_subsys[index].E
		K = sysDyn.list_subsys[index].K
		L = Ls[index]
                break
        u_seq = discretize.get_input(xh, sys_dyn_hat, disc, \
                cellid_arr[i-1], cellid_arr[i], N-j, mid_weight=3, Q=np.eye(3*(N-j)), \
                test_result=True, closed_loop=True, conservative = False)
        u0 = u_seq[0,:] # Only the first input should be used

	#u0 = np.random.random(2)-0.155
	#tmp = np.random.random()*1.74-0.58
	#u0 = np.hstack([u0,tmp])        
	#print(u0)
	print(x)
        u_arr[(i-1)*N + j,:] = u0   # Store input
        
        d =  uncertainty * 2 * (np.random.rand(3) - 0.5 )   # Simulate disturbance
        d_arr[(i-1)*N + j,:] = d    # Store disturbance
    
    #Measurement
        y = np.dot(C,x).flatten()
        y_arr = np.vstack([y_arr, y])
        x = np.dot(sys_dyn.A, x).flatten() + np.dot(sys_dyn.B, u0).flatten() + K.flatten() + np.dot(sys_dyn.E, d).flatten()

    #Estimate
        xh = np.dot(sys_dyn.A - np.multiply(L,C), xh).flatten() + K.flatten() + np.dot(B, u0).flatten() + np.dot(L,y).flatten()
        #print(x,xh) 
        xh_arr = np.vstack([xh_arr, xh])
        x_arr = np.vstack([x_arr, x])   # Store state

plt.plot(23+x_arr[:,0])
plt.savefig('AMSplots/noSwitchShrunk.png')
plt.show()
            filtDynamics = []
            filts = []
            for index in range(len(dyn)):

                    for subIndex in range(len(dyn[index].list_subsys)):
                        if pc.is_inside(dyn[index].list_subsys[subIndex].sub_domain,x):
                            dynamics.append(dyn[index].list_subsys[subIndex])
                            filtDynamics.append(dyn[index].list_subsys[subIndex])
                            filts.append(filters[index][subIndex])
                    print(len(dynamics))
                    if (len(dynamics)==0):
                        notDone=False
                        break
            for subbestIndex in range(len(dynamics)):
                        try:
                            tmp = discretize.get_input(xh, filtDynamics[subbestIndex], merged, \
                                cellid_arr[i-1], cellid_arr[i], N-j, mid_weight=0.001, R=1.*np.eye(3*(N-j)), Q=0*1.*np.eye(3*(N-j)), \
                                test_result=True, closed_loop=True, conservative = False, return_cost=True)
			    u_seq = tmp[0]
			    cost = tmp[1]
			    costs.append(cost)
                       	    index_for_cost.append(subbestIndex)
                            A = dynamics[subbestIndex].A
                            B = dynamics[subbestIndex].B
                            E = dynamics[subbestIndex].E
                            K = dynamics[subbestIndex].K
                            switch_state = subbestIndex
                            L = filts[subbestIndex]
                            #print(u_seq)
                            #break
                        except Exception as detail:
                            print(detail)


	    try:
            	subbestIndex = index_for_cost[costs.index(min(costs))]    #Index giving minimum cost
            	print(subbestIndex)
	    	print(index_for_cost)
	    except:
	    	continue
	    print(costs)
	    #raw_input()
	    switch_state = subbestIndex            
            #switch_state = subbest_to_use
            print(switch_state, env_state)
            u0 = u_seq[0,:] # Only the first input should be used
            print(u0)
            u_arr[(i-1)*N + j,:] = u0   # Store input
            
            d =  uncertainty * 2 * (np.random.rand(3) - 0.5 )   # Simulate disturbance
            d_arr[(i-1)*N + j,:] = d    # Store disturbance
               
            #Measurement
            y = np.dot(C,x).flatten()
            y_arr = np.vstack([y_arr, y])
            x = np.dot(A, x).flatten() + np.dot(B, u0).flatten() + K.flatten()+ np.dot(E, d).flatten()

            #Estimate
            xh = np.dot(A - np.dot(L,C), xh).flatten() + np.dot(B, u0).flatten() + K.flatten() + np.dot(L,y).flatten()
            print(x,xh) 
            xh_arr = np.vstack([xh_arr, xh])
            if env_state == 0:        
               x0_arr = np.vstack([x0_arr, x])   # Store state
               x0_t = np.vstack([x0_t, (i-1)*N + j])
            if env_state == 1:   
               x1_arr = np.vstack([x1_arr, x])   # Store state
               x1_t = np.vstack([x1_t, (i-1)*N + j])
            x_arr = np.vstack([x_arr, x])

            if env_state == 0 and switch_state ==0:
                          t_11.append((i-1)*N + j)

            if env_state == 0 and switch_state ==1:
                          t_12.append((i-1)*N + j)

            if env_state == 1 and switch_state ==0:
                          t_21.append((i-1)*N + j)

            if env_state == 1 and switch_state ==1:
                          t_22.append((i-1)*N + j)
"""
print(x_arr)
Tc = 23+x_arr[:,0]
Tcest = 23+xh_arr[:,0]

Tx = -5+x_arr[:,1]
Txest = -5+xh_arr[:,1]

pv = 10**5*(1.368+x_arr[:,2])
pvest = 10**5*(1.368+xh_arr[:,2])

e_arr = x_arr-xh_arr

Tcerr = e_arr[:,0]
Txerr = e_arr[:,1]
pverr = 10**5*e_arr[:,2]

c1 = 0.15 + u_arr[:,0]
c2 = 0.185 + u_arr[:,1]
Wa = 2.49+u_arr[:,2]

t = np.array(range(len(Tc)))/10.0
#plt.subplot(331)
plt.clf()
plt.plot(t,Tc, 'b', label=r'Cabin temperature')
plt.plot(t,Tcest, 'r', label=r'Estimated cabin temperature')
plt.title('Cabin temperature')
plt.legend()
plt.xlabel('Time in s')
plt.ylabel(r'$^{\circ}$C')
plt.savefig('AMSplots/AMSright/TcFilter'+'beta='+str(beta)+'.png')

#plt.subplot(332)
plt.clf()

plt.plot(t,Tx, 'r', label=r'HX temperature')
plt.plot(t,Txest, 'b-.', label=r'Estimated HX temperature')
plt.title('HX temperature')
plt.legend()
plt.xlabel('Time in s')
plt.ylabel('$^{\circ}$C')
plt.savefig('AMSplots/AMSright/TxFilter'+'beta='+str(beta)+'.png')


#plt.subplot(333)
plt.clf()

plt.plot(t,pv, 'g', label=r'Fork pressure')
plt.plot(t,pvest, 'y-.', label=r'Estimated fork pressure')

plt.title('Fork pressure')
plt.xlabel('Time in s')
plt.ylabel('Pa')
plt.legend()
plt.savefig('AMSplots/AMSright/pvFilter'+'beta='+str(beta)+'.png')


#plt.subplot(334)
plt.clf()

plt.plot(t,c1[:len(Tc)], 'm', label='Valve coefficient 1')
plt.plot(t,c2[:len(Tc)], 'c.-', label='Valve coefficient 2')
plt.title('Valve coefficients')
plt.legend()
plt.xlabel('Time in s')
plt.ylabel('Fraction of full opening')
plt.savefig('AMSplots/AMSright/CsFilter'+'beta='+str(beta)+'.png')

#plt.subplot(335)
plt.clf()

plt.plot(t,Wa[:len(Tc)], 'k', label='Inflow of cold air')

plt.title(r'$W_a$')
plt.xlabel('Time in s')
plt.ylabel('Kg/s')
plt.legend()
plt.savefig('AMSplots/AMSright/WaFilter'+'beta='+str(beta)+'.png')


#plt.subplot(336)
plt.clf()

plt.plot(t,Tcerr[:len(Tc)], 'b', label=r'Cabin temperature')
plt.plot(t,Txerr[:len(Tc)], 'r', label=r'HX temperature')

plt.title('Temperature errors')
plt.legend()
plt.xlabel('Time in s')
plt.ylabel(r'$^{\circ}$')
plt.savefig('AMSplots/AMSright/errorsFilter'+'beta='+str(beta)+'.png')


#plt.subplot(337)
plt.clf()

plt.plot(t,pverr, 'k', label=r'Fork pressure')
plt.title('Fork pressure error')
plt.xlabel('Time in s')
plt.ylabel('Pa')
plt.savefig('AMSplots/AMSright/pverrorFilter'+'beta='+str(beta)+'.png')

#plt.subplot(338)
plt.clf()

plt.plot(t,d_arr[:len(Tc),0], 'b', label='Disturbance in $T_c$ in $^{\circ}$')
plt.plot(t,d_arr[:len(Tc),1], 'r',  label='Disturbance in $T_x$ in $^{\circ}$')
plt.plot(t,d_arr[:len(Tc),2], 'g', label=r'Disturbance in $p_v$ in $10^5$ Pa')
plt.title('Disturbances')
plt.xlabel('Time in s')
plt.ylabel('Indicated units')
plt.savefig('AMSplots/AMSright/disturbancesFilter'+'beta='+str(beta)+'.png')

#plt.subplot(339)
plt.clf()

try:
    plt.plot(t_11, np.zeros((len(t_11),1)), 'co', label=r'$(\sigma, e) = (0,0)$')
except:
    print("No mode 0,0")
try:
    plt.plot(t_12, 1+np.zeros((len(t_12),1)), 'mo', label=r'$(\sigma, e) = (1,0)$')
except:
    print("No mode 1,0")

try:
    plt.plot(t_21, 2+np.zeros((len(t_21),1)), 'ro', label=r'$(\sigma, e) = (0,1)$')
except:
    print("No mode 0,1")

try:
    plt.plot(t_22, 3+np.zeros((len(t_22),1)), 'yo', label=r'$(\sigma, e) = (1,1)$')
except:
    print("No mode 1,1")
plt.xlabel('Time in s')
plt.legend()
plt.title('Illustration of different modes over time')
try:
   plt.tight_layout()
except:
      print("no tight layout, sorry")
plt.savefig('AMSplots/modesFilter'+'beta='+str(beta)+'.png')
plt.show()

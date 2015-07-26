#!/usr/bin/env python
# Import the packages that we need
from tulip import spec, synth
from tulip.transys import machines

#
# Environment specification
#
# The environment can issue a park signal that the robot must respond
# to by moving to the lower left corner of the grid.  We assume that
# the park signal is turned off infinitely often.
#
env_vars = {}
env_vars['w']={'low','med','high'}
env_init = set('w=low')         
env_safe = set('(a=1) -> ((w="low") || (w="medium"))')                
env_prog = set('(w="high")')            

#
# System dynamics
#
# The system specification describes how the system is allowed to move
# and what the system is required to do in response to an environmental
# action.  
#
sys_vars = {}
sys_vars['a'] = (1, 2)
sys_init = {'a=2'}
sys_safe = {'a=2'}
sys_prog = set('a=2')                # empty set


# Create a GR(1) specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

#
# Controller synthesis
#
# At this point we can synthesize the controller
# using one of the available methods.
# Here we make use of jtlv.
#
mealy_controller = synth.synthesize('gr1c', specs)

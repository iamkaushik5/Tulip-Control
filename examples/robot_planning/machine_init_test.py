
from tulip import spec, synth

env_vars ={}
env_vars['level'] = (0,1)
sys_vars ={'trackfl':(0,2), 'loc':(0,5)}

env_init = {'level=1'}
env_prog = set()
env_safe = set() 

#sys_init = {'loc<=3'}
sys_init = {'(loc=0) || (loc=1) || (loc=2) || (loc=3)'}
sys_safe = {
    '(loc=0) -> X ((loc=0) || (loc=1))',
    '(loc=1) -> X ((loc=0) || (loc=1) || (loc=2))',
    '(loc=2) -> X ((loc=1) || (loc=2) || (loc=3))',
    '(loc=3) -> X ((loc=2) || (loc=3) || (loc=4))',
    '(loc=4) -> X ((loc=3) || (loc=4) || (loc=5))',
    '(loc=5) -> X ((loc=4) || (loc=5))',
    '(trackfl=0) -> (loc=1)',
    '(trackfl=1) -> (loc=4)',
    '(trackfl=0) -> X ((trackfl=0) || !(level=0))',
    '(trackfl=1) -> X ((trackfl=1) || !(level=1))'
}

sys_prog = {
    '(trackfl=0) || !(level=0)',
    '(trackfl=1) || !(level=1)'
}


# Create the specification
spc = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

from tulip.interfaces import jtlvint, gr1cint
ctrl1 = gr1cint.synthesize(spc)
ctrl = synth.synthesize('gr1c', spc)

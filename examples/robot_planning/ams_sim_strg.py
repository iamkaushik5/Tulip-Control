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
from tulip.transys.executions import MachineInputSequence

#fname = 'scl10N1_pwa_'
fname = 'scl7N1_'

#sys_ts = pickle.load(open(fname+'disc_dyn', 'r') )

#cont_sys = pickle.load(open(fname+'cont_sys', 'r') )

sys_ts = pickle.load(open(fname+'ts', 'r') )

specs = pickle.load(open(fname+'specs', 'r') )

ctrl = pickle.load(open(fname+'ctrl', 'r') )

x = MachineInputSequence(ctrl)

x.set_input_sequence('level',[1])

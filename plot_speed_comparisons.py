from __future__ import print_function, division

from time import time
import numpy as np
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt

import openmdao.api as om


label_keys = {
    'num_outputs' : 'Num outputs',
    'num_inputs' : 'Num inputs',
    'bandwidth' : 'Bandwidth',
    }
    
for varied_term in label_keys:
    filename = f'timing_data_{varied_term}.pkl'

    data = OrderedDict()
    data['Analytic Sparse'] = None
    data['Analytic Dense'] = None
    data['Approximated'] = None
    data['Approximated Colored'] = None
    data['JAX Jacobian'] = None

    with open(filename, 'rb') as f:
        output_data = pickle.load(f)

    timing_data = output_data['timing_data']
    nns = output_data['nns']

    plt.figure()

    for i_method, key in enumerate(data):
        plt.loglog(nns, timing_data[:, i_method], label=key)
        
    plt.legend()
    plt.xlabel(label_keys[varied_term])
    plt.ylabel('Time to compute total derivs, secs')
    plt.savefig(f'speed_comparison_{varied_term}.pdf')
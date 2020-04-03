from __future__ import print_function, division

from time import time
import numpy as np
from collections import OrderedDict
import pickle
import matplotlib
import matplotlib.pyplot as plt

import openmdao.api as om


label_keys = {
    'num_outputs' : 'Num outputs',
    'num_inputs' : 'Num inputs',
    'bandwidth' : 'Bandwidth',
    }
    
comparison_type_keys = ['total_derivs', 'opt']
    
for comparison_type in comparison_type_keys:
    for varied_term in label_keys:
        filename = f'{comparison_type}_{varied_term}.pkl'

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

        plt.figure(figsize=(7, 4))
        

        for i_method, key in enumerate(data):
            plt.loglog(nns, timing_data[:, i_method], label=key, lw=2)
            
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel(label_keys[varied_term])
        
        if comparison_type == 'total_derivs':
            plt.ylabel('Time to compute total derivs, secs')
        else:
            plt.ylabel('Time to complete optimization, secs')
        
        ax = plt.gca()
        ax.set_xticks(nns)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.minorticks_off()
        
        plt.tight_layout()    
        plt.savefig(f'{comparison_type}_{varied_term}.pdf')
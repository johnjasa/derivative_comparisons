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
legend_indices = [4, 2, 3, 1, 0]

print('Plotting individual figures')
# Individual plots
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

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend([handles[i] for i in legend_indices], [labels[i] for i in legend_indices], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel(label_keys[varied_term])

        if comparison_type == 'total_derivs':
            plt.ylabel('Time to compute total derivs, secs')
        else:
            plt.ylabel('Time to complete optimization, secs')

        ax.set_xticks(nns)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.minorticks_off()

        plt.tight_layout()    
        plt.savefig(f'{comparison_type}_{varied_term}.png', dpi=400)
        
        
print('Plotting set of figures')
# Group of 2x3 plots
fig, axarr = plt.subplots(2, 3, figsize=(11, 6), sharex='col', sharey='row')
for i, comparison_type in enumerate(comparison_type_keys):
    for j, varied_term in enumerate(label_keys):
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

        for i_method, key in enumerate(data):
            axarr[i, j].loglog(nns, timing_data[:, i_method], label=key, lw=2)
        
        if i==1:
            axarr[i, j].set_xlabel(label_keys[varied_term])
        
        if j==0:
            if comparison_type == 'total_derivs':
                axarr[i, j].set_ylabel('Time to compute total derivs, secs')
            else:
                axarr[i, j].set_ylabel('Time to complete optimization, secs')
        
        if 'bandwidth' in varied_term:
            axarr[i, j].set_xticks(nns)
        else:
            axarr[i, j].set_xticks(np.hstack((nns[0], nns[1::2])))
            
        axarr[i, j].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axarr[i, j].minorticks_off()
        

plt.sca(axarr[0, 2])
handles, labels = axarr[0, 2].get_legend_handles_labels()
plt.legend([handles[i] for i in legend_indices], [labels[i] for i in legend_indices], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()    
plt.savefig('all_plots.png', dpi=400)

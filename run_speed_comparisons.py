from __future__ import print_function, division

from time import time
import numpy as np
from collections import OrderedDict
import pickle

import openmdao.api as om
from groups import MatrixGroup


nns = [2**i for i in range(4, 14)]
num_nns = len(nns)
num_repeats = 10

data = OrderedDict()
data['Analytic Dense'] = np.zeros((num_nns, num_repeats))
data['Analytic Sparse'] = np.zeros((num_nns, num_repeats))
data['JAX Jacobian'] = np.zeros((num_nns, num_repeats))
data['Approximated'] = np.zeros((num_nns, num_repeats))
data['Approximated Colored'] = np.zeros((num_nns, num_repeats))

timing_data = np.zeros((num_nns, len(data)))

for i_method, key in enumerate(data):
    
    print()
    print(i_method, key)    
    
    if key == 'Analytic Dense':
        from matrix_comp_analytic import MatrixComp
    if key == 'Analytic Sparse':
        from matrix_comp_analytic_sparse import MatrixComp
    if key == 'JAX Jacobian':
        from matrix_comp_jax import MatrixComp
    if key == 'Approximated':
        from matrix_comp_approx import MatrixComp
    if key == 'Approximated Colored':
        from matrix_comp_approx_colored import MatrixComp
        
    for i_nn, num_outputs in enumerate(nns):
        
        if num_outputs > 100 and 'JAX' in key:
            timing_data[i_nn, i_method] = np.nan
            continue
        
        print('Running {} cases of {} num_outputs'.format(num_repeats, num_outputs))
        
        num_inputs = 5
        # num_outputs = 3
        bandwidth = 2
        random_seed = 310

        prob = om.Problem()
        prob.model = MatrixGroup(comp_type=MatrixComp, num_inputs=num_inputs, num_outputs=num_outputs, bandwidth=bandwidth, random_seed=random_seed)
        
        prob.setup()
        prob.run_model()
        
        for i_repeat in range(num_repeats):

            pre_time = time()
            
            prob.compute_totals(['obj'], ['x'])
            
            post_time = time()
            duration = post_time - pre_time
            
            data[key][i_nn, i_repeat] = duration
            
        timing_data[i_nn, i_method] = np.mean(data[key][i_nn, :])
        
        
output_data = {
    'timing_data' : timing_data,
    'nns' : nns,
    }
    
with open('timing_data.pkl', 'wb') as f:
    pickle.dump(output_data, f)
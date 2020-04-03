from __future__ import print_function, division

from time import time
import numpy as np
from collections import OrderedDict
import pickle

import openmdao.api as om
from groups import MatrixGroup


random_seed = 314
num_repeats = 1

keys = ['num_outputs', 'num_inputs', 'bandwidth']

comparison_type_keys = ['total_derivs', 'opt']
    
for comparison_type in comparison_type_keys:
    for varied_term in keys:
        
        num_inputs = 5
        num_outputs = 3
        bandwidth = 2    
        
        if comparison_type == 'total_derivs':
            if 'output' in varied_term:
                nns = [2**i for i in range(2, 12)]
            elif 'input' in varied_term:
                nns = [2**i for i in range(2, 12)]
            elif 'bandwidth' in varied_term:
                nns = [2**i for i in range(1, 6)]
                num_inputs = 200
                num_outputs = 100
        elif comparison_type == 'opt':
            if 'output' in varied_term:
                nns = [2**i for i in range(2, 12)]
            elif 'input' in varied_term:
                nns = [2**i for i in range(2, 12)]
            elif 'bandwidth' in varied_term:
                nns = [2**i for i in range(1, 6)]
                num_inputs = 200
                num_outputs = 100

        num_nns = len(nns)

        data = OrderedDict()
        data['Analytic Sparse'] = np.zeros((num_nns, num_repeats))
        data['Analytic Dense'] = np.zeros((num_nns, num_repeats))
        data['Approximated'] = np.zeros((num_nns, num_repeats))
        data['Approximated Colored'] = np.zeros((num_nns, num_repeats))
        data['JAX Jacobian'] = np.zeros((num_nns, num_repeats))

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
                
            for i_nn, nn in enumerate(nns):
                
                if comparison_type == 'total_derivs':
                    if nn > 300 and 'JAX' in key and 'output' in varied_term:
                        timing_data[i_nn, i_method] = np.nan
                        continue
                    elif nn > 70 and 'JAX' in key and 'bandwidth' in varied_term:
                        timing_data[i_nn, i_method] = np.nan
                        continue
                else:
                    if nn > 50 and 'JAX' in key and 'output' in varied_term:
                        timing_data[i_nn, i_method] = np.nan
                        continue
                    elif nn > 20 and 'JAX' in key and 'bandwidth' in varied_term:
                        timing_data[i_nn, i_method] = np.nan
                        continue
                
                if 'outputs' in varied_term:
                    num_outputs = nn
                elif 'inputs' in varied_term:
                    num_inputs = nn
                elif 'bandwidth' in varied_term:
                    bandwidth = nn
                    
                print(f'Running {num_repeats} cases of {num_inputs} num_inputs, {num_outputs} num_outputs, {bandwidth} bandwidth')
                
                prob = om.Problem()
                prob.model = MatrixGroup(comp_type=MatrixComp, num_inputs=num_inputs, num_outputs=num_outputs, bandwidth=bandwidth, random_seed=random_seed)
                
                prob.driver = om.pyOptSparseDriver()
                prob.driver.options['optimizer'] = 'SNOPT'
                prob.driver.opt_settings['Major optimality tolerance'] = 1e-8
                prob.driver.opt_settings['Major print level'] = 0
                prob.driver.opt_settings['Minor print level'] = 0
                prob.driver.opt_settings['iPrint'] = 0
                
                prob.model.add_design_var('x', lower=-10, upper=10)
                prob.model.add_objective('obj')
                
                
                if 'inputs' in varied_term or 'bandwidth' in varied_term:
                    prob.setup(mode='rev')
                else:
                    prob.setup(mode='fwd')
                    
                prob['x'][:] = 2.
                prob.run_model()
                
                for i_repeat in range(num_repeats):

                    pre_time = time()
                    
                    if comparison_type == 'total_derivs':
                        prob.compute_totals(['obj'], ['x'])
                    elif comparison_type == 'opt':
                        print('running opt')
                        prob.run_driver()
                    
                    post_time = time()
                    duration = post_time - pre_time
                    
                    data[key][i_nn, i_repeat] = duration
                    
                timing_data[i_nn, i_method] = np.mean(data[key][i_nn, :])
                
                
        output_data = {
            'timing_data' : timing_data,
            'nns' : nns,
            'num_inputs' : num_inputs,
            'num_outputs' : num_outputs,
            'bandwidth' : bandwidth,
            'varied_term' : varied_term,
            }
            
        with open(f'{comparison_type}_{varied_term}.pkl', 'wb') as f:
            pickle.dump(output_data, f)
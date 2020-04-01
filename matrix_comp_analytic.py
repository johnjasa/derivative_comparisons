import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt


class MatrixComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_inputs', default=2)
        self.options.declare('num_outputs', default=5)
        self.options.declare('bandwidth', default=2)
        self.options.declare('random_seed', default=314)
        
    def setup(self):
        self.add_input('x', shape=self.options['num_inputs'])
        self.add_output('y', shape=self.options['num_outputs'])
        
        self.declare_partials('y', 'x')
        
        np.random.seed(self.options['random_seed'])
        self.random_array = np.random.random_sample(self.options['num_inputs'])
        
    def compute(self, inputs, outputs):
        num_inputs = self.options['num_inputs']
        num_outputs = self.options['num_outputs']
        bandwidth = self.options['bandwidth']
        x = inputs['x']
        y = outputs['y']
        
        x_and_random = x + self.random_array
        tiled_x = np.tile(x_and_random, int(np.ceil(num_outputs / num_inputs) + bandwidth))
        
        for i in range(num_outputs):
            y[i] = np.sum(tiled_x[i:i+bandwidth]**4)
        
    def compute_partials(self, inputs, partials):
        num_inputs = self.options['num_inputs']
        num_outputs = self.options['num_outputs']
        bandwidth = self.options['bandwidth']
        x = inputs['x']
        
        x_and_random = x + self.random_array
        tiled_x = np.tile(x_and_random, int(np.ceil(num_outputs / num_inputs) + bandwidth))
        
        for i in range(num_outputs):# - bandwidth - 2):
            np.put(partials['y', 'x'][i], range(i, i+bandwidth), 4 * tiled_x[i:i+bandwidth]**3, mode='wrap')
            
            
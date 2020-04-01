from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import jax
import numpy as onp
import openmdao.api as om
import time


def compute_func(x, random_array, num_inputs, num_outputs, bandwidth):
    y = np.zeros(num_outputs)
    x_and_random = x + random_array
    tiled_x = np.tile(x_and_random, int(np.ceil(num_outputs / num_inputs) + bandwidth))
    
    # time.sleep(0.1)

    for i in range(num_outputs):
        # y[i] = np.sum(tiled_x[i:i+bandwidth]**4)
        y = jax.ops.index_update(y, i, np.sum(tiled_x[i:i+bandwidth]**4))
        
    return y

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
        
        onp.random.seed(self.options['random_seed'])
        self.random_array = onp.random.random_sample(self.options['num_inputs'])
        
    def compute(self, inputs, outputs):
        num_inputs = self.options['num_inputs']
        num_outputs = self.options['num_outputs']
        bandwidth = self.options['bandwidth']
        x = inputs['x']

        outputs['y'] = compute_func(x, self.random_array, num_inputs, num_outputs, bandwidth)
        
    def compute_partials(self, inputs, partials):
        num_inputs = self.options['num_inputs']
        num_outputs = self.options['num_outputs']
        bandwidth = self.options['bandwidth']
        x = inputs['x']
    
        partials['y', 'x'] = jax.jacobian(compute_func)(x, self.random_array, num_inputs, num_outputs, bandwidth)
    
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt





class ObjectiveComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_outputs', default=5)
        
    def setup(self):
        self.add_input('y', shape=self.options['num_outputs'])
        self.add_output('obj')
        
        self.declare_partials('obj', 'y', val=1.)
        
    def compute(self, inputs, outputs):
        num_outputs = self.options['num_outputs']
        outputs['obj'] = np.sum(inputs['y'])
        
        
class MatrixGroup(om.Group):
    
    def initialize(self):
        self.options.declare('comp_type')
        self.options.declare('num_inputs', default=2)
        self.options.declare('num_outputs', default=5)
        self.options.declare('bandwidth', default=2)
        self.options.declare('random_seed', default=314)
        
    def setup(self):
        num_inputs = self.options['num_inputs']
        num_outputs = self.options['num_outputs']
        bandwidth = self.options['bandwidth']
        random_seed = self.options['random_seed']
        MatrixComp = self.options['comp_type']
        
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['x'])
        indeps.add_output('x', shape=num_inputs, val=0.)
        
        self.add_subsystem('matrix_comp', MatrixComp(num_inputs=num_inputs, num_outputs=num_outputs, bandwidth=bandwidth, random_seed=random_seed), promotes=['x', 'y'])
        
        self.add_subsystem('objective_comp', ObjectiveComp(num_outputs=num_outputs), promotes=['y', 'obj'])


if __name__ == "__main__":
    # from matrix_comp_jax import MatrixComp
    # from matrix_comp_jax_jvp import MatrixComp
    # from matrix_comp_analytic import MatrixComp
    from matrix_comp_analytic_sparse import MatrixComp

    num_inputs = 5
    num_outputs = 8
    bandwidth = 2
    random_seed = 310
    comp_type = MatrixComp

    prob = om.Problem()
    prob.model = MatrixGroup(num_inputs=num_inputs, num_outputs=num_outputs, bandwidth=bandwidth, random_seed=random_seed, comp_type=MatrixComp)

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-8

    prob.model.add_design_var('x', lower=-10, upper=10)
    prob.model.add_objective('obj')

    # prob.model.declare_coloring(show_summary=True, show_sparsity=True)
    # 
    # # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
    # prob.model.approx_totals(method='cs')

    prob.setup()
    prob.set_solver_print(level=0)
    prob['x'][:] = 2.

    check_partials_data = prob.check_partials()

    # # plot with defaults
    # om.partial_deriv_plot('y', 'x', check_partials_data, binary=False)

    # prob.run_driver()

    # prob.model.list_outputs(print_arrays=True)
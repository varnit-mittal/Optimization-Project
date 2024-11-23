import numpy as np
from src.de import CDE, ADE
from src.core import benchmarks

problem = {'cost_function': benchmarks.paraboloid,'ndim_problem': 5,'lower_bound': -10.0*np.ones((5,)), 'upper_bound': 10.0*np.ones((5,))}
options = {'max_evals': 10000,'seed_rng': 0, 'n_individuals':100, 'verbose':10}
cde = ADE(problem, options)
results = cde.optimize()
cde.print_report(results)
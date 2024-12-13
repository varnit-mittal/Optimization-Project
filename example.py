import numpy as np
from src.pso import SPSO
from src.core import benchmarks

problem = {'cost_function': benchmarks.paraboloid,'ndim_problem': 5,'lower_bound': -100.0, 'upper_bound': 100.0}
options = {'max_evals': 10000,'seed_rng': 10, 'n_individuals':100}
cmaes = SPSO(problem, options)
results = cmaes.optimize()
cmaes.print_report(results)
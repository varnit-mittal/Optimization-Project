import numpy as np

class Optimizer(object):
    def __init__(self, problem, options):
        self.cost_function = problem.get('cost_function')
        self.ndim_problem = problem.get('ndim_problem')
        assert self.ndim_problem > 0
        self.upper_bound = problem.get('upper_bound')
        self.lower_bound = problem.get('lower_bound')

        self.options = options
        self.max_evals = options.get('max_evals', np.inf)
        self.n_individuals = options.get('n_individuals')
        self.seed_rng = options.get('seed_rng')
        if self.seed_rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed_rng)
        self.seed_initialization = options.get('seed_initialization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_initialization = np.random.default_rng(self.seed_initialization)
        self.seed_optimization = options.get('seed_optimization', self.rng.integers(np.iinfo(np.int64).max))
        self.rng_optimization = np.random.default_rng(self.seed_optimization)
        self.verbose = options.get('verbose', None)
        self.is_success = "no_term"
        self.n_evals = options.get('n_evals', 0)
        self.y_best, self.x_best = options.get('y_best', np.inf), None

    def _evaluate_fitness(self, x, args=None):
        if args is None:
            y = self.cost_function(x)
        else:
            y = self.cost_function(x, args=args)
        self.n_evals += 1
        if y < self.y_best:
            self.x_best, self.y_best = np.copy(x), y
        return float(y)

    def _check_success(self):
        if self.n_evals >= self.max_evals:
            self.is_success = 0
            return True
        else:
            self.is_success = 1
            return False

    def _collect(self):
        return {'x_best': self.x_best,
                'y_best': self.y_best,
                'is_success': self.is_success}

    def optimize(self, cost_function=None):
        if cost_function is not None:
            self.cost_function = cost_function
        fitness = []
        return fitness
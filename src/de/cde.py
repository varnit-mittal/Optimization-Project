import numpy as np
from ..core.optimizer import Optimizer

class CDE(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        assert self.n_individuals > 0
        self.mu = options.get('mu', 0.5)
        self.cr = options.get('cr', 0.9)
        self.n_gens = 0

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.lower_bound, self.upper_bound,
            size=(self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_success():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        v = np.empty((self.n_individuals, self.ndim_problem))
        return x, y, v

    def mutate(self, x=None, v=None):
        for i in range(self.n_individuals):
            r = self.rng_optimization.permutation(self.n_individuals)[:4]
            r = r[r != i][:3]
            v[i] = x[r[0]] + self.mu*(x[r[1]] - x[r[2]])
        return v

    def crossover(self, v=None, x=None):
        for i in range(self.n_individuals):
            j_r = self.rng_optimization.integers(self.ndim_problem)
            tmp = v[i, j_r]
            co = self.rng_optimization.random(self.ndim_problem) > self.cr
            v[i, co] = x[i, co]
            v[i, j_r] = tmp
        return v

    def select(self, v=None, x=None, y=None, args=None):
        for i in range(self.n_individuals):
            if self._check_success():
                break
            yy = self._evaluate_fitness(v[i], args)
            if yy < y[i]:
                x[i], y[i] = v[i], yy
        return x, y

    def iterate(self, x=None, y=None, v=None, args=None):
        v = self.mutate(x, v)
        v = self.crossover(v, x)
        x, y = self.select(v, x, y, args)
        self.n_gens += 1
        return x, y

    def _print_verbose_info(self):
        if self.verbose:
            if (self.n_gens % self.verbose == 0) or self._check_success() :
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    def _collect(self):
        self._print_verbose_info()
        results = Optimizer._collect(self)
        results['n_gens'] = self.n_gens
        return results

    def optimize(self, fitness_function=None, args=None):
        if fitness_function is not None:
            self.fitness_function = fitness_function
        x, y, v = self.initialize(args)
        while not self._check_success():
            self._print_verbose_info()
            x, y = self.iterate(x, y, v, args)
        results = self._collect()
        results['n_gens'] = self.n_gens + 1
        return results
    
    def print_report(self,results):
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

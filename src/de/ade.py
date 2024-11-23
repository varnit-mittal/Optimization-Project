import numpy as np
from scipy.stats import cauchy
from ..core.optimizer import Optimizer

class ADE(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:
            self.n_individuals = 100
        assert self.n_individuals > 0
        # ADE-specific parameters
        self.n_mu = options.get('n_mu', 0.5)  # mean of normal distribution for crossover
        self.median = options.get('median', 0.5)  # location of Cauchy distribution for mutation
        self.p = options.get('p', 0.05)  # greediness parameter
        self.c = options.get('c', 0.1)  # adaptation rate
        self.n_gens = 0

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.lower_bound, self.upper_bound,
            size=(self.n_individuals, self.ndim_problem))
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_success():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        # Initialize archive of inferior solutions
        a = np.empty((0, self.ndim_problem))
        return x, y, a

    def mutate(self, x, y, a):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))
        f_mu = np.empty(self.n_individuals)
        
        # Get indices of p% best solutions
        p_best_size = max(1, int(self.p * self.n_individuals))
        best_indices = np.argsort(y)[:p_best_size]
        
        # Combine current population and archive
        x_union = np.vstack((x, a)) if len(a) > 0 else x
        
        for i in range(self.n_individuals):
            # Generate mutation factor from Cauchy distribution
            f_mu[i] = cauchy.rvs(loc=self.median, scale=0.1, random_state=self.rng_optimization)
            f_mu[i] = min(1.0, max(0.0, f_mu[i]))
            
            # Select indices for mutation
            x_best_idx = self.rng_optimization.choice(best_indices)
            r1 = self.rng_optimization.choice([j for j in range(self.n_individuals) if j != i])
            r2 = self.rng_optimization.choice([j for j in range(len(x_union)) if j != i and j != r1])
            
            # Generate mutant
            x_mu[i] = x[i] + f_mu[i] * (x[x_best_idx] - x[i]) + f_mu[i] * (x[r1] - x_union[r2])
        
        return x_mu, f_mu

    def crossover(self, x_mu, x):
        x_cr = np.copy(x)
        cr_values = np.minimum(1.0, np.maximum(0.0, 
            self.rng_optimization.normal(self.n_mu, 0.1, size=self.n_individuals)))
        
        for i in range(self.n_individuals):
            j_rand = self.rng_optimization.integers(self.ndim_problem)
            mask = (self.rng_optimization.random(self.ndim_problem) < cr_values[i])
            mask[j_rand] = True
            x_cr[i, mask] = x_mu[i, mask]
        
        return x_cr, cr_values

    def select(self, x_cr, x, y, a, f_mu, cr_values, args):
        successful_f = []
        successful_cr = []
        
        for i in range(self.n_individuals):
            if self._check_success():
                break
            
            y_trial = self._evaluate_fitness(x_cr[i], args)
            
            if y_trial < y[i]:
                # Keep track of successful parameters
                successful_f.append(f_mu[i])
                successful_cr.append(cr_values[i])
                
                # Update solution
                a = np.vstack((a, x[i])) if len(a) > 0 else x[i].reshape(1, -1)
                x[i] = x_cr[i]
                y[i] = y_trial
        
        # Update control parameters
        if len(successful_cr) > 0:
            self.n_mu = (1 - self.c) * self.n_mu + self.c * np.mean(successful_cr)
            if len(successful_f) > 0:
                self.median = (1 - self.c) * self.median + \
                    self.c * (np.sum(np.array(successful_f)**2) / np.sum(successful_f))
        
        # Trim archive size
        if len(a) > self.n_individuals:
            indices = self.rng_optimization.choice(len(a), self.n_individuals, replace=False)
            a = a[indices]
            
        return x, y, a

    def iterate(self, x, y, a, args):
        # Generate mutants
        x_mu, f_mu = self.mutate(x, y, a)
        
        # Perform crossover
        x_cr, cr_values = self.crossover(x_mu, x)
        
        # Selection and parameter adaptation
        x, y, a = self.select(x_cr, x, y, a, f_mu, cr_values, args)
        
        self.n_gens += 1
        return x, y, a

    def optimize(self, fitness_function=None, args=None):
        if fitness_function is not None:
            self.fitness_function = fitness_function
            
        x, y, a = self.initialize(args)
        
        while not self._check_success():
            self._print_verbose_info()
            x, y, a = self.iterate(x, y, a, args)
            
        results = self._collect()
        results['n_gens'] = self.n_gens + 1
        return results
        
    def _print_verbose_info(self):
        if self.verbose:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    def _collect(self):
        self._print_verbose_info()
        results = Optimizer._collect(self)
        results['n_gens'] = self.n_gens
        return results
    
    def print_report(self,results):
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")
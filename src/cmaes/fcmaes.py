import numpy as np
from ..core.optimizer import Optimizer

class FCMAES(Optimizer):
    def __init__(self, problem, options):
        # Initialize the base Optimizer class
        Optimizer.__init__(self, problem, options)
        
        # Set default population size if not provided
        self.m = self.n_individuals
        if self.n_individuals is None:
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # Population size formula
        assert self.n_individuals >= 2  # Minimum population size requirement
        
        # Number of parents to use for recombination
        self.n_parents = options.get('n_parents', int(self.n_individuals / 2))
        assert self.n_parents <= self.n_individuals and self.n_parents > 0  # Ensure valid range
        
        # Initialize mean and step size
        self.mean = options.get('mean', options.get('x'))
        self.sigma = options.get('sigma')
        assert self.sigma > 0  # Ensure positive step size
        
        # Compute weights and effective population size
        w_a = np.log((self.n_individuals + 1.0) / 2.0) - np.log(np.arange(self.n_individuals) + 1.0)
        self._mu_eff = np.square(np.sum(w_a[:self.n_parents])) / np.sum(np.square(w_a[:self.n_parents]))
        self._w = np.where(w_a >= 0, 1.0 / np.sum(w_a[w_a > 0]) * w_a, 0.1 / (-np.sum(w_a[w_a < 0])) * w_a)
        
        # Algorithm parameters
        self.c = 2.0 / (self.ndim_problem + 5.0)
        self.c_1 = 1.0 / (3.0 * np.sqrt(self.ndim_problem) + 5.0)
        self.c_s = 0.3
        self.q_star = 0.27
        self.d_s = 1.0
        self.n_steps = self.ndim_problem
        
        # Constants for updating steps
        self._x_1 = 1.0 - self.c_1
        self._x_2 = np.sqrt((1.0 - self.c_1) * self.c_1)
        self._x_3 = np.sqrt(self.c_1)
        self._p_1 = 1.0 - self.c
        self._p_2 = None  # Updated during initialization
        self._rr = None  # Ranks array for success rule
        
        # Initialize counters and best solution variables
        self.n_gens = 0
        self.x_best = None
        self.y_best = np.inf  # Start with the worst possible fitness

    def initialize(self, is_restart=False):
        # Set path parameters and ranks
        self._p_2 = np.sqrt(self.c * (2.0 - self.c) * self._mu_eff)
        self._rr = np.arange(self.n_parents * 2) + 1  # Ranks for success rule
        
        # Initialize population
        x = np.empty((self.n_individuals, self.ndim_problem))
        mean = self.rng_initialization.uniform(self.lower_bound, self.upper_bound) if self.mean is None else np.copy(self.mean)
        self.mean = np.copy(mean)
        
        # Evaluate initial population
        for i in range(self.n_individuals):
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[i] = mean + self.sigma * z  # Sample from normal distribution
            y_i = self._evaluate_fitness(x[i])  # Evaluate fitness
            if y_i < self.y_best:  # Update best solution
                self.y_best = y_i
                self.x_best = np.copy(x[i])
        
        # Initialize algorithm variables
        y = np.empty((self.n_individuals,))
        p = np.zeros((self.ndim_problem,))
        p_hat = np.zeros((self.m, self.ndim_problem))
        s = 0
        
        return mean, x, y, p, p_hat, s

    def iterate(self, mean, x, y, p, p_hat, args=None):
        # Generate and evaluate new individuals
        for i in range(self.n_individuals):
            if self._check_success():
                return x, y  # Stop if termination criteria are met
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            if self.n_gens < self.m:
                x[i] = mean + self.sigma * z  # Sample new solution
            else:
                # Weighted sampling for correlated samples
                x[i] = mean + self.sigma * (
                    self._x_1 * z +
                    self._x_2 * self.rng_optimization.standard_normal() * p_hat[i] +
                    self._x_3 * self.rng_optimization.standard_normal() * p
                )
            y[i] = self._evaluate_fitness(x[i], args)  # Evaluate fitness
            if y[i] < self.y_best:  # Update best solution
                self.y_best = y[i]
                self.x_best = np.copy(x[i])
        return x, y

    def update_distribution(self, mean, x, y, p, p_hat, s, y_bak):
        # Update mean and paths
        order = np.argsort(y)[:self.n_parents]  # Rank individuals
        y.sort()  # Sort fitness values
        mean_bak = np.dot(self._w[:self.n_parents], x[order])  # Compute new mean
        p = self._p_1 * p + self._p_2 * (mean_bak - mean) / self.sigma  # Update evolution path
        
        # Update path history
        if self.n_gens % self.n_steps == 0:
            p_hat[:-1] = p_hat[1:]  # Shift history
            p_hat[-1] = p  # Add current path
        
        if self.n_gens > 0:
            # Compute success weights based on ranks
            merged_y = np.hstack((y_bak[:self.n_parents], y[:self.n_parents]))
            r = np.argsort(merged_y)
            success_weights = np.zeros(self.n_parents)
            for i in range(2 * self.n_parents):
                idx = r[i]
                if idx < self.n_parents:
                    success_weights[idx] += self._rr[i]
                else:
                    success_weights[idx - self.n_parents] -= self._rr[i]
            
            # Update step size based on success weights
            q = np.sum(success_weights) / (self.n_parents * (2 * self.n_parents + 1))
            s = (1.0 - self.c_s) * s + self.c_s * (q - self.q_star)
            self.sigma *= np.exp(s / self.d_s)
        
        return mean_bak, p, p_hat, s

    def _print_verbose_info(self):
        # Print information about the current generation
        if self.verbose and self.x_best is not None:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    def optimize(self, fitness_function=None, args=None):
        # Main optimization loop
        if fitness_function is not None:
            self.cost_function = fitness_function
        
        mean, x, y, p, p_hat, s = self.initialize()  # Initialize population and parameters
        self._print_verbose_info()
        
        while not self._check_success():  # Termination criteria
            y_bak = np.copy(y)  # Backup fitness values
            x, y = self.iterate(mean, x, y, p, p_hat, args)  # Generate and evaluate population
            if self._check_success():
                break
            mean, p, p_hat, s = self.update_distribution(mean, x, y, p, p_hat, s, y_bak)  # Update distribution
            self.n_gens += 1  # Increment generation count
            self._print_verbose_info()
        
        results = self._collect()  # Collect results
        results['n_gens'] = self.n_gens
        results['p'] = p
        results['s'] = s
        return results

    def print_report(self, results):
        # Print final results
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

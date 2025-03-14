import numpy as np
from ..core.optimizer import Optimizer

class CMAES(Optimizer):
    # Initialization of the CMA-ES optimizer
    def __init__(self, problem, options):
        # Call the parent class constructor
        Optimizer.__init__(self, problem, options)
        
        # Set the number of individuals (population size) if not provided
        if self.n_individuals is None:
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))  # Default formula for CMA-ES
        assert self.n_individuals >= 2  # Ensure a minimum population size
        
        # Set the number of parents used for recombination
        self.n_parents = options.get('n_parents', int(self.n_individuals / 2))
        assert self.n_parents <= self.n_individuals and self.n_parents > 0  # Ensure valid number of parents
        
        # Initialize mean vector and step-size (sigma)
        self.mean = options.get('mean', options.get('x'))
        self.sigma = options.get('sigma')
        assert self.sigma > 0  # Ensure a positive step-size
        
        # Compute the expected value of the Euclidean norm of a standard normal vector
        self._e_chi = np.sqrt(self.ndim_problem) * (
            1.0 - 1.0 / (4.0 * self.ndim_problem) + 1.0 / (21.0 * np.square(self.ndim_problem))
        )
        
        # Initialize other variables
        self._w, self._mu_eff = None, None
        self.c_s, self.d_sigma = None, None
        self._p_s_1, self._p_s_2 = None, None
        self._p_c_1, self._p_c_2 = None, None
        self.c_c, self.c_1, self.c_w = None, None, None
        self.n_gens = 0  # Track the number of generations
        
        # Initialize the best solution variables
        self.x_best = None
        self.y_best = np.inf  # Start with the worst possible fitness

    # Initialize the population and algorithm parameters
    def initialize(self):
        # Compute weights for recombination
        w_a = np.log((self.n_individuals + 1.0) / 2.0) - np.log(np.arange(self.n_individuals) + 1.0)
        self._mu_eff = np.square(np.sum(w_a[:self.n_parents])) / np.sum(np.square(w_a[:self.n_parents]))  # Effective sample size
        
        # Set learning rates and adaptation parameters
        self.c_s = (self._mu_eff + 2.0) / (self.ndim_problem + self._mu_eff + 5.0)
        self.d_sigma = (
            1.0
            + 2.0 * np.maximum(0.0, np.sqrt((self._mu_eff - 1.0) / (self.ndim_problem + 1.0)) - 1.0)
            + self.c_s
        )
        self.c_c = (4.0 + self._mu_eff / self.ndim_problem) / (
            self.ndim_problem + 4.0 + 2.0 * self._mu_eff / self.ndim_problem
        )
        self.c_1 = 2.0 / (np.square(self.ndim_problem + 1.3) + self._mu_eff)
        self.c_w = np.minimum(
            1.0 - self.c_1,
            (1.0 / 4.0 + self._mu_eff + 1.0 / self._mu_eff - 2.0)
            / (np.square(self.ndim_problem + 2.0) + self._mu_eff / 2.0),
        )
        
        # Normalize weights
        self._w = np.where(w_a >= 0, 1.0 / np.sum(w_a[w_a > 0]) * w_a, 0.1 / (-np.sum(w_a[w_a < 0])) * w_a)
        
        # Set constants for path evolution
        self._p_s_1, self._p_s_2 = 1.0 - self.c_s, np.sqrt(self.c_s * (2.0 - self.c_s) * self._mu_eff)
        self._p_c_1, self._p_c_2 = 1.0 - self.c_c, np.sqrt(self.c_c * (2.0 - self.c_c) * self._mu_eff)
        
        # Initialize population
        x = np.empty((self.n_individuals, self.ndim_problem))
        mean = (
            self.rng_initialization.uniform(self.lower_bound, self.upper_bound)
            if self.mean is None
            else np.copy(self.mean)
        )
        self.mean = np.copy(mean)
        
        # Evaluate initial population
        for i in range(self.n_individuals):
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            x[i] = mean + self.sigma * z
            y_i = self._evaluate_fitness(x[i])
            if y_i < self.y_best:
                self.y_best = y_i
                self.x_best = np.copy(x[i])
        
        # Initialize CMA-ES internal variables
        p_s = np.zeros((self.ndim_problem,))
        p_c = np.zeros((self.ndim_problem,))
        cm = np.eye(self.ndim_problem)
        e_ve = np.eye(self.ndim_problem)
        e_va = np.ones((self.ndim_problem,))
        y = np.empty((self.n_individuals,))
        d = np.empty((self.n_individuals, self.ndim_problem))
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    # Perform one iteration of the CMA-ES algorithm
    def iterate(self, x, mean, e_ve, e_va, y, d, args=None):
        for k in range(self.n_individuals):
            if self._check_success():
                return x, y, d  # Stop iteration if termination condition is met
            z = self.rng_optimization.standard_normal((self.ndim_problem,))
            d[k] = np.dot(e_ve @ np.diag(e_va), z)  # Generate new individuals
            x[k] = mean + self.sigma * d[k]
            y[k] = self._evaluate_fitness(x[k], args)
            if y[k] < self.y_best:  # Update the best solution
                self.y_best = y[k]
                self.x_best = np.copy(x[k])
        return x, y, d

    # Update the distribution parameters
    def update_distribution(self, x, p_s, p_c, cm, e_ve, e_va, y, d):
        order = np.argsort(y)  # Rank solutions by fitness
        wd = np.dot(self._w[:self.n_parents], d[order[:self.n_parents]])
        mean = np.dot(self._w[:self.n_parents], x[order[:self.n_parents]])  # Update mean
        
        cm_minus_half = e_ve @ np.diag(1.0 / e_va) @ e_ve.T  # Inverse square root of covariance matrix
        p_s = self._p_s_1 * p_s + self._p_s_2 * np.dot(cm_minus_half, wd)  # Update evolution path
        self.sigma *= np.exp(self.c_s / self.d_sigma * (np.linalg.norm(p_s) / self._e_chi - 1.0))  # Update step size
        
        h_s = float(
            np.linalg.norm(p_s) / np.sqrt(1.0 - np.power(1.0 - self.c_s, 2 * (self.n_gens + 1)))
            < (1.4 + 2.0 / (self.ndim_problem + 1.0)) * self._e_chi
        )
        p_c = self._p_c_1 * p_c + h_s * self._p_c_2 * wd  # Update conjugate evolution path
        
        w_o = self._w * np.where(
            self._w >= 0,
            1.0,
            self.ndim_problem / (np.square(np.linalg.norm(cm_minus_half @ d.T, axis=0)) + 1e-8),
        )
        cm = (
            (1.0 + self.c_1 * (1.0 - h_s) * self.c_c * (2.0 - self.c_c) - self.c_1 - self.c_w * np.sum(self._w)) * cm
            + self.c_1 * np.outer(p_c, p_c)
        )
        
        for i in range(self.n_individuals):
            cm += self.c_w * w_o[i] * np.outer(d[order[i]], d[order[i]])  # Update covariance matrix
        
        cm = (cm + cm.T) / 2.0  # Ensure symmetry
        e_va, e_ve = np.linalg.eigh(cm)  # Eigen decomposition of covariance matrix
        e_va = np.sqrt(np.where(e_va < 0.0, 1e-8, e_va))  # Ensure eigenvalues are non-negative
        
        return mean, p_s, p_c, cm, e_ve, e_va

    # Print verbose information about the current generation
    def _print_verbose_info(self):
        if self.verbose and self.x_best is not None:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    # Main optimization loop
    def optimize(self, fitness_function=None, args=None):
        if fitness_function is not None:
            self.cost_function = fitness_function  # Assign the fitness function
            
        # Initialize parameters and population
        x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize()
        self._print_verbose_info()
        
        while not self._check_success():  # Continue until termination condition is met
            x, y, d = self.iterate(x, mean, e_ve, e_va, y, d, args)  # Generate new population
            if self._check_success():
                break
            mean, p_s, p_c, cm, e_ve, e_va = self.update_distribution(x, p_s, p_c, cm, e_ve, e_va, y, d)  # Update distribution
            self.n_gens += 1
            self._print_verbose_info()
        
        results = self._collect()  # Collect optimization results
        results['n_gens'] = self.n_gens  # Include number of generations in the results
        return results

    # Print a summary of the optimization results
    def print_report(self, results):
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

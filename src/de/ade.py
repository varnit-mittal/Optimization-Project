import numpy as np
from scipy.stats import cauchy
from ..core.optimizer import Optimizer

class ADE(Optimizer):
    """
    Adaptive Differential Evolution (ADE) optimizer, an enhanced variant of Differential Evolution.
    This implementation includes parameter adaptation, mutation strategies, and an archive for diversity.
    """

    def __init__(self, problem, options):
        """
        Initialize the ADE optimizer.
        Args:
            problem (dict): Problem details including cost function, dimensionality, and bounds.
            options (dict): Configuration options for ADE, including algorithm-specific parameters.
        """
        Optimizer.__init__(self, problem, options)
        # Set default number of individuals if not provided
        if self.n_individuals is None:
            self.n_individuals = 100
        assert self.n_individuals > 0

        # ADE-specific parameters
        self.n_mu = options.get('n_mu', 0.5)  # Mean of normal distribution for crossover
        self.median = options.get('median', 0.5)  # Location of Cauchy distribution for mutation
        self.p = options.get('p', 0.05)  # Greediness parameter (fraction of best individuals used for mutation)
        self.c = options.get('c', 0.1)  # Adaptation rate for parameters
        self.n_gens = 0  # Number of generations

    def initialize(self, args=None):
        """
        Initialize the population and evaluate their fitness.
        Returns:
            x (ndarray): Initial population.
            y (ndarray): Fitness values for the population.
            a (ndarray): Archive of inferior solutions.
        """
        # Randomly initialize individuals within bounds
        x = self.rng_initialization.uniform(
            self.lower_bound, self.upper_bound, size=(self.n_individuals, self.ndim_problem)
        )
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_success():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        
        # Initialize empty archive
        a = np.empty((0, self.ndim_problem))
        return x, y, a

    def mutate(self, x, y, a):
        """
        Generate mutant vectors using the population and archive.
        Args:
            x (ndarray): Current population.
            y (ndarray): Fitness values of the population.
            a (ndarray): Archive of inferior solutions.
        Returns:
            x_mu (ndarray): Mutant population.
            f_mu (ndarray): Mutation factors for individuals.
        """
        x_mu = np.empty((self.n_individuals, self.ndim_problem))
        f_mu = np.empty(self.n_individuals)

        # Determine indices of the top p% best solutions
        p_best_size = max(1, int(self.p * self.n_individuals))
        best_indices = np.argsort(y)[:p_best_size]

        # Combine current population with archive
        x_union = np.vstack((x, a)) if len(a) > 0 else x

        for i in range(self.n_individuals):
            # Generate mutation factor from Cauchy distribution
            f_mu[i] = cauchy.rvs(loc=self.median, scale=0.1, random_state=self.rng_optimization)
            f_mu[i] = min(1.0, max(0.0, f_mu[i]))  # Clip mutation factor to [0, 1]
            
            # Select indices for mutation
            x_best_idx = self.rng_optimization.choice(best_indices)
            r1 = self.rng_optimization.choice([j for j in range(self.n_individuals) if j != i])
            r2 = self.rng_optimization.choice([j for j in range(len(x_union)) if j != i and j != r1])
            
            # Generate mutant
            x_mu[i] = x[i] + f_mu[i] * (x[x_best_idx] - x[i]) + f_mu[i] * (x[r1] - x_union[r2])
        
        return x_mu, f_mu

    def crossover(self, x_mu, x):
        """
        Perform crossover between mutant and original population.
        Args:
            x_mu (ndarray): Mutant population.
            x (ndarray): Original population.
        Returns:
            x_cr (ndarray): Crossover population.
            cr_values (ndarray): Crossover rates.
        """
        x_cr = np.copy(x)
        cr_values = np.minimum(1.0, np.maximum(0.0, 
            self.rng_optimization.normal(self.n_mu, 0.1, size=self.n_individuals)))
        
        for i in range(self.n_individuals):
            # Ensure at least one dimension is crossed over
            j_rand = self.rng_optimization.integers(self.ndim_problem)
            mask = (self.rng_optimization.random(self.ndim_problem) < cr_values[i])
            mask[j_rand] = True
            x_cr[i, mask] = x_mu[i, mask]
        
        return x_cr, cr_values

    def select(self, x_cr, x, y, a, f_mu, cr_values, args):
        """
        Perform selection to decide which individuals survive to the next generation.
        Args:
            x_cr (ndarray): Crossover population.
            x (ndarray): Original population.
            y (ndarray): Fitness values of the population.
            a (ndarray): Archive of inferior solutions.
            f_mu (ndarray): Mutation factors.
            cr_values (ndarray): Crossover rates.
        Returns:
            Updated population, fitness values, and archive.
        """
        successful_f = []
        successful_cr = []
        
        for i in range(self.n_individuals):
            if self._check_success():
                break
            
            y_trial = self._evaluate_fitness(x_cr[i], args)
            
            if y_trial < y[i]:
                # Record successful parameters
                successful_f.append(f_mu[i])
                successful_cr.append(cr_values[i])
                
                # Update population and archive
                a = np.vstack((a, x[i])) if len(a) > 0 else x[i].reshape(1, -1)
                x[i] = x_cr[i]
                y[i] = y_trial
        
        # Adapt mutation and crossover parameters
        if len(successful_cr) > 0:
            self.n_mu = (1 - self.c) * self.n_mu + self.c * np.mean(successful_cr)
            if len(successful_f) > 0:
                self.median = (1 - self.c) * self.median + \
                    self.c * (np.sum(np.array(successful_f)**2) / np.sum(successful_f))
        
        # Limit archive size
        if len(a) > self.n_individuals:
            indices = self.rng_optimization.choice(len(a), self.n_individuals, replace=False)
            a = a[indices]
            
        return x, y, a

    def iterate(self, x, y, a, args):
        """
        Execute one iteration (generation) of the ADE algorithm.
        Returns:
            Updated population, fitness values, and archive.
        """
        x_mu, f_mu = self.mutate(x, y, a)
        x_cr, cr_values = self.crossover(x_mu, x)
        x, y, a = self.select(x_cr, x, y, a, f_mu, cr_values, args)
        self.n_gens += 1
        return x, y, a

    def optimize(self, fitness_function=None, args=None):
        """
        Run the ADE optimization process.
        Args:
            fitness_function (callable): Optional fitness function to override the problem's default.
        Returns:
            Optimization results.
        """
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
        """
        Print detailed information about the current generation if verbosity is enabled.
        """
        if self.verbose:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    def _collect(self):
        """
        Collect and return the final results of the optimization process.
        """
        self._print_verbose_info()
        results = Optimizer._collect(self)
        results['n_gens'] = self.n_gens
        return results
    
    def print_report(self, results):
        """
        Print a summary report of the optimization results.
        """
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

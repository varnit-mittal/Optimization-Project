import numpy as np
from ..core.optimizer import Optimizer

class CDE(Optimizer):
    def __init__(self, problem, options):
        # Initialize the base Optimizer class
        Optimizer.__init__(self, problem, options)
        
        # Set default number of individuals if not provided
        if self.n_individuals is None:
            self.n_individuals = 100
        assert self.n_individuals > 0  # Ensure population size is valid

        # Mutation scaling factor (controls the differential weight)
        self.mu = options.get('mu', 0.5)
        
        # Crossover probability (controls how much crossover occurs)
        self.cr = options.get('cr', 0.9)
        
        # Counter for number of generations
        self.n_gens = 0

    def initialize(self, args=None):
        # Initialize population within the problem bounds
        x = self.rng_initialization.uniform(self.lower_bound, self.upper_bound,
            size=(self.n_individuals, self.ndim_problem))

        # Array to store fitness values for the population
        y = np.empty((self.n_individuals,))

        # Evaluate the fitness of each individual in the initial population
        for i in range(self.n_individuals):
            if self._check_success():
                break  # Stop if termination criteria are met
            y[i] = self._evaluate_fitness(x[i], args)

        # Initialize an empty array for mutant vectors
        v = np.empty((self.n_individuals, self.ndim_problem))
        
        return x, y, v

    def mutate(self, x=None, v=None):
        # Generate mutant vectors based on the current population
        for i in range(self.n_individuals):
            # Select three random, distinct individuals from the population
            r = self.rng_optimization.permutation(self.n_individuals)[:4]
            r = r[r != i][:3]  # Ensure the selected individuals are not the same as the current individual
            
            # Compute mutant vector: x[r[0]] + mu * (x[r[1]] - x[r[2]])
            v[i] = x[r[0]] + self.mu * (x[r[1]] - x[r[2]])
        
        return v

    def crossover(self, v=None, x=None):
        # Perform crossover to generate trial vectors
        for i in range(self.n_individuals):
            # Randomly select an index for guaranteed inclusion in the trial vector
            j_r = self.rng_optimization.integers(self.ndim_problem)
            
            # Backup the value at the guaranteed index
            tmp = v[i, j_r]
            
            # Determine which elements to copy from the parent (x[i]) based on crossover probability
            co = self.rng_optimization.random(self.ndim_problem) > self.cr
            v[i, co] = x[i, co]  # Replace elements in v[i] with elements from x[i]
            
            # Ensure the value at the guaranteed index is restored
            v[i, j_r] = tmp
        
        return v

    def select(self, v=None, x=None, y=None, args=None):
        # Select the next generation by comparing trial vectors with the current population
        for i in range(self.n_individuals):
            if self._check_success():
                break  # Stop if termination criteria are met
            
            # Evaluate the fitness of the trial vector
            yy = self._evaluate_fitness(v[i], args)
            
            # Replace the current individual with the trial vector if it has better fitness
            if yy < y[i]:
                x[i], y[i] = v[i], yy
        
        return x, y

    def iterate(self, x=None, y=None, v=None, args=None):
        # Perform one generation of the optimization algorithm
        v = self.mutate(x, v)  # Generate mutant vectors
        v = self.crossover(v, x)  # Perform crossover to generate trial vectors
        x, y = self.select(v, x, y, args)  # Select the next generation
        self.n_gens += 1  # Increment generation counter
        return x, y

    def _print_verbose_info(self):
        # Print detailed information about the current generation if verbose mode is enabled
        if self.verbose:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    def _collect(self):
        # Collect results at the end of the optimization process
        self._print_verbose_info()
        results = Optimizer._collect(self)
        results['n_gens'] = self.n_gens  # Include the total number of generations in the results
        return results

    def optimize(self, fitness_function=None, args=None):
        # Main optimization loop
        if fitness_function is not None:
            self.fitness_function = fitness_function  # Set the fitness function

        # Initialize the population, fitness values, and mutant vectors
        x, y, v = self.initialize(args)

        # Iterate until the termination criteria are met
        while not self._check_success():
            self._print_verbose_info()
            x, y = self.iterate(x, y, v, args)

        # Collect and return the final results
        results = self._collect()
        results['n_gens'] = self.n_gens + 1  # Include the total number of generations
        return results

    def print_report(self, results):
        # Print the final optimization results
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

import numpy as np

class Optimizer(object):
    """
    Base class for optimization algorithms. Handles the setup of problem definitions,
    options, random number generation, and evaluation of fitness.
    """
    def __init__(self, problem, options):
        # Problem configuration
        self.cost_function = problem.get('cost_function')  # The objective function to optimize
        self.ndim_problem = problem.get('ndim_problem')   # Dimensionality of the problem
        assert self.ndim_problem > 0  # Ensure the dimensionality is valid
        self.upper_bound = problem.get('upper_bound')     # Upper bounds of the search space
        self.lower_bound = problem.get('lower_bound')     # Lower bounds of the search space

        # Optimization options
        self.options = options
        self.max_evals = options.get('max_evals', np.inf)  # Maximum allowed function evaluations
        self.n_individuals = options.get('n_individuals')  # Population size (if applicable)
        self.seed_rng = options.get('seed_rng')            # Seed for random number generation

        # Random number generator for general usage
        if self.seed_rng is None:
            self.rng = np.random.default_rng()  # Use system-generated seed if not provided
        else:
            self.rng = np.random.default_rng(self.seed_rng)

        # Random number generator for initialization
        self.seed_initialization = options.get(
            'seed_initialization',
            self.rng.integers(np.iinfo(np.int64).max)  # Generate a seed if not provided
        )
        self.rng_initialization = np.random.default_rng(self.seed_initialization)

        # Random number generator for optimization process
        self.seed_optimization = options.get(
            'seed_optimization',
            self.rng.integers(np.iinfo(np.int64).max)  # Generate a seed if not provided
        )
        self.rng_optimization = np.random.default_rng(self.seed_optimization)

        # Verbosity level for logging
        self.verbose = options.get('verbose', None)

        # Status of the optimization process
        self.is_success = "no_term"  # Default status (no termination yet)

        # Evaluation counters and best solution tracker
        self.n_evals = options.get('n_evals', 0)  # Number of evaluations already performed
        self.y_best, self.x_best = options.get('y_best', np.inf), None  # Best solution found

    def _evaluate_fitness(self, x, args=None):
        """
        Evaluates the fitness (objective function value) for a given solution `x`.
        Updates the best solution if the current fitness is better.
        """
        if args is None:
            y = self.cost_function(x)  # Evaluate cost function without additional arguments
        else:
            y = self.cost_function(x, args=args)  # Evaluate cost function with arguments

        self.n_evals += 1  # Increment evaluation count

        # Update the best solution if the current one is better
        if y < self.y_best:
            self.x_best, self.y_best = np.copy(x), y

        return float(y)

    def _check_success(self):
        """
        Checks whether the termination condition is met based on the number of evaluations.
        Returns True if the process should terminate.
        """
        if self.n_evals >= self.max_evals:
            self.is_success = 0  # Termination condition met
            return True
        else:
            self.is_success = 1  # Optimization can continue
            return False

    def _collect(self):
        """
        Collects the results of the optimization process.
        Returns a dictionary containing the best solution, its fitness, and success status.
        """
        return {'x_best': self.x_best,
                'y_best': self.y_best,
                'is_success': self.is_success}

    def optimize(self, cost_function=None):
        """
        Optimization process. This is a placeholder to be implemented by specific algorithms.
        Optionally, a new cost function can be provided.
        """
        if cost_function is not None:
            self.cost_function = cost_function  # Update the cost function if provided

        fitness = []  # Placeholder for fitness values (to be populated in derived classes)
        return fitness

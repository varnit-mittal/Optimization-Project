import numpy as np
from ..core.optimizer import Optimizer

class IPSO(Optimizer):
    def __init__(self, problem, options):
        # Initialize the base optimizer class
        Optimizer.__init__(self, problem, options)
        
        # Core IPSO parameters
        self.n_individuals = options.get('n_individuals', 1)  # Start with minimum swarm size
        self.max_n_individuals = options.get('max_n_individuals', 1000)  # Max swarm size
        assert self.max_n_individuals > 0
        
        # Enhanced cognitive and social parameters
        self.c1_start = options.get('c1_start', 2.5)  # Starting cognitive coefficient
        self.c1_end = options.get('c1_end', 1.5)  # Ending cognitive coefficient
        self.c2_start = options.get('c2_start', 1.5)  # Starting social coefficient
        self.c2_end = options.get('c2_end', 2.5)  # Ending social coefficient
        
        # Constriction and velocity parameters
        self.chi = options.get('constriction', 0.729)  # Clerc's constriction coefficient
        self.domain_range = self.upper_bound - self.lower_bound  # Problem domain range
        self.max_v = options.get('max_v', 0.15 * self.domain_range)  # Max velocity
        self.min_v = -self.max_v  # Min velocity
        
        # Population growth parameters
        self.growth_rate = options.get('growth_rate', 0.8)  # Success rate threshold for population growth
        self.learning_period = options.get('learning_period', 5)  # Period for growth evaluation
        
        # Adaptation parameters
        self.success_memory = []  # Memory to track success rates
        self.memory_size = options.get('memory_size', 10)  # Maximum memory size
        self.stagnation_threshold = options.get('stagnation_threshold', 5)  # Threshold for stagnation
        self.stagnation_counter = 0  # Counter to track stagnation
        self.prev_best = np.inf  # Track previous best fitness
        
        self.n_gens = 0  # Generation counter
        self.max_gens = options.get('max_gens', 1000)  # Maximum generations

    def initialize(self, args=None):
        # Initialize the swarm using Latin Hypercube Sampling for uniform distribution
        x = self._latin_hypercube_sampling(self.n_individuals, self.ndim_problem)
        x = x * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        # Initialize velocities and fitness values
        v = np.zeros((self.n_individuals, self.ndim_problem))  # Particle velocities
        y = np.empty(self.n_individuals)  # Fitness values
        
        # Evaluate initial fitness of all particles
        for i in range(self.n_individuals):
            if self._check_success():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        
        # Personal best positions and fitness
        p_x, p_y = np.copy(x), np.copy(y)
        return v, x, y, p_x, p_y

    def _latin_hypercube_sampling(self, n_samples, n_dims):
        # Generate samples using Latin Hypercube Sampling
        x = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            x[:, i] = (self.rng_initialization.permutation(n_samples) + 
                      self.rng_initialization.random(n_samples)) / n_samples
        return x

    def _update_parameters(self):
        # Linearly interpolate cognitive and social coefficients based on progress
        progress = min(self.n_gens / self.max_gens, 1.0)
        c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
        c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
        return c1, c2

    def _should_grow_population(self):
        # Check if the success rate warrants population growth
        if len(self.success_memory) >= self.learning_period:
            success_rate = sum(self.success_memory[-self.learning_period:]) / self.learning_period
            return success_rate >= self.growth_rate
        return False

    def _update_success_memory(self, improved):
        # Update the success memory with the latest result
        self.success_memory.append(float(improved))
        if len(self.success_memory) > self.memory_size:
            self.success_memory.pop(0)

    def _generate_new_particle(self, p_x, p_y):
        # Generate a new particle based on the best solution found so far
        best_model = p_x[np.argmin(p_y)]  # Global best particle
        
        # Random initialization within the domain
        xx = self.rng_optimization.uniform(self.lower_bound, self.upper_bound, size=self.ndim_problem)
        
        # Adaptive exploration based on the distance to the best model
        distance_to_best = np.linalg.norm(xx - best_model)
        adaptive_rate = np.clip(1.0 - distance_to_best / np.linalg.norm(self.domain_range), 0.2, 0.8)
        
        # Dimension-wise perturbation with random factors
        random_factors = self.rng_optimization.uniform(size=self.ndim_problem)
        xx += adaptive_rate * random_factors * (best_model - xx)
        
        # Ensure bounds are respected
        xx = np.clip(xx, self.lower_bound, self.upper_bound)
        return xx

    def iterate(self, x=None, y=None, v=None, p_x=None, p_y=None, args=None):
        c1, c2 = self._update_parameters()  # Update cognitive and social coefficients
        improved = False  # Track if any improvement occurred
        
        # Update particles using horizontal social learning
        for i in range(self.n_individuals):
            if self._check_success():
                break
            
            # Generate random coefficients for velocity update
            r1 = self.rng_optimization.random(self.ndim_problem)
            r2 = self.rng_optimization.random(self.ndim_problem)
            
            # Identify best neighbor in ring topology
            left = (i - 1) % self.n_individuals
            right = (i + 1) % self.n_individuals
            neighborhood = [left, i, right]
            best_neighbor = neighborhood[np.argmin([p_y[j] for j in neighborhood])]
            
            # Update velocity with constriction coefficient
            v[i] = self.chi * (v[i] +
                             c1 * r1 * (p_x[i] - x[i]) +
                             c2 * r2 * (p_x[best_neighbor] - x[i]))
            
            # Apply velocity bounds
            v[i] = np.clip(v[i], self.min_v, self.max_v)
            
            # Update position and enforce bounds
            x[i] += v[i]
            x[i] = np.clip(x[i], self.lower_bound, self.upper_bound)
            
            # Evaluate new fitness
            y_new = self._evaluate_fitness(x[i], args)
            if y_new < y[i]:
                improved = True
                y[i] = y_new
                if y[i] < p_y[i]:
                    p_x[i], p_y[i] = np.copy(x[i]), y[i]
        
        # Update success memory
        self._update_success_memory(improved)
        
        # Check if population growth is needed
        if (self.n_individuals < self.max_n_individuals and 
            self._should_grow_population() and 
            not self._check_success()):
            
            # Generate and add a new particle
            xx = self._generate_new_particle(p_x, p_y)
            yy = self._evaluate_fitness(xx, args)
            v = np.vstack((v, np.zeros(self.ndim_problem)))
            x = np.vstack((x, xx))
            y = np.append(y, yy)
            p_x = np.vstack((p_x, xx))
            p_y = np.append(p_y, yy)
            self.n_individuals += 1
        
        self.n_gens += 1  # Increment generation counter
        return x, y, v, p_x, p_y

    def _print_verbose_info(self):
        # Print detailed progress information if verbose mode is enabled
        if self.verbose:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))

    def _collect(self):
        # Collect and print the results after optimization
        self._print_verbose_info()
        results = Optimizer._collect(self)
        results['n_gens'] = self.n_gens
        return results

    def optimize(self, cost_function=None, args=None):
        # Main optimization loop
        if cost_function is not None:
            self.cost_function = cost_function
        v, x, y, p_x, p_y = self.initialize(args)
        while not self._check_success():
            self._print_verbose_info()
            x, y, v, p_x, p_y = self.iterate(x, y, v, p_x, p_y, args)
        results = self._collect()
        results['n_gens'] = self.n_gens + 1
        return results

    def print_report(self, results):
        # Print final optimization results
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

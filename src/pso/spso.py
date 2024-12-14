import numpy as np
from ..core.optimizer import Optimizer

class SPSO(Optimizer):
    """
    Standard Particle Swarm Optimization (SPSO) implementation with enhancements such as:
    - Adaptive inertia and acceleration coefficients
    - Stagnation detection and diversity handling
    - Velocity clamping and boundary handling
    """
    def __init__(self, problem, options):
        """
        Initialize the SPSO optimizer.
        Args:
            problem (dict): Problem details (cost function, dimensionality, bounds).
            options (dict): Configuration options for SPSO.
        """
        Optimizer.__init__(self, problem, options)
        
        # Set default number of individuals if not provided
        if self.n_individuals is None:
            self.n_individuals = 100
        assert self.n_individuals > 0
        
        # PSO parameters with proven defaults
        self.w_start = options.get('w_start', 0.9)  # Initial inertia weight
        self.w_end = options.get('w_end', 0.4)      # Final inertia weight
        self.c1_start = options.get('c1_start', 2.5)  # Initial cognitive weight
        self.c1_end = options.get('c1_end', 0.5)      # Final cognitive weight
        self.c2_start = options.get('c2_start', 0.5)  # Initial social weight
        self.c2_end = options.get('c2_end', 2.5)      # Final social weight
        
        # Domain range for velocity clamping
        self.domain_range = self.upper_bound - self.lower_bound
        self._max_v = options.get('max_v', 0.15 * self.domain_range)  # Maximum velocity
        self._min_v = -self._max_v  # Minimum velocity
        
        # Stagnation detection parameters
        self.stagnation_count = 0
        self.stagnation_threshold = options.get('stagnation_threshold', 10)
        self.diversity_threshold = options.get('diversity_threshold', 1e-6)
        self.prev_best = np.inf  # Previous best solution
        
        # Iteration and termination
        self.n_gens = 0
        self.max_gens = options.get('max_gens', 1000)

    def initialize(self, args=None):
        """
        Initialize the population (positions and velocities) and evaluate fitness.
        Returns:
            x: Particle positions
            y: Fitness values of particles
            v: Particle velocities
            p_x: Personal best positions
            p_y: Personal best fitness values
            n_x: Neighborhood best positions
        """
        # Use Latin Hypercube Sampling for better initial distribution
        x = self._latin_hypercube_sampling(self.n_individuals, self.ndim_problem)
        x = x * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        # Evaluate fitness for initial population
        y = np.empty((self.n_individuals,))
        for i in range(self.n_individuals):
            if self._check_success():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        
        # Initialize velocities uniformly within the allowed range
        v = self.rng_initialization.uniform(self._min_v, self._max_v,
                                            size=(self.n_individuals, self.ndim_problem))
        
        # Initialize personal bests
        p_x = np.copy(x)
        p_y = np.copy(y)
        n_x = np.zeros_like(x)  # Placeholder for neighborhood best positions
        
        return x, y, v, p_x, p_y, n_x

    def _latin_hypercube_sampling(self, n_samples, n_dims):
        """
        Generate a Latin Hypercube Sampling (LHS) for better distribution of particles.
        Returns:
            x: Sampled points
        """
        x = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            x[:, i] = (self.rng_initialization.permutation(n_samples) + 
                       self.rng_initialization.random(n_samples)) / n_samples
        return x

    def _check_diversity(self, x):
        """
        Calculate population diversity to detect stagnation.
        Returns:
            bool: True if diversity is above threshold, False otherwise.
        """
        mean_pos = np.mean(x, axis=0)
        distances = np.sqrt(np.sum((x - mean_pos)**2, axis=1))
        diversity = np.mean(distances)
        return diversity > self.diversity_threshold

    def _update_parameters(self):
        """
        Update inertia weight and acceleration coefficients based on progress.
        Returns:
            w: Updated inertia weight
            c1: Updated cognitive weight
            c2: Updated social weight
        """
        progress = min(self.n_gens / self.max_gens, 1.0)
        w = self.w_start - (self.w_start - self.w_end) * progress
        c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
        c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
        return w, c1, c2

    def _reset_particle(self, i, x, v, y, p_x, p_y):
        """
        Reset particle's position, velocity, and fitness when stagnation is detected.
        Args:
            i: Index of the particle to reset.
        Returns:
            Updated x, v, y, p_x, p_y for the particle.
        """
        x[i] = self.rng_optimization.uniform(self.lower_bound, self.upper_bound, 
                                             size=self.ndim_problem)
        v[i] = self.rng_optimization.uniform(self._min_v, self._max_v, 
                                             size=self.ndim_problem)
        y[i] = self._evaluate_fitness(x[i])
        p_x[i] = np.copy(x[i])
        p_y[i] = y[i]
        return x[i], v[i], y[i], p_x[i], p_y[i]

    def iterate(self, x=None, y=None, v=None, p_x=None, p_y=None, n_x=None, args=None):
        """
        Perform a single iteration (generation) of the SPSO algorithm.
        Updates positions, velocities, and fitness values.
        """
        # Update adaptive parameters
        w, c1, c2 = self._update_parameters()
        
        # Check for stagnation
        if abs(self.y_best - self.prev_best) < 1e-8:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self.prev_best = self.y_best
        
        # Check population diversity
        low_diversity = not self._check_diversity(x)
        
        for i in range(self.n_individuals):
            if self._check_success():
                break
            
            # Reset particle if stagnated or diversity is low
            if self.stagnation_count >= self.stagnation_threshold or low_diversity:
                x[i], v[i], y[i], p_x[i], p_y[i] = self._reset_particle(i, x, v, y, p_x, p_y)
                continue
            
            # Determine neighborhood best (ring topology)
            left = (i - 1) % self.n_individuals
            right = (i + 1) % self.n_individuals
            neighborhood = [left, i, right]
            best_neighbor = neighborhood[np.argmin([p_y[j] for j in neighborhood])]
            n_x[i] = p_x[best_neighbor]
            
            # Random components for cognitive and social influence
            r1 = self.rng_optimization.random(self.ndim_problem)
            r2 = self.rng_optimization.random(self.ndim_problem)
            
            # Update velocity using constriction coefficient
            chi = 0.7298  # Clerc's constriction coefficient
            v[i] = chi * (w * v[i] +
                          c1 * r1 * (p_x[i] - x[i]) +
                          c2 * r2 * (n_x[i] - x[i]))
            
            # Clamp velocity
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            
            # Update position and handle boundary conditions
            x[i] += v[i]
            out_of_bounds = np.logical_or(x[i] < self.lower_bound, x[i] > self.upper_bound)
            if np.any(out_of_bounds):
                # Bounce back strategy for particles crossing boundaries
                x[i] = np.clip(x[i], self.lower_bound, self.upper_bound)
                v[i][out_of_bounds] = -0.5 * v[i][out_of_bounds]  # Reverse velocity with damping
            
            # Evaluate fitness and update personal best
            y[i] = self._evaluate_fitness(x[i], args)
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = np.copy(x[i]), y[i]
        
        self.n_gens += 1
        return x, y, v, p_x, p_y, n_x

    def _print_verbose_info(self):
        """
        Print progress if verbosity is enabled.
        """
        if self.verbose:
            if (self.n_gens % self.verbose == 0) or self._check_success():
                info = '\nGENERATION {:d} ({:d} evaluations)\n\tx_best = {}\n\ty_best = {:7.5e}\n'
                formatted_x_best = np.array2string(self.x_best, precision=5, separator=', ', suppress_small=True)
                print(info.format(self.n_gens, self.n_evals, formatted_x_best, self.y_best))
    
    def _collect(self):
        """
        Collect results of the optimization process.
        """
        self._print_verbose_info()
        results = Optimizer._collect(self)
        results['n_gens'] = self.n_gens
        return results
    
    def optimize(self, cost_function=None, args=None):
        """
        Run the SPSO optimization process.
        Returns:
            Results of the optimization.
        """
        if cost_function is not None:
            self.cost_function = cost_function
        x, y, v, p_x, p_y, n_x = self.initialize(args)
        while not self._check_success():
            self._print_verbose_info()
            x, y, v, p_x, p_y, n_x = self.iterate(x, y, v, p_x, p_y, n_x, args)
        results = self._collect()
        results['n_gens'] = self.n_gens + 1
        return results
    
    def print_report(self, results):
        """
        Print a summary of the optimization results.
        """
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")

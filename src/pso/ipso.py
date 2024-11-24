import numpy as np
from ..core.optimizer import Optimizer

class IPSO(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        # Core IPSO parameters
        self.n_individuals = options.get('n_individuals', 1)  # start with minimum swarm size
        self.max_n_individuals = options.get('max_n_individuals', 1000)
        assert self.max_n_individuals > 0
        
        # Enhanced cognitive and social parameters
        self.c1_start = options.get('c1_start', 2.5)
        self.c1_end = options.get('c1_end', 1.5)
        self.c2_start = options.get('c2_start', 1.5)
        self.c2_end = options.get('c2_end', 2.5)
        
        # Constriction and velocity parameters
        self.chi = options.get('constriction', 0.729)  # Clerc's constriction coefficient
        self.domain_range = self.upper_bound - self.lower_bound
        self.max_v = options.get('max_v', 0.15 * self.domain_range)
        self.min_v = -self.max_v
        
        # Population growth parameters
        self.growth_rate = options.get('growth_rate', 0.8)  # Success rate needed for population growth
        self.learning_period = options.get('learning_period', 5)  # Generations between growth attempts
        
        # Adaptation parameters
        self.success_memory = []
        self.memory_size = options.get('memory_size', 10)
        self.stagnation_threshold = options.get('stagnation_threshold', 5)
        self.stagnation_counter = 0
        self.prev_best = np.inf
        
        self.n_gens = 0
        self.max_gens = options.get('max_gens', 1000)
    
    def initialize(self, args=None):
        # Initialize with Latin Hypercube Sampling for better coverage
        x = self._latin_hypercube_sampling(self.n_individuals, self.ndim_problem)
        x = x * (self.upper_bound - self.lower_bound) + self.lower_bound
        
        v = np.zeros((self.n_individuals, self.ndim_problem))
        y = np.empty(self.n_individuals)
        
        for i in range(self.n_individuals):
            if self._check_success():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        
        p_x, p_y = np.copy(x), np.copy(y)
        return v, x, y, p_x, p_y
    
    def _latin_hypercube_sampling(self, n_samples, n_dims):
        x = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            x[:, i] = (self.rng_initialization.permutation(n_samples) + 
                      self.rng_initialization.random(n_samples)) / n_samples
        return x
    
    def _update_parameters(self):
        # Update cognitive and social coefficients based on progress
        progress = min(self.n_gens / self.max_gens, 1.0)
        c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
        c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
        return c1, c2
    
    def _should_grow_population(self):
        if len(self.success_memory) >= self.learning_period:
            success_rate = sum(self.success_memory[-self.learning_period:]) / self.learning_period
            return success_rate >= self.growth_rate
        return False
    
    def _update_success_memory(self, improved):
        self.success_memory.append(float(improved))
        if len(self.success_memory) > self.memory_size:
            self.success_memory.pop(0)
    
    def _generate_new_particle(self, p_x, p_y):
        # Enhanced vertical social learning
        best_model = p_x[np.argmin(p_y)]
        
        # Generate new position with adaptive exploration
        xx = self.rng_optimization.uniform(self.lower_bound, self.upper_bound, 
                                         size=self.ndim_problem)
        
        # Adaptive learning rate based on distance to best
        distance_to_best = np.linalg.norm(xx - best_model)
        adaptive_rate = np.clip(1.0 - distance_to_best / np.linalg.norm(self.domain_range), 0.2, 0.8)
        
        # Enhanced social learning with dimension-wise random factors
        random_factors = self.rng_optimization.uniform(size=self.ndim_problem)
        xx += adaptive_rate * random_factors * (best_model - xx)
        
        # Ensure bounds
        xx = np.clip(xx, self.lower_bound, self.upper_bound)
        return xx
    
    def iterate(self, x=None, y=None, v=None, p_x=None, p_y=None, args=None):
        c1, c2 = self._update_parameters()
        improved = False
        
        # Horizontal social learning (particle updates)
        for i in range(self.n_individuals):
            if self._check_success():
                break
            
            # Generate random coefficients
            r1 = self.rng_optimization.random(self.ndim_problem)
            r2 = self.rng_optimization.random(self.ndim_problem)
            
            # Find best neighbor using ring topology
            left = (i - 1) % self.n_individuals
            right = (i + 1) % self.n_individuals
            neighborhood = [left, i, right]
            best_neighbor = neighborhood[np.argmin([p_y[j] for j in neighborhood])]
            
            # Update velocity with constriction
            v[i] = self.chi * (v[i] +
                             c1 * r1 * (p_x[i] - x[i]) +
                             c2 * r2 * (p_x[best_neighbor] - x[i]))
            
            # Apply velocity bounds
            v[i] = np.clip(v[i], self.min_v, self.max_v)
            
            # Update position with boundary handling
            x[i] += v[i]
            x[i] = np.clip(x[i], self.lower_bound, self.upper_bound)
            
            # Evaluate fitness
            y_new = self._evaluate_fitness(x[i], args)
            if y_new < y[i]:
                improved = True
                y[i] = y_new
                if y[i] < p_y[i]:
                    p_x[i], p_y[i] = np.copy(x[i]), y[i]
        
        # Update success memory
        self._update_success_memory(improved)
        
        # Vertical social learning (population growth)
        if (self.n_individuals < self.max_n_individuals and 
            self._should_grow_population() and 
            not self._check_success()):
            
            # Generate and evaluate new particle
            xx = self._generate_new_particle(p_x, p_y)
            yy = self._evaluate_fitness(xx, args)
            
            # Add new particle to swarm
            v = np.vstack((v, np.zeros(self.ndim_problem)))
            x = np.vstack((x, xx))
            y = np.append(y, yy)
            p_x = np.vstack((p_x, xx))
            p_y = np.append(p_y, yy)
            self.n_individuals += 1
        
        self.n_gens += 1
        return x, y, v, p_x, p_y
    
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
    
    def optimize(self, cost_function=None, args=None):
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
        print(f"Best x : {np.array2string(results['x_best'], precision=4, separator=', ', suppress_small=True)}")
        print(f"Best y : {results['y_best']}")
import argparse
from src.pso.ipso import *
from src.core.benchmarks import *
from src.core.optimizer import *
from time import perf_counter_ns

def main():
    #problem
    parser=argparse.ArgumentParser(description="Black box optimization - IPSO") 
    parser.add_argument('--cost_function',type=str, required=True, choices=list(FUNCTIONS.keys()), help="The cost function to optimize")
    parser.add_argument('--ndim_problem',type=int,default=2,help="The number of dimensions of the problem")
    parser.add_argument('--upper_bound',type=float,required=True,help="The upper bound of the optimization problem")
    parser.add_argument('--lower_bound',type=float,required=True,help="The lower bound of the optimization problem")
    #options optimizer
    parser.add_argument('--max_evals',type=int,default=None, help="Maximum number of allowed evaluations (default:infinity)")
    parser.add_argument('--seed_rng',type=int,default=42,help="Random seed for the RNG")
    parser.add_argument('--seed_initialization',type=int,default=42,help="Random seed for initialization")
    parser.add_argument('--seed_optimization',type=int,default=42,help="Random seed for optimization")
    parser.add_argument('--verbose',type=int,default=0, help="Enable verbose output (default:False)")
    parser.add_argument('--n_evals',type=int,default=None,help="Number of currently used evaluations (default:0)")
    parser.add_argument('--y_best', type=float, default=np.inf, help="Best fitness value known (default: np.inf)")
    #options algo
    parser.add_argument('--n_individuals', type=int, default=1, help="Initial number of individuals in the swarm (default: 1)")
    parser.add_argument('--max_n_individuals', type=int, default=1000, help="Maximum number of individuals in the swarm (default: 1000)")

    # Enhanced cognitive and social parameters
    parser.add_argument('--c1_start', type=float, default=2.5, help="Cognitive parameter start value (default: 2.5)")
    parser.add_argument('--c1_end', type=float, default=1.5, help="Cognitive parameter end value (default: 1.5)")
    parser.add_argument('--c2_start', type=float, default=1.5, help="Social parameter start value (default: 1.5)")
    parser.add_argument('--c2_end', type=float, default=2.5, help="Social parameter end value (default: 2.5)")

    # Constriction and velocity parameters
    parser.add_argument('--constriction', type=float, default=0.729, help="Clerc's constriction coefficient (default: 0.729)")
    parser.add_argument('--max_v', type=float, default=None, help="Maximum velocity (default: 0.15 * domain_range)")

    # Population growth parameters
    parser.add_argument('--growth_rate', type=float, default=0.8, help="Success rate needed for population growth (default: 0.8)")
    parser.add_argument('--learning_period', type=int, default=5, help="Generations between growth attempts (default: 5)")

    # Adaptation parameters
    parser.add_argument('--memory_size', type=int, default=10, help="Size of success memory (default: 10)")
    parser.add_argument('--stagnation_threshold', type=int, default=5, help="Stagnation threshold (default: 5)")

    # Generational parameters
    parser.add_argument('--max_gens', type=int, default=1000, help="Maximum number of generations (default: 1000)")


    
    args=parser.parse_args()
    problem={
        'cost_function':FUNCTIONS[args.cost_function],
        'ndim_problem':args.ndim_problem,
        'upper_bound':args.upper_bound,
        'lower_bound':args.lower_bound,
    }
    options=dict()
    if args.max_evals is not None:
        options['max_evals'] = args.max_evals

    if args.seed_rng is not None:
        options['seed_rng'] = args.seed_rng

    if args.seed_initialization is not None:
        options['seed_initialization'] = args.seed_initialization

    if args.seed_optimization is not None:
        options['seed_optimization'] = args.seed_optimization

    if args.verbose:
        options['verbose'] = args.verbose

    if args.n_evals is not None:
        options['n_evals'] = args.n_evals

    if args.n_individuals is not None:
        options['n_individuals'] = args.n_individuals
    
    if args.max_n_individuals is not None:
        options['max_n_individuals'] = args.max_n_individuals
    
    if args.c1_start is not None:
        options['c1_start'] = args.c1_start
    
    if args.c1_end is not None:
        options['c1_end'] = args.c1_end
    
    if args.c2_start is not None:
        options['c2_start'] = args.c2_start
    
    if args.c2_end is not None:
        options['c2_end'] = args.c2_end
    
    if args.constriction is not None:
        options['constriction'] = args.constriction
    
    if args.max_v is not None:
        options['max_v'] = args.max_v
    
    if args.growth_rate is not None:
        options['growth_rate'] = args.growth_rate
    
    if args.learning_period is not None:
        options['learning_period'] = args.learning_period
    
    if args.memory_size is not None:
        options['memory_size'] = args.memory_size
    
    if args.stagnation_threshold is not None:
        options['stagnation_threshold'] = args.stagnation_threshold
    
    if args.max_gens is not None:
        options['max_gens'] = args.max_gens
    
    if args.y_best is not None:
        options['y_best'] = np.array(args.y_best)


    optimizer=IPSO(problem, options)
    start=perf_counter_ns()
    results=optimizer.optimize()
    end=perf_counter_ns()
    elapsed_time_ms=(end-start) / 1e6
    optimizer.print_report(results)
    print(f"Elapsed time: {elapsed_time_ms:.4f} ms")

if __name__ == "__main__":
    main()
# python .\run_ipso.py --cost_function paraboloid --ndim_problem 5 --lower_bound -100 --upper_bound 100 --max_evals 2000 --n_individuals 10
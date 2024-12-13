import argparse
from src.de.ade import *
from src.core.benchmarks import *
from src.core.optimizer import *
from time import perf_counter_ns

def main():
    #problem
    parser=argparse.ArgumentParser(description="Black box optimization - ADE") 
    parser.add_argument('--cost_function',type=str, required=True, choices=list(FUNCTIONS.keys()), help="The cost function to optimize")
    parser.add_argument('--ndim_problem',type=int,default=2,help="The number of dimensions of the problem")
    parser.add_argument('--upper_bound',type=float,required=True,help="The upper bound of the optimization problem")
    parser.add_argument('--lower_bound',type=float,required=True,help="The lower bound of the optimization problem")
    #options optimizer
    parser.add_argument('--max_evals',type=int,default=None, help="Maximum number of allowed evaluations (default:infinity)")
    parser.add_argument('--n_individuals',type=int,default=None,help="Number of individuals in the population")
    parser.add_argument('--seed_rng',type=int,default=42,help="Random seed for the RNG")
    parser.add_argument('--seed_initialization',type=int,default=42,help="Random seed for initialization")
    parser.add_argument('--seed_optimization',type=int,default=42,help="Random seed for optimization")
    parser.add_argument('--verbose',type=int,default=0, help="Enable verbose output (default:False)")
    parser.add_argument('--n_evals',type=int,default=None,help="Number of currently used evaluations (default:0)")
    parser.add_argument('--y_best', type=float, default=np.inf, help="Best fitness value known (default: np.inf)")
    #options algo
    parser.add_argument('--n_mu', type=float, default=0.5, help="Mean of normal distribution for crossover (default: 0.5)")
    parser.add_argument('--median', type=float, default=0.5, help="Location of Cauchy distribution for mutation (default: 0.5)")
    parser.add_argument('--p', type=float, default=0.05, help="Greediness parameter (default: 0.05)")
    parser.add_argument('--c', type=float, default=0.1, help="Adaptation rate (default: 0.1)")
    
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

    if args.n_individuals is not None:
        options['n_individuals'] = args.n_individuals

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

    if args.n_mu is not None:
        options['n_mu'] = args.n_mu

    if args.median is not None:
        options['median'] = args.median

    if args.p is not None:
        options['p'] = args.p

    if args.c is not None:
        options['c'] = args.c
    
    if args.y_best is not None:
        options['y_best'] = args.y_best
    
    optimizer=ADE(problem, options)
    start=perf_counter_ns()
    results=optimizer.optimize()
    end=perf_counter_ns()
    elapsed_time_ms=(end-start) / 1e6
    optimizer.print_report(results)
    print(f"Elapsed time: {elapsed_time_ms:.4f} ms")

if __name__ == "__main__":
    main()
    
# python run_ade.py --cost_function paraboloid --ndim_problem 2 --upper_bound 100 --lower_bound -100 --n_individuals 100  --n_mu 0.6 --median 0.4 --p 0.1 --c 0.05 --verbose 100 --max_evals 1000
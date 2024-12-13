import argparse
from src.cmaes.cmaes import *
from src.core.benchmarks import *
from src.core.optimizer import *
from time import perf_counter_ns

def main():
    #problem
    parser=argparse.ArgumentParser(description="Black box optimization - CMAES") 
    parser.add_argument('--cost_function',type=str, required=True, choices=list(FUNCTIONS.keys()), help="The cost function to optimize")
    parser.add_argument('--ndim_problem',type=int,default=2,help="The number of dimensions of the problem")
    parser.add_argument('--upper_bound',type=float,required=True,help="The upper bound of the optimization problem")
    parser.add_argument('--lower_bound',type=float,required=True,help="The lower bound of the optimization problem")
    #options
    parser.add_argument('--max_evals',type=int,default=None, help="Maximum number of allowed evaluations (default:infinity)")
    parser.add_argument('--n_individuals',type=int,default=None,help="Number of individuals in the population")
    parser.add_argument('--seed_rng',type=int,default=42,help="Random seed for the RNG")
    parser.add_argument('--seed_initialization',type=int,default=42,help="Random seed for initialization")
    parser.add_argument('--seed_optimization',type=int,default=42,help="Random seed for optimization")
    parser.add_argument('--verbose',type=int,default=0, help="Enable verbose output (default:False)")
    parser.add_argument('--n_evals',type=int,default=None,help="Number of currently used evaluations (default:0)")
    parser.add_argument('--y_best', type=float, default=np.inf, help="Best fitness value known (default: np.inf)")
    #options algo
    parser.add_argument('--n_parents', type=int, default=None,help="Number of parents. Default is half of n_individuals.")
    parser.add_argument('--mean', type=float, nargs='*', default=None,help="Mean vector for initialization.")
    parser.add_argument('--sigma', type=float,default=1,help="Standard deviation for initialization.")      
    
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

    if args.n_parents is not None:
        options['n_parents'] = args.n_parents

    if args.mean is not None:
        options['mean'] = np.array(args.mean)

    if args.sigma is not None:
        options['sigma'] = args.sigma

    if args.y_best is not None:
        options['y_best'] = args.y_best
    
    optimizer=CMAES(problem, options)
    start=perf_counter_ns()
    results=optimizer.optimize()
    end=perf_counter_ns()
    elapsed_time_ms=(end-start) / 1e6
    optimizer.print_report(results)
    print(f"Elapsed time: {elapsed_time_ms:.4f} ms")

if __name__ == "__main__":
    main()

#  python .\run_cmaes.py --cost_function paraboloid --ndim_problem 5 --upper_bound 100 --lower_bound -100 --max_evals 1000 --n_individuals 5 --sigma 0.5

import copy
from turtle import color
import plotly.express as px
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from src.cmaes.cmaes import *
from src.cmaes.fcmaes import *
from src.de.ade import *
from src.de.cde import *
from src.pso.ipso import *
from src.pso.spso import *
from src.core.benchmarks import *
from time import perf_counter_ns

LOWER_BOUND = -100
UPPER_BOUND = 100
NDIM_PROBLEM = 5
OPTIMAL_Y = 0
EPSILON = 1e-3
N_INDIVIDUALS = 100

options = {
    "cmaes": {
        "max_evals": 10000,
        "n_individuals": N_INDIVIDUALS,
        "mean": np.array(0),
        "sigma": 0.5,
    },
    "fcmaes": {
        "max_evals": 10000,
        "n_individuals": N_INDIVIDUALS,
        "mean": np.array(0),
        "sigma": 0.5,
    },
    "ade": {
        "max_evals": 10000,
        "n_individuals": N_INDIVIDUALS,
    },
    "cde": {
        "max_evals": 10000,
        "n_individuals": N_INDIVIDUALS,
    },
    "ipso": {
        "max_evals": 10000,
        "n_individuals": N_INDIVIDUALS,
    },
    "spso": {
        "max_evals": 10000,
        "n_individuals": N_INDIVIDUALS,
    },
}


def accuracy(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return abs(x-y)


algorithms = {
    "cmaes": CMAES,
    "fcmaes": FCMAES,
    "ade": ADE,
    "cde": CDE,
    "ipso": IPSO,
    "spso": SPSO,
}

benchmark_functions = {
    "Paraboloid": paraboloid,
    "Rastrigin": rastrigin,
    "Discus": discus,
    "Xin_She_Yang": xin_she_yang,
}

results = {
    "Algorithm": [],
    "Benchmark Function": [],
    "Accuracy": [],
    # "MinEvals": [],
    "Runtime": [],
}
problem = {
    "ndim_problem": NDIM_PROBLEM,
    "lower_bound": LOWER_BOUND,
    "upper_bound": UPPER_BOUND,
}
# for func_name, func in benchmark_functions.items():
#     for algorithm_name, algorithm_class in algorithms.items():
#         algo_options = options[algorithm_name]
#         problem["cost_function"] = func
#         algorithm = algorithm_class(problem, algo_options)
#         start = perf_counter_ns()
#         res = algorithm.optimize()
#         end = perf_counter_ns()
#         runtime = (end - start) / 1_000_000  
#         acc = accuracy(res["y_best"], OPTIMAL_Y)
#         results["Algorithm"].append(algorithm_name)
#         results["Benchmark Function"].append(func_name)
#         results["Accuracy"].append(acc)
#         results["Runtime"].append(runtime)

for func_name, func in benchmark_functions.items():
    for algorithm_name, algorithm_class in algorithms.items():
        algo_options = options[algorithm_name]
        problem["cost_function"] = func
        algorithm = algorithm_class(problem, algo_options)
        start = perf_counter_ns()
        res = algorithm.optimize()
        end = perf_counter_ns()
        runtime = (end - start) / 1_000_000  
        acc = accuracy(res['y_best'],OPTIMAL_Y)
        # l=1
        # r=int(2e5)
        # mxeval=r
        # algo_options_temp = copy.deepcopy(algo_options)
        # while l<=r:
        #     mid = (l+r)//2
        #     print(mid)
        #     algo_options_temp['max_evals'] = mid
        #     algorithm = algorithm_class(problem, algo_options_temp)
        #     res= algorithm.optimize()
        #     if accuracy(res['y_best'],OPTIMAL_Y) <=EPSILON:
        #         r=mid-1
        #         mxeval =r
        #     else:
        #         l=mid+1
        results["Algorithm"].append(algorithm_name)
        results["Benchmark Function"].append(func_name)
        # results["MinEvals"].append(mxeval)
        results["Accuracy"].append(acc)
        results["Runtime"].append(runtime)


results_df = pd.DataFrame(results)

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Algorithm', y='Accuracy', data=results_df,hue='Benchmark Function')
# plt.title('Accuracy Comparison of Algorithms')
# plt.xticks(rotation=45)
# plt.show()

# Plot runtime vs algorithm (line plot)
# plt.figure(figsize=(10, 6))
# sns.lineplot(x='Algorithm', y='Runtime', data=results_df,hue='Benchmark Function')
# plt.title('Runtime Comparison of Algorithms')
# plt.xticks(rotation=45)
# plt.show()

fig_runtime = px.line(
    results_df,
    x="Benchmark Function",
    y="Runtime",
    color="Algorithm",
    title="Runtime Comparison of Algorithms",
    labels={"Runtime": "Runtime (ms)", "Algorithm": "Algorithm"},
)
fig_runtime.update_xaxes(tickangle=45)
fig_runtime.show()

fig_runtime = px.line(
    results_df,
    x="Benchmark Function",
    y="Accuracy",
    color="Algorithm",
    title="Accuracy Comparison of Algorithms",
    labels={"Accuracy": "|ypred - ytrue|", "Algorithm": "Algorithm"},
)
fig_runtime.update_xaxes(tickangle=45)
fig_runtime.show()

# fig_runtime = px.line(
#     results_df,
#     x="Benchmark Function",
#     y="MinEvals",
#     color="Algorithm",
#     title="Minimum Evaluations to get a certain threshold of accuracy",
#     labels={"MinEvals": "MinEvals", "Algorithm": "Algorithm"},
# )

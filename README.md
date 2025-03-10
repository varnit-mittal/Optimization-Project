# Comparing Evolutionary Algorithms for Black-Box Optimization


This project explores **black-box optimization**, a challenging domain where the mathematical structure of the objective function is inaccessible, and solutions are derived purely through evaluations. We implemented and compared three widely used evolutionary algorithms (EAs):

- **Differential Evolution (DE)**
- **Covariance Matrix Adaption Evolution Strategy (CMA-ES)**
- **Particle Swarm Optimization (PSO)**

In addition to these standard algorithms, we implemented and analyzed **enhanced variants** to improve their efficiency and adaptability in solving complex black-box optimization problems.

## Motivation

Black-box Optimization is crucial in various fields such as:

- Hyperparameter tuning in Machine Learning 
- Engineering Design Optimization
- Financial Modeling

#### Challenges include:

- High-dimensional and non-convex search spaces 
- Multiple local optima 
- Limited budget for objective function evaluations

Evolutionary algorithms are ideal for these scenarios as they rely solely on function evaluations without requiring gradient information.

## Problem Statement

Given and objective function in **n-dimensional** space with defined upper and lower bounds for all dimensions, and a maximum number of evaluations, find the **optimizer** while adhering to the evaluation budget such that it is within the constraint of the search space.

## Algorithms Implemented

### Differential Evolution (DE)

A population-based metaheuristic that iteratively improves solutions using:

- **Mutation**: Creates diversity by combining solutions.
- **Crossover**: Exploits promising areas of the search space.
- **Selection**: Retains superior solutions.

#### Variant: Adaptive Differential Evolution (ADE)

**Enhancements:**

- Adaptive mutation and crossover rates  
- Archive-based diversity  
- Improved exploration-exploitation balance  

---

### Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

An advanced algorithm that adapts search distributions over time. Key features include:

- Adaptive covariance matrix for efficient exploration  
- Weighted averaging of solutions  

#### Variant: Fast CMA-ES (FCMAES)

**Optimizations:**

- Lightweight covariance updates  
- Hybrid sampling strategy  
- Dynamic step-size adaptation  

---

### Particle Swarm Optimization (PSO)

A swarm-based algorithm inspired by social behaviors. Features include:

- Position and velocity updates  
- Personal and global best tracking  

#### Variant: Improved Particle Swarm Optimization (IPSO)

**Improvements:**

- Dynamic swarm size  
- Adaptive coefficients for better exploration  
- Stagnation detection and diversity enhancement  

## Benchmarks and Results

The algorithms were benchamarked using a suite of black-box optimization problems to evaluate:

- Accuracy
- Convergence speed
- Runtime efficiency

Detailed analysis, graphs and metrics are included in the [report](https://github.com/varnit-mittal/Optimization-Project/blob/main/Report.pdf).


## Run Locally

Clone the project

```bash
  git clone https://github.com/varnit-mittal/Optimization-Project.git
```

Go to the project directory

```bash
  cd Optimization-Project
```

Install dependencies (only `numpy` and `scipy` are required)

```bash
  pip install -r requirements.txt
```

There are 6 run files in the project directory for user's convenience and a comment on how to run the respective evolutionary algorithm is written in each one of them.

### Example: Running Adaptive Differential Evolution

To run the ADE implementation, use the `run_ade.py` script with the following arguments:

#### Problem-Specific arguments

- `--cost_function`: The cost function to optimize (**required**). Choose from predefined functions in `src/core/benchmarks.py`.
- `--ndim_problem`: Number of dimensions in the optimization problem (**default**: 2).
- `--upper_bound`: Upper bound of the search space (**required**).
- `--lower_bound`: Lower bound of the search space (**required**).

#### Optimization Options

- `--max_evals`: Maximum number of allowed evaluations (**default**: 10000).  
- `--n_individuals`: Number of individuals in the population (**default**: 50).  
- `--seed_rng`: Random seed for reproducibility (**default**: 42).  
- `--seed_initialization`: Random seed for initialization (**default**: 42).  
- `--seed_optimization`: Random seed for optimization (**default**: 42).  
- `--verbose`: Enable verbose output (**default**: 0). Setting this to 10 (for example) will output the best *x* and the best function value every 10 iterations/generations.
- `--n_evals`: Number of evaluations already used (**default**: 0).  
- `--y_best`: Best known fitness value (**default**: `np.inf`).  

#### Algorithm-Specific Parameters (specific to ADE here)

- `--n_mu`: Mean of normal distribution for crossover (**default**: 0.5).  
- `--median`: Location of the Cauchy distribution for mutation (**default**: 0.5).  
- `--p`: Greediness parameter for selection (**default**: 0.05).  
- `--c`: Adaptation rate for parameters (**default**: 0.1).

To know the arguments for a particular script file, you can use the `--help` flag. A list of arguments with their descriptions will be shown.

#### Example Usage

```bash
    python3 run_ade.py --help
```

```bash
    python3 run_ade.py --cost_function paraboloid --ndim_problem 5 --upper_bound 100 --lower_bound -100 --n_individuals 5  --n_mu 0.6 --median 0.4 --p 0.1 --c 0.05  --max_evals 1000 --verbose 10
```

You can run the other files similarly.

### Test your own benchmark function

- Just add the function definition to `src/core/benchmarks.py` and use it while running the algorithms.

## References

- R. Storn and K. Price, *"Differential Evolution - A Simple and Efficient Heuristic for Global Optimization Over Continuous Spaces,"* 1997.  
- N. Hansen, *"The CMA Evolution Strategy: A Tutorial,"* 2023.  
- Z. Li et al., *"Fast Covariance Matrix Adaptation for Large-Scale Black-Box Optimization,"* 2020.  
- G. Venter and J. Sobieszczanski-Sobieski, *"Particle Swarm Optimization,"* 2003.  

Additional references are included in the [report](https://github.com/varnit-mittal/Optimization-Project/blob/main/Report.pdf).



## Authors

- [@varnit-mittal](https://github.com/varnit-mittal)
- [@mohit086](https://github.com/mohit086)
- [@ap5967ap](https://github.com/ap5967ap)


import numpy as np

# 1
def paraboloid(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10 * np.sum(x**2)

# 2
def rastrigin(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 3
def ackley(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

# 4
def rosenbrock(x):
    '''
    Optimum at [1, 1, 1,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# 5
def griewank(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    d = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(1 + np.arange(d)))) + 1

# 6
def sum_of_powers(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(np.abs(x)**(2 + np.arange(len(x))))

# 7
def step(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(np.floor(x + 0.5)**2)

# 8
def ripple_wave(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    z = np.linalg.norm(x)
    return z**2 + 10 * np.sin(5 * z)**2

# 9
def schwefel(x):
    '''
    Optimum at [420.9687, 420.9687,...]
    Optimum value = 0
    Recommended bounds: [-500, 500] hypercube
    '''
    return np.abs(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

# 10
def ripple_plane(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 100
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10 * np.sin(100 * np.linalg.norm(x)) - np.sum(x) + 100

# 11
def narrow_valley(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(x**2) - np.exp(-np.sum(x**2) * 100) + 1

# 12
def drop_wave(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = -1
    Recommended bounds: [-100, 100] hypercube
    '''
    r = np.sqrt(np.sum(x**2))
    return - (1 + np.cos(12 * r)) / (0.5 * r**2 + 2)

# 13
def discus(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10**6 * x[0]**2 + np.sum(x[1:]**2)

# 14
def happycat(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0.5
    Recommended bounds: [-100, 100] hypercube
    '''
    d = len(x)
    return (np.abs(np.sum(x**2) - d)**(1/4)) + ((0.5 * np.sum(x**2) + np.sum(x)) / d) + 0.5

# 15
def xin_she_yang(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return (np.sum(np.abs(x))) * np.exp(-(np.sum(np.sin(x**2))))

# 16
def zakharov(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    d = len(x)
    term1 = np.sum(x**2)
    term2 = np.sum(0.5 * np.arange(1, d+1) * x)
    return term1 + term2**2 + term2**4

# 17
def levy(x):
    '''
    Optimum at [1, 1, 1,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[1:] - 1)**2 * (1 + 10 * np.sin(np.pi * w[1:])**2))
    return term1 + term2

FUNCTIONS = {
    'paraboloid':paraboloid,
    'rastrigin':rastrigin,
    'ackley':ackley,
    'rosenbrock':rosenbrock,
    'griewank':griewank,
    'sum_of_powers':sum_of_powers,
    'step':step,
    'ripple_wave':ripple_wave,
    'schwefel':schwefel,
    'ripple_plane':ripple_plane,
    'narrow_valley':narrow_valley,
    'drop_wave':drop_wave,
    'discus':discus,
    'happycat':happycat,
    'xin_she_yang':xin_she_yang,
    'zakharov':zakharov,
    'levy':levy
}
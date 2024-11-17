import numpy as np

# 1
def paraboloid(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return 5 * np.sum(x**2)

# 2
def ellipsoid(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    d = len(x)
    return np.sum((10**6) ** (np.arange(d) / (d - 1)) * (x)**2)

# 3
def rastrigin(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 4
def ackley(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

# 5
def rosenbrock(x):
    '''
    Optimum at [1, 1, 1,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


# 6
def griewank(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    d = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(1 + np.arange(d)))) + 1

# 7
def sum_of_powers(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(np.abs(x)**(2 + np.arange(len(x))))

# 8
def step(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(np.floor(x + 0.5)**2)

# 9
def ripple_wave(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    z = np.linalg.norm(x)
    return z**2 + 10 * np.sin(5 * z)**2

# 10
def schwefel(x):
    '''
    Optimum at [420.9687, 420.9687,...]
    Optimum value = 0
    Recommended bounds: [-500, 500] hypercube
    '''
    return np.abs(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

# 11
def ripple_plane(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 100
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10 * np.sin(100 * np.linalg.norm(x)) - np.sum(x) + 100

# 12
def narrow_valley(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return np.sum(x**2) - np.exp(-np.sum(x**2) * 100) + 1

# 13
def drop_wave(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = -1
    Recommended bounds: [-100, 100] hypercube
    '''
    r = np.sqrt(np.sum(x**2))
    return - (1 + np.cos(12 * r)) / (0.5 * r**2 + 2)

# 14
def discus(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return 10**6 * x[0]**2 + np.sum(x[1:]**2)

# 15
def happycat(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0.5
    Recommended bounds: [-100, 100] hypercube
    '''
    d = len(x)
    return (np.abs(np.sum(x**2) - d)**(1/4)) + ((0.5 * np.sum(x**2) + np.sum(x)) / d) + 0.5

# 16
def xin_she_yang(x):
    '''
    Optimum at [0, 0, 0,...]
    Optimum value = 0
    Recommended bounds: [-100, 100] hypercube
    '''
    return (np.sum(np.abs(x))) * np.exp(-(np.sum(np.sin(x**2))))

# 17
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

# 18
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

# ------------------------------------------------------------------------------ #

# Functions below this are strictly 2-dimensional

# 19
def beale(x):
    '''
    Optimum at [3, 0.5]
    Optimum value = 0
    Recommended bounds: [-100, 100]
    '''
    assert len(x) == 2, f"Expected 2-dimensional input, but got {len(x)} dimensions."
    return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

# 20
def six_hump_camelback(x):
    '''
    Optimum at [0.0898, -0.7126] or [-0.0898, 0.7126]
    Optimum value = -1.0316
    Recommended bounds: [-100, 100]
    '''
    assert len(x) == 2, f"Expected 2-dimensional input, but got {len(x)} dimensions."
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

# 21
def booth(x):
    '''
    Optimum at [1, 3]
    Optimum value = 0
    Recommended bounds: [-100, 100]
    '''
    assert len(x) == 2, f"Expected 2-dimensional input, but got {len(x)} dimensions."
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

# 22
def goldstein_price(x):
    '''
    Optimum at [0, -1]
    Optimum value = 3
    Recommended bounds: [-100, 100]
    '''
    assert len(x) == 2, f"Expected 2-dimensional input, but got {len(x)} dimensions."
    term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2
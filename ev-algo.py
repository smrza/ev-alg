import numpy as np
import concurrent.futures
import os

##########################
# functions - https://github.com/thieu1995/opfunu/blob/092327e4928564ee946bc77cf2a3c24b3cb54922/opfunu/dimension_based/benchmarknd.py
# https://gist.github.com/denis-bz/da697d8bc74fae4598bf
##########################
# 1
def ackley( x, a=20, b=0.2, c=2*np.pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( np.cos( c * x ))
    return -a*np.exp( -b*np.sqrt( s1 / n )) - np.exp( s2 / n ) + a + np.exp(1)

# 2
def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = np.prod( np.cos( x / np.sqrt(j) ))
    return s/fr - p + 1

# 3
def levy( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (np.sin( np.pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * np.sin( np.pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + np.sin( 2 * np.pi * z[-1] )**2 ))

# 4
michalewicz_m = .5  # orig 10: ^20 => underflow
def michalewicz( x ):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( np.sin(x) * np.sin( j * x**2 / np.pi ) ** (2 * michalewicz_m) )

# 5
def dixonprice( x ):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2

# 6
def perm( x, b=.5 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    xbyj = np.fabs(x) / j
    return np.mean([ np.mean( (j**k + b) * (xbyj ** k - 1) ) **2
            for k in j/n ])
    # original overflows at n=100 --
    # return sum([ sum( (j**k + b) * ((x / j) ** k - 1) ) **2
    #       for k in j ])

# 7
def powersum( x, b=[8,18,44,114] ):  # power.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
    return s

# 8
def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * np.cos( 2 * np.pi * x ))

# 9
def rosenbrock( x ):  # rosen.m
    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
        # a sum of squares, so LevMar (scipy.optimize.leastsq) is pretty good
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))

# 10
def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * np.sin( np.sqrt( abs( x ))))

# 11
def trid( x ):
    x = np.asarray_chkfinite(x)
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

# 12
def nesterov( x ):
    """ Nesterov's nonsmooth Chebyshev-Rosenbrock function, Overton 2011 variant 2 """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return abs( 1 - x[0] ) / 4 \
        + sum( abs( x1 - 2*abs(x0) + 1 ))

# 13
def alpine_n1(x):
    x = np.asarray_chkfinite(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

# 14
def qing(x):
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        result += (x[i] ** 2 - i - 1) ** 2
    return result

# 15
def salomon(x):
    x = np.asarray_chkfinite(x)
    return 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2))) + 0.1 * np.sqrt(np.sum(x ** 2))

# 16
def styblinski(x):
    x = np.asarray_chkfinite(x)
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

# 17
def happy_cat(x, alpha=1.0/8):
    """
    Class: multimodal, non-convex, differentiable, non-separable, parametric
    Global: one global minimum fx = 0, at [-1, ..., -1]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/happycatfcn.html

    @param solution: A numpy array with x_i in [-2, 2]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    return ((np.sum(x**2) - len(x))**2)**alpha + (0.5*np.sum(x**2)+np.sum(x))/len(x) + 0.5

# 18
def quartic(x):
    """
    Class: multimodal, non-convex, differentiable, separable, continuous, random
    Global: one global minimum fx = 0 + random, at (0, ...,0)
    Link: http://benchmarkfcns.xyz/benchmarkfcns/quarticfcn.html

    @param solution: A numpy array with x_i in [-1.28, 1.28]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        result+= (i+1)*x[i]**4
    return result+np.random.uniform(0, 1)

# 19
def shubert_3(x):
    """
    Class: multi-modal, non-convex, differentiable, separable, continuous
    Global: one global minimum fx = -29.6733337
    Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert3fcn.html

    @param solution: A numpy array with x_i in [-10, 10]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        for j in range(1, 6):
            result+= j*np.sin((j+1)*x[i] + j)
    return result

# 20
def shubert_4(x):
    """
    Class: multi-modal, non-convex, differentiable, separable, continuous
    Global: one global minimum fx = -25.740858
    Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert4fcn.html

    @param solution: A numpy array with x_i in [-10, 10]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        for j in range(1, 6):
            result += j * np.cos((j + 1) * x[i] + j)
    return result

# 21
def shubert(x):
    """
    Class: multi-modal, non-convex, differentiable, non-separable, continuous
    Global: one global minimum fx = 0, at [0, ..., 0]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/shubertfcn.html

    @param solution: A numpy array with x_i in [-100, 100]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    prod = 1.0
    for i in range(0, d):
        result = 0
        for j in range(1, 6):
            result += np.cos((j + 1) * x[i] + j)
        prod *= result
    return prod

# 22
def ackley_n4(x):
    """
    Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
    Global: on 2-d space, 1 global min fx = -4.590101633799122, at [−1.51, −0.755]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/ackleyn4fcn.html

    @param solution: A numpy array include 2 items like: [-35, 35, -35, ...]
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    score = 0.0
    for i in range(0, d-1):
        score += ( np.exp(-0.2*np.sqrt(x[i]**2 + x[i+1]**2)) + 3*(np.cos(2*x[i]) + np.sin(2*x[i+1])) )
    return score

# 23
def alpine_n2(x):
    """
    Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
    Global: one global minimum fx = 2.808^n, at [7.917, ..., 7.917]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/alpinen2fcn.html

    @param solution: A numpy array like: [1, 2, 10, 4, ...]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    x_abs = np.abs(x)
    return np.prod(np.sqrt(x_abs)*np.sin(x))

# 24
def xin_she_yang_n2(x):
    """
    Class: multi-modal, non-convex, non-differentiable, non-separable
    Global: one global minimum fx = 0, at [0, ..., 0]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn2fcn.html

    @param solution: A numpy array with x_i in [-2pi, 2pi]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    return np.sum(np.abs(x))*np.exp(-np.sum(np.sin(x**2)))

# 25
def xin_she_yang_n4(x):
    """
    Class: multi-modal, non-convex, non-differentiable, non-separable
    Global: one global minimum fx = -1, at [0, ..., 0]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn4fcn.html

    @param solution: A numpy array with x_i in [-10, 10]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    t1 = np.sum(np.sin(x)**2)
    t2 = -np.exp(-np.sum(x**2))
    t3 = -np.exp(np.sum(np.sin(np.sqrt(np.abs(x)))**2))
    return (t1 + t2) * t3

####################
# SOMA - https://ivanzelinka.eu/somaalgorithm/Codes.html
####################
#SOMA parameters
prt = 0.7
path_lenght = 3
step = 0.11

class Individual:
    """Individual of the population. It holds parameters of the solution as well as the fitness of the solution."""
    def __init__(self, params, fitness):
        self.params = params
        self.fitness = fitness

    def __repr__(self):
        return '{} fitness: {}'.format(self.params, self.fitness)

def evaluate(function, params):
    """Returns fitness of the params"""
    return function(params)

# https://chat.openai.com/share/2c03a887-42e0-4078-aeb0-77f9b366414e
def bounded(params, min_s: list, max_s: list):
    """
    Returns bounded version of params with reflection boundary check.
    All params that are outside of bounds (min_s, max_s) are reflected back into the bounds.
    """
    def reflect(param, min_val, max_val):
        if param < min_val:
            return min_val + (min_val - param)
        elif param > max_val:
            return max_val - (param - max_val)
        return param

    return np.array([reflect(params[d], min_s[d], max_s[d]) for d in range(len(params))])

def generate_population(size, min_s, max_s, dimension, function):
    def generate_individual():
        params = np.random.uniform(min_s, max_s, dimension)
        fitness = evaluate(function, params)
        return Individual(params, fitness)
    return [generate_individual() for _ in range(size)]

def generate_prt_vector(prt, dimension):
    return np.random.choice([0, 1], dimension, p=[prt, 1-prt])

def get_leader(population):
    """Finds leader of the population by its fitness (the lower the better)."""
    return min(population, key = lambda individual: individual.fitness)

def soma_all_to_one(population, prt, path_length, step, migrations, min_s, max_s, dimension, function):
    for generation in range(migrations):
        leader = get_leader(population)
        for individual in population:
            if individual is leader:
                continue
            next_position = individual.params
            prt_vector = generate_prt_vector(prt, dimension)
            for t in np.arange(step, path_length, step):
                current_position = individual.params + (leader.params - individual.params) * t * prt_vector
                current_position = bounded(current_position, min_s, max_s)
                fitness = evaluate(function, current_position)
                if fitness <= individual.fitness:
                    next_position = current_position
                    individual.fitness = fitness
            individual.params = next_position
    return get_leader(population)

def soma_all_to_all(population, prt, path_length, step, migrations, min_s, max_s, dimension, function):
    for generation in range(migrations):
        for individual in population:
            next_position = individual.params
            prt_vector = generate_prt_vector(prt, dimension)
            for leading in population:
                if individual is leading:
                    continue
                for t in np.arange(step, path_length, step):
                    current_position = individual.params + (leading.params - individual.params) * t * prt_vector
                    current_position = bounded(current_position, min_s, max_s)
                    fitness = evaluate(function, current_position)
                    if fitness <= individual.fitness:
                        next_position = current_position
                        individual.fitness = fitness
            individual.params = next_position
    return get_leader(population)

####################
# DE - https://gist.github.com/pablormier/0caff10a5f76e87857b44f63757729b0
####################
def de_rand_1_bin(fobj, bounds, popsize, its, mut=0.8, crossp=0.9):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    f = 0
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)

            # Reflective boundary check
            exceeding_min = mutant < 0
            exceeding_max = mutant > 1
            mutant[exceeding_min] = -mutant[exceeding_min]
            mutant[exceeding_max] = 2 - mutant[exceeding_max]

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        return f

def de_best_1_bin(fobj, bounds, popsize, its, mut=0.5, crossp=0.9):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    f = fitness[best_idx]
    
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b = pop[np.random.choice(idxs, 2, replace=False)]
            mutant = np.clip(best + mut * (a - b), 0, 1)

            # Reflective boundary check
            exceeding_min = mutant < 0
            exceeding_max = mutant > 1
            mutant[exceeding_min] = -mutant[exceeding_min]
            mutant[exceeding_max] = 2 - mutant[exceeding_max]

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        return f

####################
# PSO - https://chat.openai.com/share/475ad76c-eff8-485e-beed-d993a2a63611
####################
def pso(function, dimensions, population_size, iterations, c1 = 1.49618, c2 = 1.49618, w = 0.7298):
    lower_bound = -100
    upper_bound = 100

    # Initialize particles
    particles_position = lower_bound + np.random.rand(population_size, dimensions) * (upper_bound - lower_bound)
    particles_velocity = np.random.rand(population_size, dimensions)
    personal_best_position = particles_position.copy()
    personal_best_value = np.apply_along_axis(function, 1, personal_best_position)

    # Initialize global best
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index, :]

    for _ in range(iterations):
        # Update particle velocities and positions
        r1, r2 = np.random.rand(population_size, dimensions), np.random.rand(population_size, dimensions)
        particles_velocity = w * particles_velocity + c1 * r1 * (personal_best_position - particles_position) + c2 * r2 * (global_best_position - particles_position)
        particles_position = particles_position + particles_velocity

        # Reflective boundary check
        particles_position = np.clip(particles_position, lower_bound, upper_bound)

        # Update personal best
        current_value = np.apply_along_axis(function, 1, particles_position)
        update_indices = current_value < personal_best_value
        personal_best_position[update_indices, :] = particles_position[update_indices, :]
        personal_best_value[update_indices] = current_value[update_indices]

        # Update global best
        global_best_index = np.argmin(personal_best_value)
        global_best_position = personal_best_position[global_best_index, :]

    #return global_best_position, personal_best_value[global_best_index]
    return personal_best_value[global_best_index]


####################
# main
####################
####
# settings
####
def write_results(function_name, dimension, results, alg_name):
    base_dir = "results-latest"
    dimension_dir = os.path.join(base_dir, alg_name, str(dimension))
    file_path = os.path.join(dimension_dir, function_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(results))

def select_best(dict, dimension, function):
    key = (dimension, function)
    if key in dict:
        results = dict[key]
        best_result = min(results)
        return best_result
    else:
        print(f"No results found for dimension={dimension}, function={function}")
        return None

dimensions_list = [2, 10, 30]
population_sizes = {2: 10, 10: 10, 30: 50}
iterations = 30
functionEvaluations = 2000
# functions = [ackley, griewank, michalewicz, levy, dixonprice, perm, powersum, rastrigin, rosenbrock, schwefel, trid, nesterov, 
#              alpine_n1, qing, salomon, styblinski, happy_cat, quartic, shubert_3, shubert_4, shubert, ackley_n4, alpine_n2,
#              xin_she_yang_n2, xin_she_yang_n4]
functions = [ackley, griewank, michalewicz]

####
# pso
####
def evaluate_pso(cost_function, dimensions, population_size, total_evaluations):
    try:
        result = pso(cost_function, dimensions, population_size, total_evaluations)
        return dimensions, cost_function.__name__, result
    except Exception as e:
        return dimensions, cost_function.__name__, None
    
def calculate_pso(cost_function):
    results_dict_pso = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for dimensions in dimensions_list:
            population_size = population_sizes[dimensions]
            total_evaluations = functionEvaluations * dimensions
            
            for iteration in range(iterations):
                # print(f"Running dimension={dimensions}, function={cost_function.__name__}, iteration={iteration + 1}")
                future = executor.submit(
                    evaluate_pso, cost_function, dimensions, population_size, total_evaluations
                )
                futures.append(future)

        concurrent.futures.wait(futures)

        for future in futures:
            try:
                dimension, function, result = future.result()
                key = (dimension, function)
                if key not in results_dict_pso:
                    results_dict_pso[key] = []
                results_dict_pso[key].append(result)
            except Exception as e:
                print(f"Error retrieving result: {e}")

    for dimensions in dimensions_list:
        best_result = select_best(results_dict_pso, dimensions, cost_function.__name__)
        # print(f"Best result for dimension={dimensions}, function={cost_function.__name__}: {best_result}")
        write_results(cost_function.__name__, dimensions, best_result, "PSO")

####
# soma - all to one
####
def evaluate_soma_all_to_one(cost_function, dimensions, population_size, total_evaluations):
    try:
        bounds_min = []
        bounds_max = []
        for i in range(dimensions):
            bounds_min.append(-100)
        for i in range(dimensions):
            bounds_max.append(100)
        population = generate_population(population_size, bounds_min, bounds_max, dimensions, cost_function)
        result = soma_all_to_one(population, prt, path_lenght, step, total_evaluations, bounds_min, bounds_max, dimensions, cost_function)
        return dimensions, cost_function.__name__, result
    except Exception as e:
        return dimensions, cost_function.__name__, None
    
def calculate_soma_all_to_one(cost_function):
    results_dict_soma = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for dimensions in dimensions_list:
            population_size = population_sizes[dimensions]
            total_evaluations = functionEvaluations * dimensions
            
            for iteration in range(iterations):
                # print(f"Running dimension={dimensions}, function={cost_function.__name__}, iteration={iteration + 1}")
                future = executor.submit(
                    evaluate_soma_all_to_one, cost_function, dimensions, population_size, total_evaluations
                )
                futures.append(future)

        concurrent.futures.wait(futures)

        for future in futures:
            try:
                dimension, function, result = future.result()
                key = (dimension, function)
                if key not in results_dict_soma:
                    results_dict_soma[key] = []
                str_result = str(result)
                start_index = str_result.find("[[") + 2
                end_index = str_result.find("]")
                found_array_str = str_result[start_index:end_index]
                result_to_append = min(np.fromstring(found_array_str, sep=' '))
                results_dict_soma[key].append(result_to_append)
            except Exception as e:
                print(f"Error retrieving result: {e}")

    for dimensions in dimensions_list:
        best_result = select_best(results_dict_soma, dimensions, cost_function.__name__)
        #print(f"Best result for dimension={dimensions}, function={cost_function.__name__}: {best_result}")
        write_results(cost_function.__name__, dimensions, best_result, "SOMA_all_to_one")
            
####
# soma - all to all
####
def evaluate_soma_all_to_all(cost_function, dimensions, population_size, total_evaluations):
    try:
        bounds_min = []
        bounds_max = []
        for i in range(dimensions):
            bounds_min.append(-100)
        for i in range(dimensions):
            bounds_max.append(100)
        population = generate_population(population_size, bounds_min, bounds_max, dimensions, cost_function)
        result = soma_all_to_all(population, prt, path_lenght, step, total_evaluations, bounds_min, bounds_max, dimensions, cost_function)
        return dimensions, cost_function.__name__, result
    except Exception as e:
        return dimensions, cost_function.__name__, None
    
def calculate_soma_all_to_all(cost_function):
    results_dict_soma = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for dimensions in dimensions_list:
            population_size = population_sizes[dimensions]
            total_evaluations = functionEvaluations * dimensions
            
            for iteration in range(iterations):
                # print(f"Running dimension={dimensions}, function={cost_function.__name__}, iteration={iteration + 1}")
                future = executor.submit(
                    evaluate_soma_all_to_all, cost_function, dimensions, population_size, total_evaluations
                )
                futures.append(future)

        concurrent.futures.wait(futures)

        for future in futures:
            try:
                dimension, function, result = future.result()
                key = (dimension, function)
                if key not in results_dict_soma:
                    results_dict_soma[key] = []
                str_result = str(result)
                start_index = str_result.find("[[") + 2
                end_index = str_result.find("]")
                found_array_str = str_result[start_index:end_index]
                result_to_append = min(np.fromstring(found_array_str, sep=' '))
                results_dict_soma[key].append(result_to_append)
            except Exception as e:
                print(f"Error retrieving result: {e}")

    for dimensions in dimensions_list:
        best_result = select_best(results_dict_soma, dimensions, cost_function.__name__)
        #print(f"Best result for dimension={dimensions}, function={cost_function.__name__}: {best_result}")
        write_results(cost_function.__name__, dimensions, best_result, "SOMA_all_to_all")

####
# DE rand/1/bin
####
def evaluate_de_rand_1_bin(cost_function, dimensions, population_size, total_evaluations):
    try:
        result = de_rand_1_bin(cost_function, [-100, 100], population_size, total_evaluations)
        return dimensions, cost_function.__name__, result
    except Exception as e:
        return dimensions, cost_function.__name__, None
    
def calculate_de_rand_1_bin(cost_function):
    results_dict_de_rand_1_bin = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for dimensions in dimensions_list:
            population_size = population_sizes[dimensions]
            total_evaluations = functionEvaluations * dimensions
            
            for iteration in range(iterations):
                # print(f"Running dimension={dimensions}, function={cost_function.__name__}, iteration={iteration + 1}")
                future = executor.submit(
                    evaluate_de_rand_1_bin, cost_function, dimensions, population_size, total_evaluations
                )
                futures.append(future)

        concurrent.futures.wait(futures)

        for future in futures:
            try:
                dimension, function, result = future.result()
                key = (dimension, function)
                if key not in results_dict_de_rand_1_bin:
                    results_dict_de_rand_1_bin[key] = []
                results_dict_de_rand_1_bin[key].append(result)
            except Exception as e:
                print(f"Error retrieving result: {e}")

    for dimensions in dimensions_list:
        best_result = select_best(results_dict_de_rand_1_bin, dimensions, cost_function.__name__)
        # print(f"Best result for dimension={dimensions}, function={cost_function.__name__}: {best_result}")
        write_results(cost_function.__name__, dimensions, best_result, "DE_rand_1_bin")

####
# DE best/1/bin
####
def evaluate_de_best_1_bin(cost_function, dimensions, population_size, total_evaluations):
    try:
        result = de_best_1_bin(cost_function, [-100, 100], population_size, total_evaluations)
        return dimensions, cost_function.__name__, result
    except Exception as e:
        return dimensions, cost_function.__name__, None
    
def calculate_de_best_1_bin(cost_function):
    results_dict_de_best_1_bin = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for dimensions in dimensions_list:
            population_size = population_sizes[dimensions]
            total_evaluations = functionEvaluations * dimensions
            
            for iteration in range(iterations):
                # print(f"Running dimension={dimensions}, function={cost_function.__name__}, iteration={iteration + 1}")
                future = executor.submit(
                    evaluate_de_best_1_bin, cost_function, dimensions, population_size, total_evaluations
                )
                futures.append(future)

        concurrent.futures.wait(futures)

        for future in futures:
            try:
                dimension, function, result = future.result()
                key = (dimension, function)
                if key not in results_dict_de_best_1_bin:
                    results_dict_de_best_1_bin[key] = []
                results_dict_de_best_1_bin[key].append(result)
            except Exception as e:
                print(f"Error retrieving result: {e}")

    for dimensions in dimensions_list:
        best_result = select_best(results_dict_de_best_1_bin, dimensions, cost_function.__name__)
        # print(f"Best result for dimension={dimensions}, function={cost_function.__name__}: {best_result}")
        write_results(cost_function.__name__, dimensions, best_result, "DE_best_1_bin")

####
# run
####
# for function in functions:
#     calculate_pso(function)
# for function in functions:
#     calculate_de_rand_1_bin(function)
# for function in functions:
#     calculate_de_best_1_bin(function)
for function in functions:
    calculate_soma_all_to_one(function)
for function in functions:
    calculate_soma_all_to_all(function)

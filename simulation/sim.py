import numpy as np
import sys
from loss_linear import claculate_bias, hard_linear_classifier, lower_bound
from faith import faith_test
import json
import multiprocessing as mp
from functools import partial
import itertools

# i = int(float(sys.argv[1]))
# cpus = int(float(sys.argv[2]))
# theta2 = np.arange(-4, 4.1, step = 0.4)
# x, y = np.load(f'data/x_{i}.npy'), np.load(f'data/y_{i}.npy')



def fair_distance(x1, x2):
    return np.abs(x1[0] - x2[0])

def c_distance(z1, z2, infty_equiv = 1000):
    x1, y1 = z1
    x2, y2 = z2
    if y1 != y2:
        return infty_equiv
    else:
        return fair_distance(x1, x2)


# test_stats = []
# for t in theta2:
#     theta = [1, t]
#     bias = claculate_bias(theta, x, y)
#     classifier = hard_linear_classifier(theta, bias)

    
#     T_n, T_n_tilde = lower_bound(x, y, theta=theta, bias=bias, fair_direction=[0, 1], num_steps = 500, learning_rate=2e-2, regularizer=200, cpus=cpus)
#     psi_n = faith_test(x, y, c_distance, classifier, delta = 1e-5, cpus=cpus)
#     test_stats.append({'t2': t, 'T': T_n, 'T-tilde': T_n_tilde, 'psi': psi_n, 'iter': i})


def calculate_tests(t, x, y, fair_direction, num_steps, learning_rate,\
     regularizer, c_distance, delta, B, iters):
    theta = [1, t]
    bias = claculate_bias(theta, x, y)
    classifier = hard_linear_classifier(theta, bias)

    
    T_n, T_n_tilde = lower_bound(x[:500, :], y[:500], theta=theta, bias=bias, fair_direction=fair_direction,\
         num_steps = num_steps, learning_rate=learning_rate, regularizer=regularizer, cpus=1)
    psi_n = faith_test(x, y, c_distance, classifier, delta = delta, cpus=1, B = B)
    ts =  {'t2': t, 'T': T_n, 'T-tilde': T_n_tilde, 'psi': psi_n, 'iter': iters}
    
    return ts

# tests = partial(calculate_tests, x = x, y = y, fair_direction = [0, 1], learning_rate = 2e-2,\
#      num_steps = 5, regularizer = 200, c_distance = c_distance, delta = 1e-5, i = i, B = 2)

# theta2 = [(0.4 * _ - 4) for _ in range(21)]

# if cpus > 1:
#     with mp.Pool(cpus) as pool:
#         test_stats = pool.map(tests, theta2)

# else:
#     test_stats = list(map(tests, theta2))

# with open(f'test_stats/ts{i}', 'w') as f:
#     json.dump(test_stats, f)

if __name__ == '__main__':
    i = int(float(sys.argv[1]))
    ITER = 100
    theta2 = [(0.4 * _ - 4) for _ in range(21)]

    pars = list(itertools.product(range(ITER), theta2))
    iters, t = pars[i]
    x, y = np.load(f'data/x_{iters}.npy'), np.load(f'data/y_{iters}.npy')

    tests = partial(calculate_tests, x = x, y = y, fair_direction = [0, 1], learning_rate = 2e-2,\
     num_steps = 500, regularizer = 200, c_distance = c_distance, delta = 1e-5, iters = iters, B = 1000)
    ts = tests(iters)
    print(str(ts))

    with open(f'test_stats/ts{iters}_{t}', 'w') as f:
        json.dump([ts], f)

    




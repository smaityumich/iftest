import numpy as np
import sys
from loss_linear import claculate_bias, hard_linear_classifier, lower_bound
from faith import faith_test
import json

i = int(float(sys.argv[1]))
cpus = int(float(sys.argv[2]))
theta2 = np.arange(-4, 4.1, step = 0.4)
x, y = np.load(f'data/x_{i}.npy'), np.load(f'data/y_{i}.npy')



def fair_distance(x1, x2):
    return np.abs(x1[0] - x2[0])

def c_distance(z1, z2, infty_equiv = 1000):
    x1, y1 = z1
    x2, y2 = z2
    if y1 != y2:
        return infty_equiv
    else:
        return fair_distance(x1, x2)


test_stats = []
for t in theta2:
    theta = [1, t]
    bias = claculate_bias(theta, x, y)
    classifier = hard_linear_classifier(theta, bias)

    
    T_n, T_n_tilde = lower_bound(x, y, theta=theta, bias=bias, fair_direction=[0, 1], num_steps = 500, learning_rate=2e-2, regularizer=200, cpus=cpus)
    psi_n = faith_test(x, y, c_distance, classifier, delta = 1e-5, cpus=cpus)
    test_stats.append({'t2': t, 'T': T_n, 'T-tilde': T_n_tilde, 'psi': psi_n, 'iter': i})

with open(f'test_stats/ts{i}', 'w') as f:
    json.dump(test_stats, f)




import numpy as np
import sys
np.random.seed(0)
def create_data(n = 1000, d1 = 4, d2 = 2, iterations = 100):
    for i in range(iterations):

        x0 = np.random.binomial(d1, 0.5, size = (n, 1))/d1 - 0.5
        x1 = np.random.binomial(d2, 0.5, size = (n, 1))/d2 - 0.5
        x = np.concatenate((x0, x1), axis=1)
        
        p = 1/(1 + np.exp(- 2 * x0)).reshape((-1,))
        y = np.random.binomial(1, p, size = (n, ))

        np.save(f'data/x_{i}.npy', x), np.save(f'data/y_{i}.npy', y.reshape((-1, )))


if __name__ == '__main__':
    n = int(float(sys.argv[1]))
    iterations = int(float(sys.argv[2]))
    create_data(n = n, iterations=iterations)

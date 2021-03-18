import numpy as np
import sys
np.random.seed(0)
def create_data(n = 1000, d = 4, iterations = 100):
    for i in range(iterations):

        # x1 = np.random.binomial(d, 0.5, size = (n, 1))
        # x2 = np.random.binomial(1, 0.5, size = (n, 1))
        # x = np.concatenate((x1, x2), axis = 1)

        # b = np.array([[1, 0]])
        # p = 1/(1 + np.exp(- x @ b.T + d/2)).reshape((-1,))
        # y = np.random.binomial(1, p, size = (n, ))

        y = np.random.binomial(1, 0.5, size = (n, 1))
        g = np.random.binomial(1, 0.05, size = (n, 1))
        x0 = np.random.binomial(4, 0.5, size = (n, 1))/8 * (2 * y - 1) * (2 * g - 1) + 1.5 * (2 * g - 1)
        x1 = np.random.binomial(4, 0.5, size = (n, 1))
        x = np.concatenate((x0, x1), axis = 1)


        np.save(f'data/x_{i}.npy', x), np.save(f'data/y_{i}.npy', y.reshape((-1, )))


if __name__ == '__main__':
    n = int(float(sys.argv[1]))
    iterations = int(float(sys.argv[2]))
    create_data(n = n, iterations=iterations)

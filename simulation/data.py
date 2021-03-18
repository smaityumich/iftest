import numpy as np
def create_data(n = 1000, d = 4, iterations = 100):
    for i in range(iterations):

        x1 = np.random.binomial(d, 0.5, size = (n, 1))
        x2 = np.random.binomial(1, 0.5, size = (n, 1))
        x = np.concatenate((x1, x2), axis = 1)

        b = np.array([[1, 0]])
        p = 1/(1 + np.exp(- x @ b.T + d/2)).reshape((-1,))
        y = np.random.binomial(1, p, size = (n, ))


        np.save(f'data/x_{i}.npy', x), np.save(f'data/y_{i}.npy', y)
import numpy as np
from faith import auditor_problem, faith_test
import itertools, sys



def load_data(iters):
    x, y = np.load(f'data/x_{iters}.npy'), np.load(f'data/y_{iters}.npy')
    y = y.reshape((-1, 1))
    return np.concatenate((x, y), axis = 1)


def c_distance(z1, z2, infty_equiv = 1000):

    if z1[0] == z2[0] and z1[2] == z2[2]:
        return 0
    else:
        return infty_equiv



def faith_lb(Z_n, t1, B = 20, n = 250):

    d1 = 4
    d2 = 2

    X0 = np.array(range(d1+1))/d1 - 0.5
    X1 = np.array(range(d2+1))/d2 - 0.5
    Y = range(2)


    K = (d1+1) * (d2+1) * 2
    Z = itertools.product(X0, X1, Y)
    Z = np.array(list(Z))
    # print(Z)
    p = np.zeros(shape=K)

    theta = np.array([2,t1])
    bias = 0



    for i in range(n):
        z = Z_n[i, :]
        ind = int(z[2] + (z[1] + 0.5) * d2  * 2 + (z[0] + 0.5) * d1 * 2 * d2) 
        p[ind] += 1
    p = p/n



    C = np.zeros(shape = (K, K))
    for i in range(K):
        for j in range(K):
            if i < j:
                C[i, j] = c_distance(Z[i, :], Z[j, :])
                C[j, i] = C[i, j]
    # print(C)
    l = np.zeros(shape = (K,))
    x = Z[:, :2]
    y_hat = (x @ theta.reshape((-1, 1)) + bias > 0).astype('float64')
    y_hat = y_hat.reshape((-1, ))
    y_unique = Z[:, 2]
    l = (y_hat - y_unique) ** 2


    ft = faith_test(p_n = p, C = C, l = l, delta = 1e-6, B = B, K = K, n = n, cpus=1)
    return ft

if __name__ == '__main__':
    t1 = float(sys.argv[1])
    Z_n = load_data(0)
    print(faith_lb(Z_n, t1))




    

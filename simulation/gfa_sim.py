import numpy as np
from loss_linear import lower_bound
import sys



def lbs_gfa(x, y,  t1, m = 100):
    
    lbs = lower_bound(x[:m, :], y[:m], theta = [2, t1], bias = 0, fair_direction=[1, 0],\
         regularizer=200, cpus=1, num_steps=20, learning_rate= 2e-2)
    return lbs


if __name__ == '__main__':
    t1 = float(sys.argv[1])
    iters = 0
    x, y = np.load(f'data/x_{iters}.npy'), np.load(f'data/y_{iters}.npy')
    print(lbs_gfa(x, y, t1, 100))



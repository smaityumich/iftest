import itertools
import sys
import os
import numpy as np

def part_summary(args):
    expt, i, lr = args
    np.random.seed(1)
    seeds = np.load('./seeds.npy')
    if expt == 'reduction':
        seed = seeds[i, 0]
        os.system(f'python3 ./{expt}/summary.py {seed} {lr}')
    elif expt == 'baseline_bal':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')
    elif expt == 'baseline':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')

    else:
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')


if __name__ == '__main__':
    
    expts = ['baseline', 'project', 'sensr', 'reduction'] 
    iteration = range(10)
    lrs =  [4e-3, 2e-3, 6e-3]

    a = list(itertools.product(expts, iteration, lrs))
    for b in a:
        part_summary(b)
        print('Done: ' + str(b))
    i = int(float(sys.argv[1]))
    






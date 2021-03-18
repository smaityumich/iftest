import json, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse




def join(ITER, file):
    summary = []
    for i in range(ITER):
        with open(file+str(i), 'r') as f:
            summary += json.load(f)

    return summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--iter', dest='iter', type=int, help='number of iters')
    parser.add_argument('--file', dest='file', type=str, help='iteration summaries; put filename without iteration number')
    parser.add_argument('--dest-file', dest='dfile', type=str, help='destination file')
    parser.add_argument('--clean-dir', dest='clean', type=int, help='put 1 if to delete all the files in directory')
    args = parser.parse_args()

    summary = join(args.iter, args.file)
    with open(args.dfile, 'w') as f:
        json.dump(summary, f)

    if args.clean:
        os.system('rm '+args.file+'*')
        


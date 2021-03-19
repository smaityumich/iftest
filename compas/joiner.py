import json, sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse




def join(file):
    summary = []
    for fname in glob.glob(file + '*'):
        with open(fname, 'r') as f:
            summary += json.load(f)

    return summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--file', dest='file', type=str, help='iteration summaries; put filename without iteration number')
    parser.add_argument('--dest-file', dest='dfile', type=str, help='destination file')
    parser.add_argument('--clean-dir', dest='clean', type=int, help='put 1 if to delete all the files in directory')
    args = parser.parse_args()

    summary = join(args.file)
    with open(args.dfile, 'w') as f:
        json.dump(summary, f)

    if args.clean:
        os.system('rm '+args.file+'*')



'''
import multiprocessing as mp
from pathlib import Path
import sys
import os
from multiprocessing import Pool
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)
from probability import set_customer_probability
import constants
def run_smth(i,j):
   # print("running", i)
    return (i+j)

def run(k):

    pool = Pool(mp.cpu_count())
    arr = []
   # pool = Pool(processes=3)
    argument = {}

    arr = pool.starmap(run_smth, [(i,i+1) for i in range(k)])
    return(arr)

if __name__ == '__main__':
    print(run(3))
'''
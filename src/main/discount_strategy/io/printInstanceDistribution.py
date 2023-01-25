from pathlib import Path
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import sys
from sys import argv
import os
#from collections import Counter
#import numpy as np
#import matplotlib.pyplot as plt
from time import process_time
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"io")
sys.path.insert(1, path_to_io)
import OCVRPParser
from print_functions import Painter

path_to_branchAndBound = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact", "bab")
sys.path.insert(2, path_to_branchAndBound)
from BranchAndBound import BranchAndBoundExact

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(3, path_to_exact)
from ring_star_without_TW import ring_star_deterministic_no_TW
from TSPSolver import TSPSolver

path_to_stochastic = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "stochastic")
sys.path.insert(4, path_to_stochastic)
from enumeration_scenarios import ScenarioEnumerationSolver
from sample_average import sampleAverageApproximation

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(5, path_to_util)
import constants

prefix="tag: "


def set_DistribDict(policy, TSPSolver, OCVRPInstance):
    n = OCVRPInstance.NR_CUST
    p_dev = OCVRPInstance.deviationProbability
    discount = OCVRPInstance.FLAT_RATE_SHIPPING_FEE
    distribDict = {}
    setMayVary = []
    for index, disc in enumerate(bin(policy)[::-1]):
        if disc == '1':
            setMayVary.append(index + 1)
    for v in range(0, len(setMayVary) + 1):
        givenDisc = (len(setMayVary) - v) * discount * (1 - p_dev)
        all_combinations = combinations(setMayVary, v)
        for combination in all_combinations:
            scenario = policy
            for offset in combination:
                mask = ~(1 << (offset - 1))
                scenario = scenario & mask
            # print("scenario", bin(scenario)[2:], TSPSolver.tspCost(scenario))
            distribDict[TSPSolver.tspCost(scenario) + givenDisc] = ((1 - p_dev) ** (len(setMayVary) - v) * p_dev ** v)
    return distribDict

if __name__ == "__main__":

    file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_TSPLIB_berlin52",
                                 "instance_berlin52_8_20_240_2_40.txt")


    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True, dpi=100)
    OCVRPInstance = OCVRPParser.parse(file_instance)
    TSPSolver = TSPSolver(instance=OCVRPInstance, solverType='Gurobi')

    policyBAB = 131143
    distribDictBAB = set_DistribDict(policyBAB, TSPSolver, OCVRPInstance)


    sns.distplot(list(distribDictBAB.keys()), hist_kws={"weights": list(distribDictBAB.values())},
                 ax= axes[0], axlabel='S-OCTSP', kde_kws={'bw':100})

    policyRS = 135807
    distribDictRS = set_DistribDict(policyRS, TSPSolver, OCVRPInstance)
    sns.distplot(list(distribDictRS.keys()), hist_kws={"weights": list(distribDictRS.values())},
                 ax= axes[1], axlabel='D-OCTSP', kde_kws={'bw':100})

    policyNODISC = 0
    distribDictNODISC = {TSPSolver.tspCost(0):0.5, TSPSolver.tspCost(0)+10:0.5 }
    sns.distplot(list(distribDictNODISC.keys()), hist_kws={"weights": list(distribDictNODISC.values())},
                 ax=axes[2], axlabel='NODISC', kde_kws = {'bw': 1})

    #print(sorted(distribDict))
    #plt.hist(list(distribDict.keys()), weights=list(distribDict.values()))
    #plt.bar(list(distribDict.keys()), distribDict.values(), color='g')
    plt.ylabel('Probability')
    plt.show()




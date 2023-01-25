from pathlib import Path
import csv
import sys
from sys import argv
import random
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
from BranchAndBoundExact import BranchAndBoundExact

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(3, path_to_exact)
from ring_star_without_TW import ring_star_deterministic_no_TW

path_to_stochastic = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "stochastic")
sys.path.insert(4, path_to_stochastic)
from enumeration_scenarios import ScenarioEnumerationSolver
from sample_average import sampleAverageApproximation

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(5, path_to_util)
import constants

path_to_exact =  os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(6, path_to_exact)
from TSPSolver import TSPSolver

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[3]), "algorithms", "exact")
sys.path.insert(2, path_to_exact)
from TSPSolver import TSPSolver
from ring_star_without_TW import ring_star_deterministic_no_TW

prefix="tag: "

def set_scenario(sample,n):
    scenario = 2**n - 1
    #print(sorted(sample))
    #print("scenario", scenario, bin(scenario))
    for i in sample:
        #print(i)
        scenario -= 2**(i-1)
    #print( bin(scenario))
    return scenario

if __name__ == "__main__":


    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_TSPLIB_heuristic",
    #                             "instance_berlin52_4_50_210_3_40.txt")
    file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_TSPLIB_heuristic",
                                 "instance_berlin52_1_20_30_1_20.txt")



    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    painter = Painter()
    painter.printVertex(OCVRPInstance)

    rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)
    print(bin(rsPolicyID)[2:], "RS cost", rsValue)


    n = OCVRPInstance.NR_CUST
    p_dev = OCVRPInstance.deviationProbability
    discount = OCVRPInstance.FLAT_RATE_SHIPPING_FEE


    TSPSolver = TSPSolver(instance=OCVRPInstance, solverType='Gurobi')
    #TSPSolver.tspCost(scenario)
    #sample - customers that should be visited

    # customer form star:
    addedCust = 18
    #all customers from star:
    starCusts =[i for i in range(1,17) ]

    #for this specific instance:
    must_be_visited = [19,20,17]
    #starCusts = [i for i in range(1, addedCust)] + [i for i in range(addedCust + 1, n + 1)]
    for iter in range(10):
        sample = random.sample(starCusts, k = iter)
        for size in range(1,iter):
            subsample = random.sample(sample, k=size)

            scenario_without = set_scenario(sample+must_be_visited,n)
            subscenario_without = set_scenario(subsample+must_be_visited, n)

            scenario_with = set_scenario(sample+[addedCust] + must_be_visited, n)
            subscenario_with = set_scenario(subsample+[addedCust] + must_be_visited, n)

            insertion_cost_sample = TSPSolver.tspCost(scenario_with)-TSPSolver.tspCost(scenario_without)
            insertion_cost_subsample = TSPSolver.tspCost(subscenario_with) - TSPSolver.tspCost(subscenario_without)

            if(insertion_cost_sample > insertion_cost_subsample):
                print("")
                print("visited customers in sample: ", sorted(sample+must_be_visited),
                      "\nvistied customers in subsample: ", sorted(subsample+must_be_visited))
                print("insert in sample: ",insertion_cost_sample,"\ninsert in subsample: ", insertion_cost_subsample)


    # for RS:
    scenario_without = set_scenario(must_be_visited, n)
    scenario_with = set_scenario([addedCust] + must_be_visited, n)
    insertion_cost_rs = TSPSolver.tspCost(scenario_with) - TSPSolver.tspCost(scenario_without)

    print("")
    print("For RS:")
    print("insert in rs: ", insertion_cost_rs)
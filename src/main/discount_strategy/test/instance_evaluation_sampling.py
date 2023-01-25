from pathlib import Path
import csv
import sys
from sys import argv
import os
from time import process_time

path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"io")
sys.path.insert(1, path_to_io)
import OCVRPParser
path_to_enumeration= os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact","enumeration")
sys.path.insert(2, path_to_enumeration)
from enumeration_scenarios import ScenarioEnumerationSolver

path_to_exact= os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(3, path_to_exact)
from ring_star_without_TW import ring_star_deterministic_no_TW

path_to_heuristic= os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "heuristic")
sys.path.insert(4, path_to_heuristic)
from sample_average import sampleAverageApproximation_PoissonBinomial
from sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from sample_average import one_policy_cost_estimation
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(5, path_to_util)
import constants

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))


prefix="tag: "
if __name__ == "__main__":

    instance = str(sys.argv[-1])
    file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data","instances_Santini_adapted", instance.split("+")[1]+".txt")

    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data","instances_TSPLIB_2","instance_berlin52_1_20_10_40.txt")

    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_TSPLIB_berlin52",
    #                             "instance_berlin52_1_20_80_1_20.txt")
    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_Santini_adapted",
    #                             "sz-9-prob_type-direct_dist-prob-0.25-fee_type-direct_prob-fee-3.1.txt")
    # instance = '255'


    solverType = 'Gurobi'
    #solverType = 'Concorde'

    OCVRPInstance = OCVRPParser.parse(file_instance)

    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)

    if 2**OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
        estimation_rs = one_policy_cost_estimation(instance = OCVRPInstance, policy = int(instance.split("+")[0]), solverType =solverType)

    else:
        estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample(instance = OCVRPInstance, policy = int(instance.split("+")[0]), solverType = solverType)



    print(prefix, "policy_ID: ",  instance.split()[0])
    print(prefix+"estimation of RS with enumeration: ", estimation_rs)







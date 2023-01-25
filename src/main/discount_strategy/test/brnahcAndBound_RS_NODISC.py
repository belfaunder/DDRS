from pathlib import Path
import csv
import sys
from sys import argv
import os
from time import process_time


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f})".format(int(hours), int(minutes), seconds))


path_to_heuristic = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "heuristic")
sys.path.insert(2, path_to_heuristic)
from BAB_sampling_heuristic import BABHeuristic
from sample_average import sampleAverageApproximation_PoissonBinomial
from sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from sample_average import one_policy_cost_estimation

path_to_branchAndBound = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact", "bab")
sys.path.insert(3, path_to_branchAndBound)
from BAB_exact import BABExact

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "util")
sys.path.insert(5, path_to_util)
import constants

path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "io")
sys.path.insert(1, path_to_io)
import OCVRPParser
from print_functions import Painter
from ring_star_without_TW import ring_star_deterministic_no_TW

prefix = "tag: "

if __name__ == "__main__":
    # solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix, "solverType: ", solverType)
    print(prefix, "TIME_LIMIT:", constants.TIME_LIMIT)

    file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "i_VRPDO_const_density",
                                 str(sys.argv[-1]) + ".txt")
    # file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "i_7_types_nr_cust",
    #                             "eil76_size_12_phome_0.2_ppup_0.2_discsize_6_1.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)

    start_time = process_time()
    bab = BABExact(instance=OCVRPInstance, solverType=solverType)
    babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()

    # babSampling = BABHeuristic(instance=OCVRPInstance, solverType =solverType)
    # babPolicy=  babSampling.runBranchAndBound()

    if 2 ** OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance=OCVRPInstance, policy=babPolicy, solverType=solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance, policy=babPolicy,
                                                                            solverType=solverType)

    print(prefix, 'Estimated_BAB_cost:', estimation_bab)

    rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)
    if 2 ** OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
        estimation_rs = one_policy_cost_estimation(instance=OCVRPInstance, policy=rsPolicyID, solverType=solverType)
        estimation_nodisc = one_policy_cost_estimation(instance=OCVRPInstance, policy=0, solverType=solverType)
        estimation_uni = one_policy_cost_estimation(instance=OCVRPInstance, policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                    solverType=solverType)
    else:
        if rsPolicyID == babPolicy:
            estimation_rs = estimation_bab
        else:
            estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance,
                                                                               policy=rsPolicyID, solverType=solverType)
        if babPolicy == 0:
            estimation_nodisc = estimation_bab
        else:
            estimation_nodisc = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance, policy=0,
                                                                                   solverType=solverType)
        estimation_uni = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance,
                                                                            policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                                            solverType=solverType)

    print(prefix, 'RS_policy:', rsPolicyID)
    print(prefix, 'Estimated_RS_cost:', estimation_rs)
    print(prefix, 'Estimated_NODISC_cost:', estimation_nodisc)
    print(prefix, 'Estimated_UNIFORM_cost:', estimation_uni)



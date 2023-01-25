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

    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data","disc_size_sensitivity", str(sys.argv[-1])+".txt")
    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data","instances_TSPLIB_2","instance_berlin52_1_20_10_40.txt")

    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_TSPLIB_berlin52",
    #                             "instance_berlin52_1_20_80_1_20.txt")
    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_3_segments",
    #                             "temp_size_10.txt")

    file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "nr_cust_variation",
                                 "berlin52_size_14_phome_0.0_ppup_0.4_discsize_270.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)

    OCVRPInstance.calculateInsertionBounds()
    #print(OCVRPInstance)
    rsPolicy, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)

    estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, rsPolicy)

    print("estimation_rs", estimation_rs)

    estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, 0)

    print("estimation_bab", estimation_bab)


    if 2**OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
        estimation_rs = one_policy_cost_estimation(OCVRPInstance, rsPolicy)
        estimation_nodisc = one_policy_cost_estimation(OCVRPInstance, 0)
        estimation_uniform_disc = one_policy_cost_estimation(OCVRPInstance, 2**OCVRPInstance.NR_CUST-1)

    else:
        estimation_rs  = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, rsPolicy)
        estimation_nodisc = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, 0)
        estimation_uniform_disc = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, 2 ** OCVRPInstance.NR_CUST - 1)


    print(prefix+ "policy_rs: ", rsPolicy)
    print(prefix+ "policy_rs_ID: ", bin(rsPolicy)[2:])
    print(prefix+ "RS_value_expected value: ", estimation_rs)
    print(prefix+ "no_discount value : ", estimation_nodisc)
    print(prefix+ "uniform_discount value : ", estimation_uniform_disc)
    #estimation_rs = one_policy_cost_estimation(OCVRPInstance, 4159)
    #print(prefix, "RS_value (expected cost): ", estimation_rs)

    #print("")
    #EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance)
    #EnumerationSolver.exactPolicyByEnumeration(True)





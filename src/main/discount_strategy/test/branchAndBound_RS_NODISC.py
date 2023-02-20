from pathlib import Path
import sys
from sys import argv
import os
from time import process_time
import cProfile
import pstats
from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.io import print_functions
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW
#prefix="tag: "
prefix=constants.PREFIX
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

prefix = "tag: "

if __name__ == "__main__":
    solverType = 'Gurobi'

    print(prefix, "Exact BAB")
    print(prefix, "solverType: ", solverType)
    print(prefix, "TIME_LIMIT:", constants.TIME_LIMIT)
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_2segm_manyPUP_30", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30",
                                     "VRPDO_size_30_phome_0_ppup_0.0_incrate_0.03_nrpup3_0.txt")
        # file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP",
        #                             "VRPDO_size_10_phome_0.2_ppup_0.0_incrate_0.03_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)

    bab = BABExact(instance=OCVRPInstance, solverType=solverType)
    babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()

    if 2 ** bitCount(babPolicy) < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance=OCVRPInstance, policy=babPolicy, solverType=solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance,
                                                                            policy=babPolicy, solverType=solverType)
    print(prefix+'Estimated_BAB_cost:', estimation_bab)



    if rsPolicyID == babPolicy:
        estimation_rs = estimation_bab
    else:
        if 2 ** bitCount(rsPolicyID)  < constants.SAMPLE_SIZE:
            estimation_rs = one_policy_cost_estimation(instance=OCVRPInstance, policy=rsPolicyID, solverType=solverType)
        else:
            estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance,
                                                                               policy=rsPolicyID, solverType=solverType)

    if babPolicy == 0:
        estimation_nodisc = estimation_bab
    else:
        estimation_nodisc = one_policy_cost_estimation(instance=OCVRPInstance, policy=0, solverType=solverType)

    if babPolicy == 2 ** OCVRPInstance.NR_CUST - 1:
        estimation_uni = estimation_bab
    else:
        if 2  ** OCVRPInstance.NR_CUST< constants.SAMPLE_SIZE:
            estimation_uni = one_policy_cost_estimation(instance=OCVRPInstance, policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                    solverType=solverType)
        else:
            estimation_uni = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance,
                                                                            policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                                            solverType=solverType)
    print(prefix+ 'RS_policy:', rsPolicyID)
    print(prefix+ 'Estimated_RS_cost:', estimation_rs)
    print(prefix+ 'Estimated_NODISC_cost:', estimation_nodisc)
    print(prefix+ 'Estimated_UNIFORM_cost:', estimation_uni)



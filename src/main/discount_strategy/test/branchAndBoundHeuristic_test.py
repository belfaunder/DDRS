from pathlib import Path
from sys import argv
import os
from time import process_time
import sys
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.heuristic.remote_customers import BABHeuristic
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW
from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.io import OCVRPParser
path_to_data = constants.PATH_TO_DATA
prefix="tag: "
from src.main.discount_strategy.io.print_functions import Painter

if __name__ == "__main__":

    #solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix, "Heuristic Sampling")
    print(prefix, "solverType: ", solverType)
    print(prefix, "HEURISTIC_SAMPLE_SIZE: ", constants.HEURISTIC_SAMPLE_SIZE)
    print(prefix, "HEURISTIC_TIME_LIMIT:", constants.HEURISTIC_TIME_LIMIT)
    print(prefix, "NEIGHBOURHOOD_HEURISTIC: " ,constants.NEIGHBOURHOOD_HEURISTIC)
    print(prefix, "EPS_PRECISION_HEURISTIC: ", constants.EPSILON_H)

    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_2segm_manyPUP_large", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_large",
                                     "VRPDO_size_10_phome_0.4_ppup_0.0_incrate_0.06_nrpup3_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    start_time = process_time()
    babSampling = BABHeuristic(instance=OCVRPInstance, solverType =solverType)
    babPolicy=  babSampling.runBranchAndBound()

    if 2 ** bitCount(babPolicy) < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance=OCVRPInstance, policy=babPolicy, solverType=solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                            policy=babPolicy, solverType=solverType)

    print(prefix, 'Estimated_BAB_cost:',estimation_bab )

    rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)

    if rsPolicyID == babPolicy:
        estimation_rs = estimation_bab
    else:
        if 2 ** bitCount(rsPolicyID) < constants.SAMPLE_SIZE:
            estimation_rs = one_policy_cost_estimation(instance=OCVRPInstance, policy=rsPolicyID, solverType=solverType)
        else:
            estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                               policy=rsPolicyID, solverType=solverType)

    if babPolicy == 0:
        estimation_nodisc = estimation_bab
    else:
        estimation_nodisc = one_policy_cost_estimation(instance=OCVRPInstance, policy=0, solverType=solverType)

    if babPolicy == 2 ** OCVRPInstance.NR_CUST - 1:
        estimation_uni = estimation_bab
    else:
        if 2 ** OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
            estimation_uni = one_policy_cost_estimation(instance=OCVRPInstance, policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                        solverType=solverType)
        else:
            estimation_uni = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                                policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                                                solverType=solverType)
    print(prefix + 'RS_policy:', rsPolicyID)
    print(prefix + 'Estimated_RS_cost:', estimation_rs)
    print(prefix + 'Estimated_NODISC_cost:', estimation_nodisc)
    print(prefix + 'Estimated_UNIFORM_cost:', estimation_uni)



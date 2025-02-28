# Code to run the BAB Heuristic policy and compare it with the cost of alternative policies

import os
from pathlib import Path
import sys
from time import process_time
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.heuristic.BAB_sampling_heuristic import BABHeuristic
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.io import OCVRPParser

path_to_data = constants.PATH_TO_DATA
prefix=constants.PREFIX

if __name__ == "__main__":
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix+ "Heuristic Sampling")
    print(prefix+ "solverType: ", solverType)
    print(prefix+ "HEURISTIC_SAMPLE_SIZE: ", constants.HEURISTIC_SAMPLE_SIZE)
    print(prefix+ "HEURISTIC_TIME_LIMIT:", constants.HEURISTIC_TIME_LIMIT)
    print(prefix+ "NEIGHBOURHOOD_HEURISTIC: " ,constants.NEIGHBOURHOOD_HEURISTIC)
    print(prefix+ "EPS_PRECISION_HEURISTIC: ", constants.EPSILON_H)

    # In case the scipt is run from the command line (mip.sh in directory scripts), the last argument is the instance name
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "i_DDRS", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(constants.PATH_TO_DATA, "test_instance.txt")

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

    print(prefix+ 'Estimated_BAB_cost:',[cost/constants.SCALING_FACTOR for cost in  estimation_bab])
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
        estimation_all = estimation_bab
    else:
        if 2 ** OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
            estimation_all = one_policy_cost_estimation(instance=OCVRPInstance, policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                        solverType=solverType)
        else:
            estimation_all = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                                policy=2 ** OCVRPInstance.NR_CUST - 1,
                                                                                solverType=solverType)
    print(prefix + 'RS_policy:', rsPolicyID)
    print(prefix + 'Estimated_RS_cost:', [cost/constants.SCALING_FACTOR for cost in estimation_rs])
    print(prefix + 'Estimated_NODISC_cost:', [cost/constants.SCALING_FACTOR for cost in estimation_nodisc])
    print(prefix + 'Estimated_ALL_cost:', [cost/constants.SCALING_FACTOR for cost in estimation_all])
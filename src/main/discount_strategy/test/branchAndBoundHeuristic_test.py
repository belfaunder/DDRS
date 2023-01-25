from pathlib import Path
from sys import argv
import os
from time import process_time
import sys
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

from src.main.discount_strategy.algorithms.heuristic.BAB_sampling_heuristic import BABHeuristic
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation

from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.io import OCVRPParser
path_to_data = constants.PATH_TO_DATA
prefix="tag: "

if __name__ == "__main__":
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix,"Heuristic Sampling")
    print(prefix,"solverType: ", solverType)
    print(prefix,"HEURISTIC_SAMPLE_SIZE: ", constants.HEURISTIC_SAMPLE_SIZE)
    print(prefix, "HEURISTIC_TIME_LIMIT:", constants.HEURISTIC_TIME_LIMIT)
    print(prefix,"NEIGHBOURHOOD_HEURISTIC: " ,constants.NEIGHBOURHOOD_HEURISTIC)
    print(prefix, "EPS_PRECISION_HEURISTIC: ", constants.EPSILON_H)

    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_discount_proportional_2segm", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional",
                                     "VRPDO_size_12_phome_0.2_ppup_0.0_incrate_0.06_0.txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    #estimation_nodisc = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance, policy=0,
    #                                                                       solverType=solverType)
    start_time = process_time()
    # bab = BABExact(instance=OCVRPInstance, solverType = solverType)
    # babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()

    babSampling = BABHeuristic(instance=OCVRPInstance, solverType =solverType)
    babPolicy=  babSampling.runBranchAndBound()

    if 2**OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance = OCVRPInstance, policy = babPolicy, solverType =solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample(instance = OCVRPInstance, policy = babPolicy, solverType = solverType)

    print(prefix, 'Estimated_BAB_cost:',estimation_bab )

    # rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)
    # if 2**OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
    #     estimation_rs = one_policy_cost_estimation(instance = OCVRPInstance, policy = rsPolicyID, solverType =solverType)
    #     estimation_nodisc = one_policy_cost_estimation(instance=OCVRPInstance, policy=0, solverType=solverType)
    #     estimation_uni = one_policy_cost_estimation(instance=OCVRPInstance, policy=2 ** OCVRPInstance.NR_CUST - 1, solverType=solverType)
    # else:
    #     if rsPolicyID==babPolicy:
    #         estimation_rs = estimation_bab
    #     else:
    #         estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample(instance = OCVRPInstance, policy = rsPolicyID, solverType = solverType)
    #     if babPolicy==0:
    #         estimation_nodisc = estimation_bab
    #     else:
    #         estimation_nodisc = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance, policy=0,
    #                                                                        solverType=solverType)
    #     estimation_uni = sampleAverageApproximation_PoissonBinomial_1sample(instance=OCVRPInstance, policy=2 ** OCVRPInstance.NR_CUST - 1,
    #                                                                        solverType=solverType)
    #
    #
    # print(prefix, 'RS_policy:', rsPolicyID)
    # print(prefix, 'Estimated_RS_cost:', estimation_rs)
    # print(prefix, 'Estimated_NODISC_cost:', estimation_nodisc)
    # print(prefix, 'Estimated_UNIFORM_cost:', estimation_uni)



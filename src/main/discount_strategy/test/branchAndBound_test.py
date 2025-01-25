# code to run exact branch and bound algorithm

import os
from pathlib import Path
import sys
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.io import OCVRPParser
prefix=constants.PREFIX

if __name__ == "__main__":
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix, "Exact BAB")
    print(prefix,"solverType: ", solverType)
    print(prefix, "TIME_LIMIT:", constants.TIME_LIMIT)

    # In case the scipt is run from the command line (mip.sh in directory scripts), the last argument is the instance name
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "i_DDRS", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(constants.PATH_TO_DATA, "test_instance.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)

    # run the B&B
    bab = BABExact(instance=OCVRPInstance, solverType = solverType)
    babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()

    print(prefix,"pruned_by_cliques_nonleaf:", bab.pruned_cliques_nonleaf)
    print(prefix,"pruned_by_cliques_leaf:", bab.pruned_cliques_leaf)
    print(prefix, "pruned_by_insertionCost_nonleaf:", bab.pruned_insertionCost_nonleaf)
    print(prefix, "pruned_by_insertionCost_leaf:", bab.pruned_insertionCost_leaf)
    print(prefix, "pruned_by_bounds_nonleaf:", bab.pruned_bounds_nonleaf)
    print(prefix, "pruned_by_bounds:", bab.pruned_by_bounds_leaf )

    # Estimate the cost of the best policy through sampling.
    # We have to do this because the B&B can conlude that the policy is optimal when the gap is too large,
    # in this case, we need a more precise estimation of the policy cost.
    # For small instances, the policy cost is computed exactly by enumerating all possible scenarios.
    # For larger instances, we take the average cost over constants.SAMPLE_SIZE number of random scenarios.
    if 2 ** bitCount(babPolicy) < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance=OCVRPInstance, policy=babPolicy, solverType=solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                                  policy=babPolicy, solverType=solverType)
    print(prefix, 'Estimated_BAB_cost:', [cost/constants.SCALING_FACTOR for cost in estimation_bab])

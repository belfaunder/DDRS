from pathlib import Path
import sys
from sys import argv
import os
from src.main.discount_strategy.util import probability
import seaborn as sns
from time import process_time
import cProfile
import pstats
from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
path_to_images = constants.PATH_TO_IMAGES
from src.main.discount_strategy.util.bit_operations import bitCount
import pickle
import numpy as np
from matplotlib.ticker import PercentFormatter
from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
#from src.main.discount_strategy.algorithms.exact.enumeration.enumeration_scenarios_2_segm import ScenarioEnumerationSolver
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.io.print_functions import Painter
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.io import print_functions
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW
prefix=constants.PREFIX
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

import matplotlib.pyplot as plt
if __name__ == "__main__":
#def test():
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix, "Exact BAB")
    print(prefix,"solverType: ", solverType)
    print(prefix, "TIME_LIMIT:", constants.TIME_LIMIT)
    # fig, axes = plt.subplots(3, 1, figsize=(4, 12), sharex = True)
    # plt.xscale('log')
    # iter =0
    # for prob in [0.1, 0.4, 0.7]:
    #     file_instance = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPup_classes",
    #                                  "VRPDO_size_15_phome_" + str(prob)+"_ppup_0.0_incrate_0.06_nrpup3_0.txt")
    #
    #     OCVRPInstance = OCVRPParser.parse(file_instance)
    #     OCVRPInstance.calculateInsertionBounds()
    #     bab = BABExact(instance=OCVRPInstance, solverType = solverType)
    #     babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()
    #
    #     n = OCVRPInstance.NR_CUST
    #     # temp_prob = []
    #     # for scenario in range(2**n):
    #     #     if not (~babPolicy & scenario):
    #     #         temp_prob.append(probability.scenarioProb_2segm(scenario, babPolicy, n, n, OCVRPInstance.p_pup_delta))
    #     # sns.displot(temp_prob, bins=50)
    #     # plt.xlabel('Probability best policy')
    #     # plt.yscale('log')
    #     # plt.show()
    #     policy_all = 2**n - 1
    #     policy_all = babPolicy
    #     temp_prob = []
    #     for scenario in range(2 ** n):
    #         if not (~policy_all & scenario):
    #             temp_prob.append(probability.scenarioProb_2segm(scenario, policy_all, n, n, OCVRPInstance.p_pup_delta))
    #
    #
    #     axes[iter].hist(temp_prob, bins=[0.0000001, 0.000001, 0.00001, 0.0001,0.001, 0.01, 0.1,1])
    #     axes[iter].set_ylabel("delta " + str(round(1-prob,1)))
    #     iter += 1
    # plt.xlabel('Probability policy_best')
    # plt.savefig(os.path.join(path_to_images, 'distribution_probability_scenarios_best.png'), transparent=False, bbox_inches='tight')
    # plt.show()

    # N = 21
    # x = list(range(N))
    # y = [(x1- N)/(2*x1 - N) for x1 in x]
    # plt.plot(x, y)
    # plt.show()
    N = 21
    N0 = 20
    x = list(i/20 for i in range(0,20))

    y = [(1-x1)**N0 * (x1)**(N-N0) for x1 in x]
    plt.plot(x, y)
    plt.show()

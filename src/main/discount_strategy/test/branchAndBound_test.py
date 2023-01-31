from pathlib import Path
import sys
from sys import argv
import os
from time import process_time

from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
from src.main.discount_strategy.util.bit_operations import bitCount
#import pickle
from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
from src.main.discount_strategy.algorithms.exact.enumeration.enumeration_scenarios_2_segm import ScenarioEnumerationSolver

from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation

from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.io import print_functions

#solverType = 'Gurobi'
#solverType = 'Concorde'
prefix="tag: "

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

import matplotlib.pyplot as plt
if __name__ == "__main__":
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    print(prefix, "Exact BAB")
    print(prefix,"solverType: ", solverType)
    print(prefix, "TIME_LIMIT:", constants.TIME_LIMIT)
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_discount_proportional_2segm_manyPUP", str(sys.argv[-1])+".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm",
                                     "VRPDO_size_10_phome_0.2_ppup_0.0_incrate_0.06_9.txt")
        #file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP",
        #                             "VRPDO_size_10_phome_0.2_ppup_0.0_incrate_0.03_0.txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)

    #painter = Painter()
    #painter.printVertexDisc(OCVRPInstance, 0)
    # rs_policy, rs_cost = ring_star_deterministic_no_TW(OCVRPInstance, 0)
    # print(rs_cost/15)
    # print((sum(OCVRPInstance.shipping_fee[1:])/len(OCVRPInstance.shipping_fee[1:]))*(6/5)/ (rs_cost/15))
    #"VRPDO_size_25_phome_0.2_ppup_0.2_incrate_0.4_2.txt", 122881)
    #estimation_rs = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, rsPolicy)
    #print("estimation_rs", estimation_rs)
    #estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample(OCVRPInstance, 0)

    #print("estimation_bab", estimation_bab)
    #print()
    #start_time = process_time()
    #EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance)
    #EnumerationSolver.exactPolicyByEnumeration(True)
    #print(prefix, 'Time_enumeration ', process_time()-start_time)
    #print("\n")

    bab = BABExact(instance=OCVRPInstance, solverType = solverType)
    babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()
    # mainDirStorage =  os.path.join(path_to_data,"output")
    # convergence = os.path.join(mainDirStorage, 'convergence.txt')
    # with open(convergence, 'wb') as file:
    #     pickle.dump(time, file)
    #     pickle.dump(lbPrint, file)
    #     pickle.dump(ubPrint, file)
    #painter.printConvergence(time, lbPrint, ubPrint, bab_obj)

    if 2**OCVRPInstance.NR_CUST < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance = OCVRPInstance, policy = babPolicy, solverType = solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample(instance = OCVRPInstance, policy = babPolicy, solverType = solverType)

    print(prefix, 'Estimated_BAB_cost:',estimation_bab )

    #EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType=solverType)
    #EnumerationSolver.exactPolicyByEnumeration_withoutGurobi_2segm()
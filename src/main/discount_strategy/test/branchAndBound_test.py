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
import pickle
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
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_2segm_manyPup_classes", str(sys.argv[-1])+".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPup_classes",
                                     "VRPDO_size_12_phome_0.1_ppup_0.0_incrate_0.06_nrpup3_3.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    bab = BABExact(instance=OCVRPInstance, solverType = solverType)
    #rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)
    # #babPolicy = rsPolicyID
    # babPolicy = 25865
    babPolicy, time, lbPrint, ubPrint = bab.runBranchAndBound()
    # painter = Painter()
    # OCVRPInstance_clustered = OCVRPParser.parse(file_instance)
    # OCVRPInstance_random = OCVRPParser.parse(os.path.join(path_to_data, "data", "solomon","artificial", "solomonR101_discount_0.3.txt"))
    # painter.printVertexDiscTemp(OCVRPInstance_clustered,21504, OCVRPInstance_random, 25865)
    # for pup in OCVRPInstance.pups:
    #     print(pup.id, pup.closest_cust_id)
    #start_time = process_time()
    #rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)
    #EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance)
    #EnumerationSolver.exactPolicyByEnumeration(True)
    #print(prefix, 'Time_enumeration ', process_time()-start_time)

    print(prefix,"pruned_by_cliques_nonleaf:", bab.pruned_cliques_nonleaf)
    print(prefix,"pruned_by_cliques_leaf:", bab.pruned_cliques_leaf)
    print(prefix,"pruned_by_rs_nonleaf:", bab.pruned_rs_nonleaf)
    print(prefix,"pruned_by_rs_leaf:", bab.pruned_rs_leaf)
    print(prefix, "pruned_by_insertionCost_nonleaf:", bab.pruned_insertionCost_nonleaf)
    print(prefix, "pruned_by_insertionCost_leaf:", bab.pruned_insertionCost_leaf)
    print(prefix, "pruned_by_bounds_nonleaf:", bab.pruned_bounds_nonleaf)
    print(prefix, "pruned_by_bounds:", bab.nrNodes - bab.pruned_branching - bab.pruned_cliques_leaf - bab.pruned_cliques_nonleaf - bab.pruned_rs_leaf -\
          bab.pruned_rs_nonleaf-  bab.pruned_insertionCost_nonleaf - bab.pruned_insertionCost_leaf -  bab.pruned_bounds_nonleaf)

    # mainDirStorage =  os.path.join(path_to_data,"output")
    # convergence = os.path.join(mainDirStorage, 'convergence.txt')
    #
    # with open(convergence, 'wb') as file:htop
    #    pickle.dump(time, file)
    #    pickle.dump(lbPrint, file)
    #    pickle.dump(ubPrint, file)
    #
    # with open(convergence, "rb") as file:
    #     time = pickle.load(file)
    #     lbPrint = pickle.load(file)
    #     ubPrint = pickle.load(file)
    #Painter.printConvergence(OCVRPInstance, time, lbPrint, ubPrint, ubPrint[-1])
    if 2**bitCount(babPolicy) < constants.SAMPLE_SIZE:
        estimation_bab = one_policy_cost_estimation(instance = OCVRPInstance, policy = babPolicy, solverType = solverType)
    else:
        estimation_bab = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance = OCVRPInstance,
                                                                            policy = babPolicy, solverType = solverType)
    print(prefix, 'Estimated_BAB_cost:',estimation_bab )

    # EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType=solverType)
    # EnumerationSolver.exactPolicyByEnumeration_withoutGurobi_2segm()

# prof = cProfile.Profile()

# if __name__ == '__main__':
#     arr = []
#     prof.enable()
#     test()
#     prof.disable()
#     #prof.print_stats()
#     #prof.dump_stats("main_func.prof")
#     p = pstats.Stats(prof)
#     p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)

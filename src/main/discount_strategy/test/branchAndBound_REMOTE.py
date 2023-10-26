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
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.algorithms.heuristic.remote_customers import policy_remote_customers,policy_remote_customers2,policy_remote_rs, policy_insights_Nevin

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
    folder_large = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPup_classes")

    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_2segm_manyPup_classes", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPup_classes",
                                     "VRPDO_size_19_phome_0.4_ppup_0.0_incrate_0.06_nrpup3_4.txt")
        #file_instance = os.path.join(folder_large,"VRPDO_size_16_phome_0.1_ppup_0.0_incrate_0.06_nrpup3_1.txt")
        # file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP",
        #                             "VRPDO_size_10_phome_0.2_ppup_0.0_incrate_0.03_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    remote_policy_ID = policy_remote_customers(OCVRPInstance)
    print(bin(remote_policy_ID))




    if 2 ** bitCount(remote_policy_ID) < constants.SAMPLE_SIZE:
        estimation_remote = one_policy_cost_estimation(instance=OCVRPInstance, policy=remote_policy_ID, solverType=solverType)
    else:
        estimation_remote = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                            policy=remote_policy_ID, solverType=solverType)

    # print(bin(39388))
    # estimation_remote = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
    #                                                                              policy=39388,
    #                                                                              solverType=solverType)

    print(prefix+ 'Remote_policy:', remote_policy_ID)
    print(prefix+ 'Estimated_REMOTE_cost:', estimation_remote)
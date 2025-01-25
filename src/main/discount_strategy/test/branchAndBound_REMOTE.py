# Code to construct an easy benchmark policy, based on the remoteness of the customers


import os
from pathlib import Path
import sys
from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.algorithms.heuristic.remote_customers import policy_remote_customers

prefix=constants.PREFIX

if __name__ == "__main__":
    solverType = 'Gurobi'
    print(prefix, "Exact BAB")
    print(prefix, "solverType: ", solverType)
    print(prefix, "TIME_LIMIT:", constants.TIME_LIMIT)

    # In case the scipt is run from the command line (mip.sh in directory scripts), the last argument is the instance name
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "i_DDRS", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(constants.PATH_TO_DATA, "test_instance.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    remote_policy_ID = policy_remote_customers(OCVRPInstance)

    if 2 ** bitCount(remote_policy_ID) < constants.SAMPLE_SIZE:
        estimation_remote = one_policy_cost_estimation(instance=OCVRPInstance, policy=remote_policy_ID, solverType=solverType)
    else:
        estimation_remote = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=OCVRPInstance,
                                                                            policy=remote_policy_ID, solverType=solverType)
    print(prefix+ 'Remote_policy:', bin(remote_policy_ID)[2:].zfill(OCVRPInstance.NR_CUST))
    print(prefix+ 'Remote_policy_ID:', remote_policy_ID)
    print(prefix+ 'Estimated_REMOTE_cost:', estimation_remote)
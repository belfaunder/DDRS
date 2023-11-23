from pathlib import Path
import csv
import sys
from sys import argv
import os
from time import process_time
from src.main.discount_strategy.io import OCVRPParser
import cProfile
import pstats

from src.main.discount_strategy.algorithms.exact.enumeration.enumeration_scenarios_2_segm import ScenarioEnumerationSolver
from src.main.discount_strategy.util import constants


prefix="tag: "
path_to_data = constants.PATH_TO_DATA

if __name__ == '__main__':
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "i_7_types_nr_cust",str(sys.argv[-1])+".txt")

    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_discount_proportional_2segm_manyPUP", str(sys.argv[-1])+".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_DDRS",
                                     "DDRS_nrcust_10_nrpup3_delta_0.6_u_0.06_4.txt")
        #file_instance = os.path.join(path_to_data, "data", "i_DDRS", "DDRS_nrcust_10_nrpup3_delta_0.6_u_0.03_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    print(OCVRPInstance)

    #start_time = process_time()
    #EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType =solverType)
    #EnumerationSolver.exactPolicyByEnumeration(settings = 'NotWarmStart')
    #EnumerationSolver.exactPolicyByEnumeration()

    start_time = process_time()
    EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType=solverType)
    EnumerationSolver.exactPolicyByEnumeration_withoutGurobi_2segm()

    #EnumerationSolver.exactPolicyByEnumeration_2segm()
    print(prefix + 'Time_running_enumeration,s: ', process_time() - start_time)
    #EnumerationSolver.exactPolicyByEnumeration_withoutGurobi()
    #print(prefix + 'Time_running_enumeration_withouGurobi,s: ', process_time() - start_time)



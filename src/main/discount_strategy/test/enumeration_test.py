from pathlib import Path
import csv
import sys
from sys import argv
import os
from time import process_time
from src.main.discount_strategy.io import OCVRPParser

from src.main.discount_strategy.algorithms.exact.enumeration.enumeration_scenarios_2_segm import ScenarioEnumerationSolver
from src.main.discount_strategy.util import constants


prefix="tag: "
path_to_data = constants.PATH_TO_DATA

if __name__ == "__main__":
    #solverType = 'Concorde'
    solverType = 'Gurobi'
    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "i_7_types_nr_cust",str(sys.argv[-1])+".txt")

    file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional",
                                 "VRPDO_size_12_phome_0.2_ppup_0.0_incrate_0.06_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    print(OCVRPInstance)


    #start_time = process_time()
    #EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType =solverType)
    #EnumerationSolver.exactPolicyByEnumeration(settings = 'NotWarmStart')
    #EnumerationSolver.exactPolicyByEnumeration()

    start_time = process_time()
    EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType=solverType)
    EnumerationSolver.exactPolicyByEnumeration_withoutGurobi_2segm()
    print(prefix + 'Time_running_enumeration,s: ', process_time() - start_time)
    #EnumerationSolver.exactPolicyByEnumeration_withoutGurobi()
    #print(prefix + 'Time_running_enumeration_withouGurobi,s: ', process_time() - start_time)





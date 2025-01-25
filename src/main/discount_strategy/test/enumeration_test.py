# Code to find the best policy by enumerating through all possible policies
from pathlib import Path
import sys
import os
from time import process_time
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.algorithms.exact.enumeration.enumeration_scenarios_2_segm import ScenarioEnumerationSolver
from src.main.discount_strategy.util import constants
if __name__ == '__main__':
    solverType = 'Gurobi'#'Concorde'

    # In case the scipt is run from the command line (mip.sh in directory scripts), the last argument is the instance name
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "i_DDRS", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(constants.PATH_TO_DATA, "test_instance.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    print(OCVRPInstance)

    start_time = process_time()
    EnumerationSolver = ScenarioEnumerationSolver(instance=OCVRPInstance, solverType=solverType)
    EnumerationSolver.exactPolicyByEnumeration_withoutGurobi_2segm()
    print(constants.PREFIX + 'Time_running_enumeration,s: ', process_time() - start_time)

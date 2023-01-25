from pathlib import Path
import csv
import sys
from sys import argv
import os
from time import process_time
path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"io")
sys.path.insert(1, path_to_io)
import OCVRPParser

path_to_enumeration = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact","enumeration")
sys.path.insert(2, path_to_enumeration)
#from enumeration_scenarios import ScenarioEnumerationSolver
from enumeration_scenarios_2_segm import ScenarioEnumerationSolver
#path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
#sys.path.insert(5, path_to_util)
#import constants

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(3, path_to_exact)
from TSPSolver import TSPSolver


prefix="tag: "
if __name__ == "__main__":
    solverType = 'Concorde'
    #solverType = 'Gurobi'
    #file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_Santini_adapted",str(sys.argv[-1])+".txt")
    file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data", "instances_Santini_adapted",
                                                              "sz-13-prob_type-uniform-prob-0.30-fee_type-inverse_prob-fee-3.2.txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    print(OCVRPInstance)

    solverGurobi = TSPSolver(instance = instance, solverType = 'Gurobi')
    solverConcorde = TSPSolver(instance=instance, solverType='Concorde')

    for scenario in range(2 ** n):
        cost_Gurobi= solverGurobi.tspCost(scenario)
        cost_Concorde = solverConcorde.tspCost(scenario)
        if cost_Gurobi!=cost_Concorde:
            print("scenario", scenario, bin(scenario)[2:],cost_Gurobi, cost_Concorde )


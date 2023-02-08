
import random
import os

from src.main.discount_strategy.io import OCVRPParser

from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW

prefix="tag: "

def set_scenario(sample,n):
    scenario = 2**n - 1
    #print(sorted(sample))
    #print("scenario", scenario, bin(scenario))
    for i in sample:
        #print(i)
        scenario -= 2**(i-1)
    #print( bin(scenario))
    return scenario

if __name__ == "__main__":
    file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP",
                                 "VRPDO_size_18_phome_0.4_ppup_0.0_incrate_0.06_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    print(OCVRPInstance)
    rsPolicyID, rsValue = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST)
    print(bin(rsPolicyID)[2:], "RS cost", rsValue)

    n = OCVRPInstance.NR_CUST
    p_dev = OCVRPInstance.p_home
    discount = OCVRPInstance.FLAT_RATE_SHIPPING_FEE

    TSPSolver = TSPSolver(instance=OCVRPInstance, solverType='Gurobi')
    #TSPSolver.tspCost(scenario)
    #sample - customers that should be visited

    # customer form star:
    addedCust = 18
    #all customers from star:
    starCusts =[i for i in range(1,17) ]

    #for this specific instance:
    must_be_visited = [19,20,17]
    #starCusts = [i for i in range(1, addedCust)] + [i for i in range(addedCust + 1, n + 1)]
    for iter in range(10):
        sample = random.sample(starCusts, k = iter)
        for size in range(1,iter):
            subsample = random.sample(sample, k=size)

            scenario_without = set_scenario(sample+must_be_visited,n)
            subscenario_without = set_scenario(subsample+must_be_visited, n)

            scenario_with = set_scenario(sample+[addedCust] + must_be_visited, n)
            subscenario_with = set_scenario(subsample+[addedCust] + must_be_visited, n)

            insertion_cost_sample = TSPSolver.tspCost(scenario_with)-TSPSolver.tspCost(scenario_without)
            insertion_cost_subsample = TSPSolver.tspCost(subscenario_with) - TSPSolver.tspCost(subscenario_without)

            if(insertion_cost_sample > insertion_cost_subsample):
                print("")
                print("visited customers in sample: ", sorted(sample+must_be_visited),
                      "\nvistied customers in subsample: ", sorted(subsample+must_be_visited))
                print("insert in sample: ",insertion_cost_sample,"\ninsert in subsample: ", insertion_cost_subsample)

    # for RS:
    scenario_without = set_scenario(must_be_visited, n)
    scenario_with = set_scenario([addedCust] + must_be_visited, n)
    insertion_cost_rs = TSPSolver.tspCost(scenario_with) - TSPSolver.tspCost(scenario_without)

    print("")
    print("For RS:")
    print("insert in rs: ", insertion_cost_rs)
import numpy as np
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from time import process_time
prefix="tag: "
path_to_data = constants.PATH_TO_DATA

class ScenarioEnumerationSolver:

    def __init__(self, instance,solverType):
        self.instance = instance
        self.TSPSolver = TSPSolver(instance = instance, solverType = solverType)


    def exactPolicyByEnumeration_withoutGurobi_2segm(self):
        print("Solving DDRS with by exactly enumerating over all possible policies")
        n = self.instance.NR_CUST
        exp_discount = np.multiply(self.instance.shipping_fee, self.instance.p_pup_delta)
        Y = range(2 ** n)
        routeCost = {}
        start_time = process_time()
        for scenario in range(2**n):
            routeCost[scenario] = self.TSPSolver.tspCost(scenario)
        print(prefix+"time_calculation_all_TSPs:",process_time() - start_time)
        bestPolicy = 0
        bestPolicyCost = constants.BIGM_POLICY_COST

        for policy in Y:
            policyCost = 0
            for i in range(1, n + 1):
                if policy & (1 << (i - 1)):
                    policyCost += exp_discount[i]

            for scenario in Y:
                if not (~policy & scenario):
                    scenarioProb = probability.scenarioProb_2segm(scenario, policy, n, n, self.instance.p_pup_delta)
                    policyCost += scenarioProb* routeCost[scenario]
            if policyCost < bestPolicyCost:
                bestPolicyCost = policyCost
                bestPolicy = policy
        print(prefix+ "Obj_val: ", round(bestPolicyCost/constants.SCALING_FACTOR, 5))
        print(prefix+ "Best_policy_enumeration_ID: ", bestPolicy)
        print(prefix+ "Best_policy_enumeration: ",bin(bestPolicy)[2:].zfill(n))

        return (bestPolicyCost, bestPolicy)

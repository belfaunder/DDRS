import math
import numpy as np
import gurobipy as grb
import sys
import os
from pathlib import Path
from src.main.discount_strategy.util import constants

import numpy as np
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver

prefix="tag: "
path_to_data = constants.PATH_TO_DATA

def scenarioPossible(scenario, policy):
    for customerIndex, customerChoice in enumerate(scenario):
        if customerChoice == 1 and policy[customerIndex ] == 0:
            return False
    return True

class ScenarioEnumerationSolver:

    def __init__(self, instance, solverType):
        self.instance = instance
        self.TSPSolver = TSPSolver(instance = instance, solverType = solverType)


    def exactPolicyByEnumeration(self, **kwargs):
        if 'settings' in kwargs:
            settings = kwargs['settings']
        else:
            settings = 'Warmstart'

        n = self.instance.NR_CUST
        p_home = self.instance.p_home
        p_pup = self.instance.p_pup

        exp_discount  = np.multiply(self.instance.shipping_fee, np.subtract(np.ones(n+1),p_home)  )

        #print("exp_discount", exp_discount)
        Y = range(2**n)
        bestPolicy = 0
        routeCost = {}

        if settings == 'NotWarmStart':
            for scenario in range(2 ** n):
                routeCost[scenario] = self.TSPSolver.tspCostWihoutReuse(scenario)
        else:
            for scenario in range(2**n):
                routeCost[scenario] = self.TSPSolver.tspCost(scenario)



        opt_model = grb.Model(name="MIP Model")
        opt_model.setParam('OutputFlag', False)

        teta_vars = {policy: opt_model.addVar(vtype=grb.GRB.BINARY,
                                              name="y_{0}".format(policy)) for policy in Y}
        opt_model.addConstr(grb.quicksum(teta_vars[policy] for policy in Y) == 1)


        objective = 0
        for policy in Y:
            policy_exp_disc = 0
            for i in range(1, n + 1):
                if policy & (1 << (i - 1)):
                    policy_exp_disc += exp_discount[i]

            #routing = 0
            #objective += teta_vars[policy] * (bitCount(policy) * discount * (1 - p_dev))
            objective += teta_vars[policy] *policy_exp_disc
            for scenario in Y:
                scenarioProb = probability.scenarioProb(scenario, policy, n, n, p_home, p_pup)
                objective += teta_vars[policy] * (scenarioProb* routeCost[scenario])

        opt_model.ModelSense = grb.GRB.MINIMIZE
        opt_model.setObjective(objective)
        opt_model.optimize()

        for policy in Y:
            if teta_vars[policy].x > 0.5:
                opt_strategy_stochastic = policy
        expected_cost_stoch = opt_model.objVal
        print(prefix+ "Obj_val: ", round(expected_cost_stoch, 5))
        print(prefix+ "Best_policy_enumeration_ID: ", opt_strategy_stochastic)
        print(prefix+ "Best_policy_enumeration: ",bin(opt_strategy_stochastic)[2:].zfill(n))
        return (expected_cost_stoch, opt_strategy_stochastic)

    def exactPolicyByEnumeration_withoutGurobi(self):
        n = self.instance.NR_CUST
        p_home = self.instance.p_home
        p_pup = self.instance.p_pup

        exp_discount  = np.multiply(self.instance.shipping_fee, np.subtract(np.ones(n+1),p_home)  )

        #print("exp_discount", exp_discount)
        Y = range(2**n)
        bestPolicy = 0
        routeCost = {}

        for scenario in range(2**n):
            routeCost[scenario] = self.TSPSolver.tspCost(scenario)


        bestPolicy = 0
        bestPolicyCost = 10**10

        for policy in Y:
            policyCost = 0
            for i in range(1, n + 1):
                if policy & (1 << (i - 1)):
                    policyCost += exp_discount[i]

            for scenario in Y:
                scenarioProb = probability.scenarioProb(scenario, policy, n, n, p_home, p_pup)
                policyCost += scenarioProb* routeCost[scenario]

            if policyCost < bestPolicyCost:
                bestPolicyCost = policyCost
                bestPolicy = policy

        print(prefix+ "Obj_val: ", round(bestPolicyCost, 5))
        print(prefix+ "Best_policy_enumeration_ID: ", bestPolicy)
        print(prefix+ "Best_policy_enumeration: ",bin(bestPolicy)[2:].zfill(n))
        return (bestPolicyCost, bestPolicy)


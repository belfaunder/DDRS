import itertools
from src.main.discount_strategy.util.bit_operations import bitCount
import gurobipy as grb

import numpy as np
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from time import process_time

prefix="tag: "
path_to_data = constants.PATH_TO_DATA

prefix="tag: "

class ScenarioEnumerationSolver:

    def __init__(self, instance,solverType):
        self.instance = instance
        self.TSPSolver = TSPSolver(instance = instance, solverType = solverType)

    def exactPolicyByEnumeration(self):
        n = self.instance.NR_CUST
        exp_discount  = np.multiply(self.instance.shipping_fee, self.instance.p_pup_delta)
        Y = range(2**n)
        routeCost = {}
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
            objective += teta_vars[policy] *policy_exp_disc
            for scenario in Y:
                # if scenario is possible given the policy
                if not (~policy & scenario):
                    scenarioProb = probability.scenarioProb_2segm(scenario, policy, n, n, self.instance.p_pup_delta)
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

    def exactPolicyByEnumeration_withoutGurobi_2segm(self):
        n = self.instance.NR_CUST
        exp_discount = np.multiply(self.instance.shipping_fee, self.instance.p_pup_delta)
        Y = range(2 ** n)
        routeCost = {}
        start_time = process_time()
        for scenario in range(2**n):
            routeCost[scenario] = self.TSPSolver.tspCost(scenario)
        print(prefix+"time_calculation_all_TSPs:",process_time() - start_time)
        bestPolicy = 0
        bestPolicyCost = 10**10

        for policy in Y:
            policyCost = 0
            for i in range(1, n + 1):
                if policy & (1 << (i - 1)):
                    policyCost += exp_discount[i]

            for scenario in Y:
                if not (~policy & scenario):
                    scenarioProb = probability.scenarioProb_2_segm(scenario, policy, n, n, self.instance.p_pup_delta)
                    policyCost += scenarioProb* routeCost[scenario]

            if policyCost < bestPolicyCost:
                bestPolicyCost = policyCost
                bestPolicy = policy

        print(prefix+ "Obj_val: ", round(bestPolicyCost, 5))
        print(prefix+ "Best_policy_enumeration_ID: ", bestPolicy)
        print(prefix+ "Best_policy_enumeration: ",bin(bestPolicy)[2:].zfill(n))
        return (bestPolicyCost, bestPolicy)


    def expectedPolicyValue(self, policy):

        n = self.instance.NR_CUST
        p_dev = self.instance.deviationProbability
        discount = self.instance.FLAT_RATE_SHIPPING_FEE

        setMayVary = []
        for index, disc in enumerate(bin(policy)[::-1]):
            if disc == '1':
                setMayVary.append(index + 1)

        minNumDev = int((len(setMayVary) + 1) * p_dev - 1)
        numDev = minNumDev
        for numDev in range(minNumDev, len(setMayVary) + 1):
            if ((1 - p_dev) ** ( len(setMayVary) - numDev) * p_dev ** numDev) * comb( len(setMayVary), numDev) <= constants.EPSILONRS:
                break
        expectedValueDisc = discount * len(setMayVary) * (1 - p_dev)
        lbScenario = self.TSPSolver.tspCost(policy)
        ubScenario = self.TSPSolver.tspCost(0)
        expectedValueLb = lbScenario
        expectedValueUb = ubScenario
        for numDeviated in range(0, numDev + 1):
            #all_combinations = itertools.combinations(setMayVary, numDeviated)
            for combination in itertools.combinations(setMayVary, numDeviated):
                scenario = policy
                for offset in combination:
                    mask = ~(1 << (offset - 1))
                    scenario = scenario & mask
                scenarioCost = self.TSPSolver.tspCost(scenario)
                #scenarioCost =  self.TSPSolver.tspCostConcorde(scenario)
                scenarioProb = (1 - p_dev) ** bitCount(policy & scenario) * p_dev ** (
                                    bitCount(policy ^ scenario))
                #expectedValue += scenarioCost*scenarioProb
                expectedValueLb += scenarioProb * (scenarioCost - lbScenario)
                expectedValueUb -= scenarioProb * (ubScenario - scenarioCost)

                if (expectedValueUb-expectedValueLb)/expectedValueLb < constants.EPSILONRS:
                    break

            numDeviated += 1
        print("tag: CurrentVal: ", expectedValueLb + expectedValueDisc, expectedValueUb + expectedValueDisc)


        return expectedValueUb + expectedValueDisc

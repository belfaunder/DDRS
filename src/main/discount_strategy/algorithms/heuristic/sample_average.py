import math
import numpy as np
from itertools import combinations
import random
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
import itertools

def average_error(sampleCost, ss):
    average = 0
    error = 0
    for w in sampleCost:
        average += w
    average = average/ss
    for w in sampleCost:
        error += (w - average)**2

    #ss == 1 iff the size of set of deviated scenarios = 1
    if ss > 1:
        error = error / ((ss-1)*ss)
    else:
        error = 0
    return average, math.sqrt(error)*2


# Method to compute exact cost of a policy by enumerating all possible scenario costs.
# Used for small instances.
def one_policy_cost_estimation(instance, policy, solverType, **kwargs):
    n = instance.NR_CUST
    setMayVary = []
    for offset in range(n):
        mask = 1 << offset
        if policy & mask:
            setMayVary.append(offset + 1)
    exp_discount = np.multiply(instance.shipping_fee, np.subtract(np.ones(n+1), instance.p_home)  )
    solver = TSPSolver(instance=instance, solverType=solverType)
    cost = 0

    if 'routeCosts' in kwargs:
        routeCosts = kwargs['routeCosts']
    else:
        routeCosts = {}
        routeCosts[0] = solver.tspCost(0)
        scenario_1 = 2 ** instance.NR_CUST - 1
        routeCosts[scenario_1] = solver.tspCost(scenario_1)

    policy_exp_disc = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            policy_exp_disc += exp_discount[i]*constants.PROB_PLACE_ORDER
    cost += policy_exp_disc

    for num_may_vary in  range(bin(policy).count("1") + 1) :
        for combination in itertools.combinations(setMayVary, num_may_vary):
            scenario = policy
            # make a scenario given policy and combination of deviated nodes
            for offset in combination:
                mask = ~(1 << (offset - 1))
                scenario = scenario & mask
            scenarioProb = probability.scenarioProb_2segm(scenario, policy, n, n, instance.p_pup_delta)
            if scenario not in routeCosts:
                routeCosts[scenario] = solver.tspCost(scenario)
            cost += scenarioProb * routeCosts[scenario]
    cost =float(cost)
    if 'routeCosts' in kwargs:
        return   [cost,cost,cost], routeCosts
    else:
        return  [cost,cost,cost]

array_0_1 = np.arange(start = 0, stop = 2)
def set_random_combination(setMayVary, prob_dev_setMayVary):
    combination = []
    for index, i in enumerate(setMayVary):
        if np.random.choice(array_0_1, size=1, p= [prob_dev_setMayVary[index], 1-prob_dev_setMayVary[index]])==0:
            combination.append(i)
    return combination

# Method to evaluate cost of a poliy by samplling.
def sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance, policy, solverType, **kwargs):
    n = instance.NR_CUST
    p_home = instance.p_home
    exp_discount = np.multiply(instance.shipping_fee, instance.p_pup_delta)
    if 'solver' in kwargs:
        solver = kwargs['solver']
    else:
        solver = TSPSolver(instance, solverType)
    if 'sample_size' in kwargs:
        sample_size = kwargs['sample_size']
    else:
        sample_size = constants.SAMPLE_SIZE

    if 'setMayVary' in kwargs:
        setMayVary = kwargs['setMayVary']
    else:
        setMayVary = [cust.id for cust in instance.customers if policy &(1<< int(cust.id-1))]
    if 'routeCosts' in kwargs:
        routeCosts = kwargs['routeCosts']
    else:
        routeCosts = {}
        routeCosts[0] = solver.tspCost(0)
        scenario_1 = 2 ** instance.NR_CUST - 1
        routeCosts[scenario_1] = solver.tspCost(scenario_1)
    prob_dev_setMayVary = []
    for customer in setMayVary:
        mask = (1 << (customer - 1))
        if mask & policy:
            prob_dev_setMayVary.append(p_home[customer])
    policy_cost = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            policy_cost += exp_discount[i]

    sampleCost = []
    for i in range(sample_size):
        combination = set_random_combination(setMayVary, prob_dev_setMayVary)
        scenario = policy
        # make a scenario given policy and combination of deviated nodes
        for offset in combination:
            mask = ~(1 << (offset - 1))
            scenario = scenario & mask

        if scenario not in routeCosts:
            routeCosts[scenario] = solver.tspCost(scenario)
        sampleCost.append(routeCosts[scenario])
    sampleCost_mean, policy_cost_error = average_error(sampleCost, sample_size)

    policy_cost_lb = policy_cost +sampleCost_mean - policy_cost_error
    policy_cost_ub = policy_cost+sampleCost_mean+ policy_cost_error

    policy_cost_average = policy_cost +sampleCost_mean

    policy_cost_average = float(policy_cost_average)
    policy_cost_lb = float(policy_cost_lb)
    policy_cost_ub = float(policy_cost_ub)

    if 'routeCosts' in kwargs:
        return  [policy_cost_average, policy_cost_lb, policy_cost_ub], routeCosts
    else:
        return [policy_cost_average, policy_cost_lb, policy_cost_ub]

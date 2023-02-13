import sys
import os
from pathlib import Path

import math
import numpy as np
from itertools import combinations
import random

from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability

path_to_poibin = os.path.join((Path(os.path.abspath(__file__)).parents[5]),"ext","poibin-master")
sys.path.insert(1, path_to_poibin)
from poibin import PoiBin

import matplotlib.pyplot as plt


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

def samplingDeviated(routeCosts, setMayVary, v, policy, solver):
    sampleCost = 0
    for __ in range(constants.SAMPLE_SIZE):
        combination = random.sample(setMayVary, v)
        scenario = policy
        for offset in combination:
            mask = ~(1 << (offset - 1))
            scenario = scenario & mask
        if scenario not in routeCosts:
            routeCosts[scenario] = solver.tspCost(scenario)
        sampleCost += routeCosts[scenario]
    return sampleCost/constants.SAMPLE_SIZE, routeCosts

def samplingDeviated_PoissonBinomial(routeCosts, prob_dev_setMayVary_NORMED, setMayVary, policy,v,n,  solver, p_home, p_pup):
    sampleCost = 0
    sampleProb = 0

    for __ in range(constants.SAMPLE_SIZE):
        combination = random.sample(setMayVary, v)
        #combination = np.random.choice(setMayVary,size = v, p=prob_dev_setMayVary_NORMED,  replace=False)
        scenario = 2**n-1
        #print(combination)

        for offset in combination:
            mask = (1 << (offset - 1))
            scenario = scenario & ~mask

        if scenario not in routeCosts:
            routeCosts[scenario] = solver.tspCost(scenario)

        scenarioProb = probability.scenarioProb( scenario, policy, n, n, p_home, p_pup)
        #sampleCost += routeCosts[scenario]
        sampleCost += routeCosts[scenario]* scenarioProb
        sampleProb += scenarioProb
        #print("scenario", bin(scenario)[2:], routeCosts[scenario])
    #print("average:",v, sampleCost/constants.SAMPLE_SIZE)
    return sampleCost/sampleProb, routeCosts
    #return sampleCost / constants.SAMPLE_SIZE, routeCosts


# evaluation of cost of a policy with enumeration of all possible scenario costs
def one_policy_cost_estimation(instance, policy, solverType, **kwargs):
    n = instance.NR_CUST
    Y = range(2 ** n)
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

    for scenario in Y:
        # if scenario is possible given the policy
        if probability.scenarioPossible_2segm(scenario, policy, n,n):
            scenarioProb = probability.scenarioProb_2segm(scenario, policy, n, n, instance.p_pup_delta)
            if scenario not in routeCosts:
                routeCosts[scenario] = solver.tspCost(scenario)
            cost += scenarioProb * routeCosts[scenario]
            #print(scenario, bin(scenario), scenarioProb, routeCosts[scenario])

    if 'routeCosts' in kwargs:
        return   [cost,cost,cost], routeCosts
    else:
        return  [cost,cost,cost]

def set_random_combination(setMayVary, prob_dev_setMayVary):
    combination = []
    for index, i in enumerate(setMayVary):
        if np.random.choice([1, 0], size=1, p= [prob_dev_setMayVary[index], 1-prob_dev_setMayVary[index]])==1:
            combination.append(i)
    return combination

# evaluation of cost of a poliy by samplling. 1sample states that we do not sample for different gammas( number of visited people)
def sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance,setMayVary, policy, solverType, **kwargs):
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

    if 'routeCosts' in kwargs:
        routeCosts = kwargs['routeCosts']
    else:
        routeCosts = {}
        routeCosts[0] = solver.tspCost(0)
        scenario_1 = 2 ** instance.NR_CUST - 1
        routeCosts[scenario_1] = solver.tspCost(scenario_1)
    prob_dev_setMayVary = []
    total_sum_prob = 0
    for customer in setMayVary:
        prob_dev_setMayVary.append(p_home[customer])
    # for offset in range(1, n+1):
    #     mask = (1 << (offset - 1))
    #     if mask & policy:
    #         prob_dev_setMayVary.append(p_home[offset])
    #         total_sum_prob += p_home[offset]
    #     else:
    #         prob_dev_setMayVary.append(1-p_pup[offset])
    #         total_sum_prob += 1-p_pup[offset]

    policy_cost = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            policy_cost += exp_discount[i]
    policy_cost_error = 0
    cumulative_sum = 0
    sampleCost = []
    for i in range(sample_size):
        combination = set_random_combination(setMayVary, prob_dev_setMayVary)
        scenario = policy
        # make a scenario given policy and combination of deviated nodes
        for offset in combination:
            mask = ~(1 << (offset - 1))
            scenario = scenario & mask
        # scenario = 2 ** n - 1
        # for offset in combination:
        #     mask = (1 << (offset - 1))
        #     scenario = scenario & ~mask
        if scenario not in routeCosts:
            routeCosts[scenario] = solver.tspCost(scenario)
        sampleCost.append(routeCosts[scenario])
    sampleCost_mean, policy_cost_error = average_error(sampleCost, sample_size)

    policy_cost_lb = policy_cost +sampleCost_mean - policy_cost_error
    policy_cost_ub = policy_cost+sampleCost_mean+ policy_cost_error

    policy_cost_average = policy_cost +sampleCost_mean

    if 'routeCosts' in kwargs:
        return  [policy_cost_average, policy_cost_lb, policy_cost_ub], routeCosts
    else:
        return [policy_cost_average, policy_cost_lb, policy_cost_ub]

def sampleAverageApproximation_PoissonBinomial_1sample(instance, policy, solverType, **kwargs):
    n = instance.NR_CUST
    p_home = instance.p_home
    p_pup = instance.p_pup
    exp_discount = np.multiply(instance.shipping_fee, instance.p_pup_delta)
    if 'solver' in kwargs:
        solver = kwargs['solver']
    else:
        solver = TSPSolver(instance, solverType)
    if 'sample_size' in kwargs:
        sample_size = kwargs['sample_size']
    else:
        sample_size = constants.SAMPLE_SIZE

    if 'routeCosts' in kwargs:
        routeCosts = kwargs['routeCosts']
    else:
        routeCosts = {}
        routeCosts[0] = solver.tspCost(0)
        scenario_1 = 2 ** instance.NR_CUST - 1
        routeCosts[scenario_1] = solver.tspCost(scenario_1)
    prob_dev_setMayVary = []
    total_sum_prob = 0
    for offset in range(1, n+1):
        mask = (1 << (offset - 1))
        if mask & policy:
            prob_dev_setMayVary.append(p_home[offset])
            total_sum_prob += p_home[offset]
        else:
            prob_dev_setMayVary.append(1-p_pup[offset])
            total_sum_prob += 1-p_pup[offset]

    setMayVary = list(range(1,n+1))

    policy_cost = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            policy_cost += exp_discount[i]
    policy_cost_error = 0
    cumulative_sum = 0
    sampleCost = []
    for i in range(sample_size):
        combination = set_random_combination(setMayVary, prob_dev_setMayVary)
        scenario = 2 ** n - 1
        for offset in combination:
            mask = (1 << (offset - 1))
            scenario = scenario & ~mask
        if scenario not in routeCosts:
            routeCosts[scenario] = solver.tspCost(scenario)
        sampleCost.append(routeCosts[scenario])

    sampleCost_mean, policy_cost_error = average_error(sampleCost, sample_size)

    policy_cost_lb = policy_cost +sampleCost_mean - policy_cost_error
    policy_cost_ub = policy_cost+sampleCost_mean+ policy_cost_error

    policy_cost_average = policy_cost +sampleCost_mean

    if 'routeCosts' in kwargs:
        return  [policy_cost_average, policy_cost_lb, policy_cost_ub], routeCosts
    else:
        return [policy_cost_average, policy_cost_lb, policy_cost_ub]

def importanceSampling_RS(Bab,  policy):

    n = Bab.instance.NR_CUST
    p_home = Bab.instance.p_home
    p_pup = Bab.instance.p_pup
    exp_discount = np.multiply(Bab.instance.shipping_fee, np.subtract(np.ones(n+1),p_home)  )

    solver = Bab.TSPSolver
    sample_size= constants.IMPORTANCE_SAMPLE_SIZE

    sampleScenarioID, sampleScenarioCost =[],[]

    prob_dev_setMayVary = []
    total_sum_prob = 0
    for offset in range(1, n+1):
        mask = (1 << (offset - 1))
        if mask & policy:
            prob_dev_setMayVary.append(p_home[offset])
            total_sum_prob += p_home[offset]
        else:
            prob_dev_setMayVary.append(1-p_pup[offset])
            total_sum_prob += 1-p_pup[offset]

    setMayVary = list(range(1,n+1))

    policy_cost = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            policy_cost += exp_discount[i]
    policy_cost_error = 0
    cumulative_sum = 0
    for i in range(sample_size):
        combination = set_random_combination(setMayVary, prob_dev_setMayVary)
        scenario = 2 ** n - 1
        for offset in combination:
            mask = (1 << (offset - 1))
            scenario = scenario & ~mask
        if scenario not in Bab.instance.routeCost:
            Bab.instance.routeCost[scenario] = solver.tspCost(scenario)
        sampleScenarioID.append(scenario)
        sampleScenarioCost.append( Bab.instance.routeCost[scenario])
    sampleCost_mean, policy_cost_error = average_error(sampleScenarioCost, sample_size)
    policy_cost_average = policy_cost +sampleCost_mean
    print("rs cost average",policy,  policy_cost_average)
    return  sampleScenarioID, sampleScenarioCost

def importance_probability( scenario, policy, rsPolicy, n,p_home, p_pup):
    probability_rs =  probability.scenarioProb(scenario, rsPolicy, n, n, p_home, p_pup)
    probability_policy = probability.scenarioProb(scenario, policy, n, n, p_home, p_pup)
    imp_prob = probability_policy/probability_rs
    return imp_prob

def importanceSampling(instance, policy, rsPolicy, sampleScenarioID, sampleScenarioCost):
    n = instance.NR_CUST
    p_home = instance.p_home
    p_pup = instance.p_pup
    exp_discount = np.multiply(instance.shipping_fee, np.subtract(np.ones(n+1),p_home)  )

    discount_cost = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            discount_cost += exp_discount[i]

    sampleCost = []
    for index, scenario in enumerate(sampleScenarioID):
        sampleCost.append(sampleScenarioCost[index]*importance_probability( scenario, policy, rsPolicy, n,p_home, p_pup))

    sampleCost_mean, policy_cost_error = average_error(sampleCost, constants.IMPORTANCE_SAMPLE_SIZE)

    policy_cost_average = discount_cost +sampleCost_mean

    print("with importance sampling", policy,policy_cost_average,  policy_cost_average -policy_cost_error,  policy_cost_average +policy_cost_error )

    normal_sampling = sampleAverageApproximation_PoissonBinomial_1sample(instance = instance, policy=policy, solverType='Gurobi',
                                                       sample_size=constants.HEURISTIC_SAMPLE_SIZE)
    print(normal_sampling)

    #rint()
    return policy_cost_average



def sampleAverageApproximation_PoissonBinomial(instance, policy):
    n = instance.NR_CUST
    p_home = instance.p_home
    p_pup = instance.p_pup
    exp_discount = np.multiply(instance.shipping_fee, np.subtract(np.ones(n+1),p_home)  )
    solver = TSPSolver(instance, 'Gurobi')

    routeCosts = {}
    routeCosts[0] = solver.tspCost(0)
    scenario_1 = 2 ** instance.NR_CUST - 1
    routeCosts[scenario_1] = solver.tspCost(scenario_1)

    prob_dev_setMayVary = []
    total_sum_prob = 0
    for offset in range(1, n+1):
        mask = (1 << (offset - 1))
        if mask & policy:
            prob_dev_setMayVary.append(p_home[offset])
            total_sum_prob += p_home[offset]
        else:
            prob_dev_setMayVary.append(1-p_pup[offset])
            total_sum_prob += 1-p_pup[offset]
    print(prob_dev_setMayVary)
    prob_dev_setMayVary_NORMED = []
    for i in  prob_dev_setMayVary:
        prob_dev_setMayVary_NORMED.append(i / total_sum_prob)
    print(prob_dev_setMayVary_NORMED)
    # for  index, prob in enumerate(prob_dev_setMayVary):
    #    prob_dev_setMayVary[index] = prob/total_prob_vary
    #print(prob_dev_setMayVary_NORMED)
    weight = []
    # the lib for estimation for Poisson binomial distribution
    pb = PoiBin(prob_dev_setMayVary)
    for num_visited in range(n + 1):
        weight.append(pb.pmf([num_visited])[0])
        # weight.append(((1 - np.amin(p_dev[1:])) ** (len(setMayVary) - numDev) * np.amin(p_dev[1:]) ** numDev) * comb(len(setMayVary),
        #                                                                        numDev))

    weightNotCalculated = 0

    setMayVary = list(range(1,n+1))

    #print(prob_dev_setMayVary_NORMED, setMayVary)
    policy_cost = 0
    for i in range(1, n + 1):
        if policy & (1 << (i - 1)):
            policy_cost += exp_discount[i]
    policy_cost_error = 0
    #v - number of visited customers
    for v in range(0, len(weight)):
        #print("\nvisited",v,  weight[v])
        average_v = 0
        num_v = 0
        if weight[v] > constants.EPSILONRS / n:
            sample = []
            limit_ss = math.factorial(n)/(math.factorial(v)*math.factorial(n-v))
            #if True:
            if limit_ss < constants.SAMPLE_SIZE * constants.NUMBER_OF_SAMPLES:
                all_combinations = combinations(setMayVary, v)
                for combination in all_combinations:
                    scenario = 2**n-1
                    for offset in combination:
                        mask = (1 << (offset - 1))
                        scenario = scenario & ~mask
                    policy_cost += solver.tspCost(scenario) * probability.scenarioProb(scenario, policy, n, n, p_home, p_pup)
                    average_v  += solver.tspCost(scenario)* probability.scenarioProb(scenario, policy, n, n, p_home, p_pup)
                    #num_v+=1
                    #if v==5:
                    #    print(bin(scenario)[2:], solver.tspCost(scenario) )
                print("exactly:", v, average_v/weight[v])
            else:
            #if True:
                cumulative_sum = 0
                sampleCost = []
                for i in range(constants.NUMBER_OF_SAMPLES):
                    sampleCostTemp, routeCosts = samplingDeviated_PoissonBinomial(routeCosts, prob_dev_setMayVary_NORMED,
                                                                                  setMayVary,policy, v, n, solver, p_home, p_pup)
                    sampleCost.append(sampleCostTemp)
                #print(sampleCost)
                average, error_mean = average_error(sampleCost, constants.NUMBER_OF_SAMPLES)
                print("estimation:", v, average, " 3 sigma: ",error_mean,"\n")

                policy_cost += average * weight[v]
                policy_cost_error += error_mean * weight[v]
        else:
            weightNotCalculated +=weight[v]

    policy_cost_lb = policy_cost + weightNotCalculated * routeCosts[scenario_1] - policy_cost_error
    policy_cost_ub = policy_cost + weightNotCalculated * routeCosts[0] + policy_cost_error


    return policy_cost_lb, policy_cost_ub


def sampleAverageApproximation_Binomial(instance, policy):
    n = instance.NR_CUST
    p_dev = instance.deviationProbability
    discount = instance.FLAT_RATE_SHIPPING_FEE
    solver = TSPSolver(instance,'Gurobi')

    routeCosts = {}
    routeCosts[0] = solver.tspCost(0)
    scenario_1 = 2 ** instance.NR_CUST - 1
    routeCosts[scenario_1] = solver.tspCost(scenario_1)

    setMayVary = []
    for index, disc in enumerate(bin(policy)[::-1]):
        if disc == '1':
            setMayVary.append(index + 1)

    minNumDev = int((len(setMayVary) + 1) * p_dev - 1)
    numDev = minNumDev
    weight = []
    #for numDev in range(len(setMayVary) + 1):
    #    weight.append(((1 - p_dev) ** (len(setMayVary) - numDev) * p_dev ** numDev) * comb(len(setMayVary),
    #                                                                            numDev))

    weightToCalc = 0
    for index, i in enumerate(weight):
        if weightToCalc < 1- constants.EPSILONRS/2:
            weightToCalc += i
        else:
            weight.pop(index)
    #for index, i in enumerate(weight):
    #    print(index, i, weight[index])
    weightNotCalculated = 1-weightToCalc

    policy_cost = discount * len(setMayVary) * (1 - p_dev)
    policy_cost_error = 0

    policy_cost += solver.tspCost(policy) * weight[0]
    for v in range(1, len(weight)):
        sample = []
        limit_ss = comb(len(setMayVary),v)
        if limit_ss < constants.SAMPLE_SIZE*constants.NUMBER_OF_SAMPLES:
            sampleCost = 0
            all_combinations = combinations(setMayVary, v)
            for combination in all_combinations:
                scenario = policy
                for offset in combination:
                    mask = ~(1 << (offset - 1))
                    scenario = scenario & mask
                #print("scenario", bin(scenario)[2:], solver.tspCost(scenario))
                sampleCost += solver.tspCost(scenario)
            policy_cost += sampleCost*((1 - p_dev) ** (len(setMayVary) - v) * p_dev ** v)

        else:
            cumulative_sum = 0
            sampleCost = []
            for i in range(constants.NUMBER_OF_SAMPLES):
                sampleCostTemp, routeCosts = samplingDeviated(routeCosts, setMayVary, v, policy, solver)
                sampleCost.append(sampleCostTemp)
            average, error_mean = average_error(sampleCost, constants.NUMBER_OF_SAMPLES)
            policy_cost += average * weight[v]
            policy_cost_error += error_mean* weight[v]
    return policy_cost+weightNotCalculated*routeCosts[scenario_1]-policy_cost_error,policy_cost+weightNotCalculated*routeCosts[0]+policy_cost_error

def sampling_comparison_scenarios_2_segments(y1, y2, sets, input_data, max_gap, costs_w, opt_model):
    cum_weight = 0

    n1, n2 = inpup(y1), inpup(y2)
    set_discount1 = []
    iter = 1
    for i in y1[1:]:
        if i == 1:
            set_discount1.append(iter)
        iter += 1
    set_discount2 = []
    iter = 1
    for i in y2[1:]:
        if i == 1:
            set_discount2.append(iter)
        iter += 1

    discount = input_data['discount']
    sample_size = 20
    rough_average = 150
    rough_gap = 100

    policy_cost = 0
    policy_cost_error = 0
    start_time = datetime.now()
    segment_probability = input_data["segment_probability"]
    p_dev = segment_probability[0]
    weight = []
    for i in range(n1 + 1):
        weight.append(
            ((1 - p_dev) ** (n1 - i) * p_dev ** i) * math.factorial(n1) / (math.factorial(n1 - i) * math.factorial(i)))

    for v in range(n + 1):
        if weight[v] > 0.005:
            sample = []
            limit_ss = round(math.factorial(n) / (math.factorial(n - v) * math.factorial(v)))
            if limit_ss < 20:
                ss = limit_ss
                all_combinations = combinations(set_discount, n - v)
                for combination in all_combinations:
                    w = []
                    for cust in range(input_data["n"] + 1):
                        if cust in combination:
                            w.append(1)
                        else:
                            w.append(0)
                    sample.append(w)
            else:
                ss = min(sample_size, limit_ss)
                for i in range(ss):
                    w = []
                    customers_to_change = np.random.choice(set_discount, v)
                    for cust in range(1, input_data["n"]):
                        if cust in customers_to_change:
                            w.append(0)
                        else:
                            w.append(y[cust])
                    w.insert(0, 0)
                    sample.append(w)
            iter = 0
            sample_cost = {}
            for w in sample:
                if tuple(w) not in costs_w:
                    sample_cost[iter], opt_model = tsp_lazy(w, sets, input_data, opt_model)
                    costs_w[tuple(w)] = sample_cost[iter]
                else:
                    sample_cost[iter] = costs_w[tuple(w)]
                iter += 1
            average, error_mean = average_error(sample_cost, ss)
            policy_cost += average * weight[v]
            if limit_ss > 100:
                policy_cost_error += error_mean * weight[v]
        else:
            policy_cost_error += rough_average * weight[v]



    weight_dict = {}
    for i in range(n + 1):
        weight_dict[i] = ((1-p_dev) ** (n - i) * p_dev ** i) * math.factorial(n) / (math.factorial(n - i) * math.factorial(i))
    gap =  (inpup(y1) - inpup(y2))*discount*(1-p_dev)
    iterator = -1
    error = max_gap

    while (iterator < n) and (abs(gap) < error) and (error-2*(1-cum_weight)*max_gap)<=( abs(gap)):

        iterator +=1

        v = max(weight_dict, key=weight_dict.get)
        current_weight = weight_dict[v]
        del weight_dict[v]

        limit_ss = round(math.factorial(n) / (math.factorial(n - v) * math.factorial(v)))
        ss = min(sample_size, limit_ss)
        list_average = {}
        sample1 = []
        sample2 = []
        for i in range(ss):
            w1 = []
            customers_to_change = np.random.randint(1, n+1,size=v)
            for cust in range(1,n+1):
                if cust in customers_to_change:
                    w1.append(abs(y1[cust] - 1))
                else:
                    w1.append(y1[cust])
            w1.insert(0, 0)
            sample1.append(w1)
        #for the second sytategy
        for i in range(ss):
            w2 = []
            customers_to_change = np.random.randint(1, n+1,size=v)
            for cust in range(1, n + 1):
                if cust in customers_to_change:
                    w2.append(abs(y2[cust] - 1))
                else:
                    w2.append(y2[cust])
            w2.insert(0, 0)
            sample2.append(w2)
        iter = 0
        sample_cost1 = {}
        for w in sample1:
            if tuple(w) not in costs_w:
                sample_cost1[iter],opt_model = tsp_lazy(w, sets, input_data, opt_model)
                costs_w[tuple(w)] = sample_cost1[iter]
            else:
                sample_cost1[iter] = costs_w[tuple(w)]
            iter += 1
        iter = 0
        sample_cost2 = {}
        for w in sample2:
            if tuple(w) not in costs_w:
                sample_cost2[iter], opt_model = tsp_lazy(w, sets, input_data, opt_model)
                costs_w[tuple(w)] = sample_cost2[iter]
            else:
                sample_cost2[iter] = costs_w[tuple(w)]
            iter +=1

        #sample_cost_average = {}
        #for i in sample_cost1:
        #    sample_cost_average[i] = sample_cost1[i]-sample_cost2[i]
        #average_gap, error_mean = average_error(sample_cost_average, ss)
        #error += error_mean ** 2  * current_weight
        average1, error_mean1 = average_error(sample_cost1, ss)
        average2, error_mean2 = average_error(sample_cost2, ss)

        if ss < limit_ss:
            error += math.sqrt(error_mean1**2 + error_mean2**2 )*current_weight

        cum_weight += current_weight
        gap += (average1 - average2) * current_weight
        #gap += average_gap* current_weight
        error -= current_weight*max_gap

    if (error-2*(1-cum_weight)*max_gap)>=( abs(gap)):
        min_y = [y1, y2]
    else:
        if (gap > 0):
            min_y = [y2]
        else:
            min_y = [y1]

    return min_y, costs_w, opt_model


if __name__ == "__main__":
    setMayVary = list(range(1,15))
    prob_dev_setMayVary = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.5]
    print(setMayVary)
    prob_dev_setMayVary_NORMED = [0.02631578947368421, 0.02631578947368421, 0.02631578947368421, 0.02631578947368421, 0.02631578947368421,
     0.02631578947368421, 0.13157894736842105, 0.13157894736842105, 0.13157894736842105, 0.13157894736842105,
     0.13157894736842105, 0.02631578947368421, 0.02631578947368421, 0.13157894736842105]

    size_sample = 1000
    sample = []
    for i in range(size_sample):
        sample.extend(set_random_combination(setMayVary,prob_dev_setMayVary))

    #print(sample)
    plt.hist(sample, bins = 14)
    plt.show()
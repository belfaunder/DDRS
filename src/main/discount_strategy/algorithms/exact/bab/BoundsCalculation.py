import os
import sys
from pathlib import Path

import itertools
#import random
from scipy.special import comb
import math
from itertools import dropwhile

from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.util.bit_operations import bitCount

from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from src.main.discount_strategy.algorithms.heuristic.sample_average import importanceSampling


def coveredWithoutIntersect(policy, prev, covered, n, p_home):
    #covered = policy & covered
    weight_intersect = 0
    if prev:
        prev_current = []
        for s in prev:
            weight_intersect += coveredWithoutIntersect(policy, prev_current, covered & s, n, p_home)
            prev_current.append(s)
    return setWeightCovered(policy, covered, n, p_home) - weight_intersect

#CHANGESPROBABILITY
def setWeightCovered(policy, scenario, n, p_home):
    weightCovered = 1
    # cumulative weight of all scenarios that more expensive than this lbScenario
    for offset in range(n):
        mask = 1 << offset
        if not (scenario & mask) and (policy & mask):
            weightCovered *= p_home[offset+1]#*constants.PROB_PLACE_ORDER
           # else:
           #     weightCovered *= (1-p_pup[offset + 1])*constants.PROB_PLACE_ORDER
        # if (scenario & mask) < (policy & mask):
        #    probTemp += 1
    #weightCovered = p_dev ** (probTemp)
    return weightCovered


def recalculateLbCovered(p_home, node, n):
    #return 0,  0
    # gamma is the number of customers who are vistied
    # here we look for the min gamma for which no tsp was exactly calculated
    # for gamma in range(n , len(node.setNotGivenDiscount) - 1, -1):
    #    if node.tspDict.get(gamma):
    #        if len(node.tspDict[gamma]) > 0:
    #            break
    # if gamma == 0:
    #    gamma = n
    # lbScenarios = {k: v for k, v in node.lbScenarios.items() if  bitCount(k)>= n-gamma}

    # policy = policy that offers discount to the same customers as node + those who were not defined
    policy = node.withDiscountID + ((1 << (n+1)) - (1<<node.layer))
    #for i in range(node.layer, n):
    #    policy += 2 ** i

    weight_scenario_solely = {}
    node.lbCoveredWeightBest = 0
    for scenario in node.lbScenarios:
        #weight_scenario_solely[scenario] = node.lbScenarios[scenario]
        weight_scenario_solely[scenario] = setWeightCovered(policy, scenario, n, p_home) * node.lbScenarios[
            scenario]
        node.lbCoveredWeightBest = max(weight_scenario_solely[scenario], node.lbCoveredWeightBest)

        #node.lbCoveredWeightBest = max(weight_scenario_solely[scenario], node.lbCoveredWeightBest)
    #lbSorted = sorted(node.lbScenarios.items(), key=lambda x: x[1], reverse=True)
    lbSorted = sorted(node.lbScenarios.items(), key=lambda x: weight_scenario_solely[x[0]], reverse=True)
    #lbSorted = sorted(node.lbScenarios.items(), key=lambda x: node.lbScenarios[x[0]], reverse=True)

    lbAdded = []
    alreadyCovered = []
    lbScenariosNew = {}
    lbCovered = 0
    probCovered = 0

    # treshold limit the number of
    if len(node.lbScenarios) > 5:
        treshold = 0.1 * node.lbScenario[1]
    else:
        treshold = 0

    # TODO: sort lbSorted by the weightCovered*scenario[1] (without considering intersections) and then do the normal pickipng
    # (or even take only 5-10 first from sorted
    # TODO: add new scenarios into lbScenarios when calculate new TSP or from dict


    for scenario in lbSorted:
        if scenario[0] not in lbAdded:
            if probCovered < 1:
                weightCovered = coveredWithoutIntersect(policy, lbAdded, scenario[0], n, p_home)
                improve = weightCovered * scenario[1]
                if improve > treshold:
                    lbCovered += improve
                    lbAdded.append(scenario[0])
                    probCovered += weightCovered
                    # if scenario itself is in tspDict we do not calculate it in "prob covered", but in
                    # "prob exact", however, we calculate possible covered scenarios

                    # if weightCovered > treshold:
                    # if improve> treshold:
                    lbScenariosNew[scenario[0]] = scenario[1]

    # remove any scenario from lbCoveredProb if this scenario was included in calculation of exactValue(prob)
    if  len(node.tspProbDict) < 6:
        for scenario in node.tspProbDict:
            for sc_covered in lbScenariosNew:
                to_remove = 1
                for offset in range(n):
                    if ((1 << offset) & sc_covered) < ((1 << offset) & scenario):
                        to_remove = 0
                        break
                if to_remove:
                    probCovered -= node.tspProbDict[scenario]
                    lbCovered -= node.tspProbDict[scenario]*lbScenariosNew[sc_covered]
                    break
    else:
        lbCovered = max(0, lbCovered-node.exactValue )
        probCovered = max(0, probCovered - node.exactValueProb)
    node.lbScenarios = lbScenariosNew

    if probCovered>0:
        lbDensityCovered = lbCovered/probCovered
    else:
        lbDensityCovered = node.lbScenario[1]
    return probCovered, lbDensityCovered

#CHANGESPROBABILITY
def ub_insertion_cost(instance, setNotGivenDiscount, diff_customer):
    p_home = instance.p_home
    p_pup = instance.p_pup
    ub_insertion = instance.ubInsertion[diff_customer]

    insertion_cost = 0
    p_left = 1
    for i in ub_insertion:
        if i != 0:
            if i in setNotGivenDiscount:
                insertion_cost += ub_insertion[i] *  p_left#*constants.PROB_PLACE_ORDER
                p_left = 0
            else:
                insertion_cost += ub_insertion[i] * p_home[i] * p_left#*constants.PROB_PLACE_ORDER
                p_left *= (1 - p_home[i])
        else:
            insertion_cost += ub_insertion[i] * p_left
    ub = (insertion_cost  - instance.shipping_fee[diff_customer])* instance.p_pup_delta[diff_customer]
    return ub

#CHANGESPROBABILITY
def lb_insertion_cost(instance, setGivenDiscount, diff_customer):

    lb_insertion = instance.lbInsertion[diff_customer]
    p_home = instance.p_home
    p_pup = instance.p_pup
    n = instance.NR_CUST
    insertion_cost = 0
    p_left = 1
    previous_insertion = []

    for (i, j) in lb_insertion:
        if p_left > 0:
            if [i, j] != [0, 0]:
                prob_current = 1
                if i == j:
                    if i in setGivenDiscount:
                        prob_current = p_home[i]
                else:
                    if (i not in [0, n + 1]):
                        if (i in setGivenDiscount):
                            prob_current *= p_home[i]
                    if (j not in [0, n + 1]):
                        if j in setGivenDiscount:
                            prob_current *= p_home[j]

                considered_pair =[]
                for (m, k) in previous_insertion:
                    if (m == i or m == j) and k not in considered_pair:
                        considered_pair.append(k)
                        if (k not in [0, n + 1]):
                            if k in setGivenDiscount:
                                prob_current *= (1 - p_home[k])
                            else:
                                prob_current = 0
                        else:
                            prob_current = 0
                    if k == i or k == j and m not in considered_pair:
                        considered_pair.append(m)
                        if m not in [0, n + 1]:
                            if m in setGivenDiscount:
                                prob_current *= (1 - p_home[m])
                            else:
                                prob_current = 0
                        else:
                            prob_current *= 0

                insertion_cost += min(prob_current, p_left) * lb_insertion[i, j]
                if min(prob_current, p_left) > 0:
                    previous_insertion.append([i, j])

                p_left -= prob_current
            else:
                insertion_cost += lb_insertion[i, j] * p_left
        else:
            break
    lb =  (insertion_cost-instance.shipping_fee[diff_customer]) * instance.p_pup_delta[diff_customer]

    return lb


def updateByInsertionCost(node, Bab):
    status_not_changed = True
    fork_node_child = node
    fork_node = node.parent

    if node.fathomedState:
        return False
    else:
        if node.parent is Bab.root:
            diff_customer = 1
        else:
            diff_customer = (node.parent.withDiscountID + node.parent.noDiscountID).bit_length() + 1
        if node.isRightChild():
            if node.parent.children[0].fathomedState:
                additionSibling = node.parent.children[0].lbVal() - Bab.bestNode.ubVal()
            else:
                additionSibling = 0
            lbInsertionCost = lb_insertion_cost(Bab.instance, node.setGivenDiscount, diff_customer)

            if lbInsertionCost + additionSibling > -constants.EPS*Bab.bestNode.lbVal():
                node.lbRoute = node.parent.children[0].lbVal() + lbInsertionCost + additionSibling
                node.ubRoute = node.lbRoute
                node.lbExpDiscount = 0
                node.ubExpDiscount = 0
                # node.lbRoute = Bab.bestUb + lbInsertionCost + additionSibling + constants.EPS
                # node.fathomed()
                node.fathomedState = True
                status_not_changed = False
        else:

            if node.parent.children[1].fathomedState:
                additionSibling = node.parent.children[1].lbVal() - Bab.bestNode.ubVal()

            else:
                additionSibling = 0
            ubInsertionCost = ub_insertion_cost(Bab.instance, node.setNotGivenDiscount, diff_customer)

            if -ubInsertionCost + additionSibling > -constants.EPS*Bab.bestNode.lbVal():
                node.lbRoute = node.parent.children[1].lbVal() - ubInsertionCost + constants.EPS + additionSibling
                node.ubRoute = node.lbRoute
                node.lbExpDiscount = 0
                node.ubExpDiscount = 0
                # node.lbRoute = Bab.bestUb - ubInsertionCost + constants.EPS + additionSibling
                # node.fathomed()
                node.fathomedState = True
                status_not_changed = False

    return status_not_changed


def compareDynamicParent(node, sibling, diff_customer, Bab):
    # additionSibling = sibling.lbVal() - Bab.bestNode.ubVal()
    additionSibling = 0
    ubInsertionCost = ub_insertion_cost(Bab.instance, sibling.setNotGivenDiscount, diff_customer)
    if -ubInsertionCost + additionSibling >= 0:
        node.lbRoute = sibling.lbVal() - ubInsertionCost + additionSibling
        node.ubRoute = node.lbRoute
        node.lbExpDiscount = 0
        node.ubExpDiscount = 0
        node.fathomedState = True
        return True
    return False


def compareDynamicChild(node, sibling, diff_customer, Bab):
    # additionSibling = sibling.lbVal() - Bab.bestUb
    additionSibling = 0
    lbInsertionCost = lb_insertion_cost(Bab.instance, node.setGivenDiscount, diff_customer)
    # for i in range(1, node.NR_CUST+1):
    #    if i not in node.setGivenDiscount:
    if lbInsertionCost + additionSibling >= 0:
        node.lbRoute = node.parent.children[0].lbVal() + lbInsertionCost + additionSibling
        node.ubRoute = node.lbRoute
        node.lbExpDiscount = 0
        node.ubExpDiscount = 0
        node.fathomedState = True
        status_not_changed = False
        return True
    return False

def updateBoundsFromLayer(Bab, node):
    if (Bab.bestNode.number_disc - node.number_disc == 1) and node.layer == Bab.instance.NR_CUST:
        pass
    elif (Bab.bestNode.number_disc - node.number_disc == -1) and node.layer == Bab.instance.NR_CUST:
        diff_customer =  (node.withDiscountID - Bab.bestNode.withDiscountID).bit_length()
        lbInsertionCost = lb_insertion_cost(Bab.instance, node.setGivenDiscount, diff_customer)
        ubInsertionCost = ub_insertion_cost(Bab.instance, node.setNotGivenDiscount, diff_customer)
        if -ubInsertionCost + additionSibling > -constants.EPS * Bab.bestNode.lbVal():
            node.lbRoute = node.parent.children[1].lbVal() - ubInsertionCost + constants.EPS + additionSibling
            node.ubRoute = node.lbRoute
            node.lbExpDiscount = 0
            node.ubExpDiscount = 0
            # node.lbRoute = Bab.bestUb - ubInsertionCost + constants.EPS + additionSibling
            # node.fathomed()
            node.fathomedState = True
            status_not_changed = False

        if lbInsertionCost  > -constants.EPS * Bab.bestNode.lbVal():
            node.lbRoute = node.parent.children[0].lbVal() + lbInsertionCost + additionSibling
            node.ubRoute = node.lbRoute
            node.lbExpDiscount = 0
            node.ubExpDiscount = 0
            # node.lbRoute = Bab.bestUb + lbInsertionCost + additionSibling + constants.EPS
            # node.fathomed()
            node.fathomedState = True
            status_not_changed = False
        pass


def updateBoundsFromDictionary(Bab, node):
    n = Bab.instance.NR_CUST
    # calculate the worst UB on the insetion cost given the information about customers with discount
    if node.parent is not None:
        if node is not Bab.bestNode:
            status_not_changed = updateByInsertionCost(node, Bab)
        else:
            status_not_changed = True

        if status_not_changed:
            setGivenDiscount = node.setGivenDiscount
            number_offered_discounts = len(setGivenDiscount)
            routeCost = Bab.instance.routeCost

            # limit the number of deviated since function may be too slow for small probabilities
            #num_may_deviate = Bab.instance.maxNumDeviated[len(setGivenDiscount)]

            if (2 ** n - 1 - node.noDiscountID) in routeCost:
                node.updateLbScenario(routeCost[2 ** n - 1 - node.noDiscountID], Bab.instance.p_home, n)

            # limit the number of deviated since function may be too slow for small probabilities
            num_may_deviate = Bab.instance.maxNumDeviated[len(setGivenDiscount)]
            setMayVary = node.setGivenDiscount

            # gamma is the number of visited customers
            for gamma in range(n - len(setMayVary),
                               min(n + 1,
                                   num_may_deviate + len(node.setNotGivenDiscount) + n - node.layer)):
                if gamma not in node.tspDict:
                    node.tspDict[gamma] = []
                if len(node.tspDict[gamma]) == comb(len(node.setGivenDiscount), n - gamma):
                    continue
                for combination in itertools.combinations(setMayVary, gamma - len(
                        node.setNotGivenDiscount) - n + node.layer):
                    scenario = node.withDiscountID
                    # make a scenario given policy and combination of deviated nodes
                    for offset in combination:
                        mask = ~(1 << (offset - 1))
                        scenario = scenario & mask
                    # check if scenario is already in node
                    if scenario not in node.tspDict[gamma]:
                        scenarioCost = routeCost.get(scenario)
                        if scenarioCost:
                            node.tspDict[gamma].append(scenario)
                            scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, node.layer, n,
                                                                          Bab.instance.p_pup_delta)

                            node.tspProbDict[scenario] = scenarioProb
                            lbCoveredProbNew = max(0, node.lbCoveredProb - scenarioProb)
                            node.lbRoute += scenarioProb * (scenarioCost - node.lbScenario[1]) + (
                                    node.lbDensityCovered - node.lbScenario[1]) * (
                                                        lbCoveredProbNew - node.lbCoveredProb)
                            node.lbCoveredProb = lbCoveredProbNew
                            node.ubRoute -= (routeCost[0] - scenarioCost) * scenarioProb
                            node.exactValueProb += scenarioProb
                            node.exactValue += scenarioCost * scenarioProb

            # for scenario, scenarioCost in dropwhile(lambda x: x[0] != node.lastEnteranceDictionary, routeCost.items()):
            #     if probability.scenarioPossible_2segm(scenario,  node.withDiscountID, node.layer, n):
            #         #for scenario in routeCost:
            #         gamma = n - bitCount(scenario)
            #         if not node.tspDict.get(gamma):
            #             node.tspDict[gamma] = []
            #         if scenario not in node.tspDict[gamma]:
            #             node.tspDict[gamma].append(scenario)
            #             scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, node.layer,
            #                                                     n, Bab.instance.p_pup_delta)
            #             node.tspProbDict[scenario] = scenarioProb
            #             lbCoveredProbNew = max(0, node.lbCoveredProb - scenarioProb)
            #             node.lbRoute += scenarioProb * (scenarioCost - node.lbScenario[1]) + (
            #                     node.lbDensityCovered - node.lbScenario[1]) * (lbCoveredProbNew - node.lbCoveredProb)
            #             node.lbCoveredProb =lbCoveredProbNew
            #             node.ubRoute -= (routeCost[0] - scenarioCost) * scenarioProb
            #             node.exactValueProb += scenarioProb
            #             node.exactValue += scenarioCost * scenarioProb
            #lastEnteranceDictionary is the last element of Route Cost that was checked for this node
            node.lastEnteranceDictionary = next(reversed(routeCost))

            '''

            for alpha in range(min(num_may_deviate, node.numDefined+1)):
                for combination in itertools.combinations(list(range(1,node.numDefined+1)), alpha):
                    scenario = node.withDiscountID
                    # make a scenario given policy and combination of deviated nodes
                    for offset in combination:
                        mask = (1 << (offset - 1))
                        if policy & mask:
                            scenario = scenario & ~mask
                        else:
                            scenario = scenario | mask
                    # check if scenario is already in node
                    if scenario not in node.tspDict[gamma]:
                        scenarioCost = routeCost.get(scenario)
                        if scenarioCost:
                            node.tspDict[gamma].append(scenario)
                            scenarioProb = probability.scenarioProb(scenario, node.withDiscountID, node.layer,
                                                                      n, Bab.instance.p_dev)

                            coveredProbNew = max(0, node.lbCoveredProb - scenarioProb)
                            node.lbRoute += scenarioProb * (scenarioCost - node.lbScenario) + (
                                        node.lbDensityCovered - node.lbScenario) * (coveredProbNew - node.lbCoveredProb)
                            node.lbCoveredProb = coveredProbNew

                            node.ubRoute -= (routeCost[0] - scenarioCost) * scenarioProb
                            node.exactValueProb +=  scenarioProb
                            node.exactValue += scenarioCost*scenarioProb
                            assert node.lbRoute >= 0, 'loop3 Dict node lbRoute <0'

                            #num_scenarios_gamma += 1
                            #cum_cost_gamma += scenarioCost
                '''
                #scenarioProbLb = (1 - p_dev) ** (n - gamma) * p_dev ** (gamma - len(node.setNotGivenDiscount))
                #node.lbRoute += (cum_cost_gamma - num_scenarios_gamma * node.lbScenario) * scenarioProbLb
                #node.ubRoute -= (routeCost[0] * num_scenarios_gamma - cum_cost_gamma) * scenarioProbLb
                #node.exactValueProb += num_scenarios_gamma * scenarioProbLb
        # TODO recalculate LB so that we do not have scenarios with gamma in LB

    '''
            for numDeviated in range(num_may_deviate+1):
                all_combinations = itertools.combinations(setMayVary, numDeviated)
                for combination in all_combinations:
                    scenario = node.withDiscountID
                    #make a scenario given policy and combination of deviated nodes
                    for offset in combination:
                        mask = ~(1 << (offset-1))
                        scenario = scenario & mask
                    #check if scenario is already in node
                    if scenario not in node.:
                        scenarioCost = routeCost.get(scenario)
                        if scenarioCost:
                            scenarioProbLb = (1 - p_dev) ** bitCount(node.withDiscountID & scenario) * p_dev ** (
                                    bitCount(node.withDiscountID ^ scenario) + n - layer)
                            node.exactValueProb += scenarioProbLb
                            node.lbRoute += scenarioProbLb * (scenarioCost - node.lbScenario)
                            node.ubRoute -= scenarioProbLb * (routeCost[0] - scenarioCost)

            '''


    '''
        if status_not_changed:
        #for scenario in list(routeCost.keys())[node.lastAddedScenario:node.withDiscountID+1]:
        #for scenario in routeCost:
            if node.lastAddedScenario <=  scenario and  scenario <=node.withDiscountID:
                newLastAddedScenario += 1
                if not (~node.withDiscountID & scenario):
                    if scenario not in node.:

                        scenarioProbLb = (1-p_dev) ** bitCount(node.withDiscountID & scenario) * p_dev ** (
                            bitCount(node.withDiscountID ^ scenario) + n - (node.withDiscountID + node.noDiscountID).bit_length())

                        scenarioCost = routeCost[scenario]
                        node..append(scenario)
                        node.exactValueProb += scenarioProbLb
                        node.lbRoute += scenarioProbLb * (scenarioCost - node.lbScenario)
                        node.ubRoute -= scenarioProbLb * (routeCost[0]- scenarioCost)
        node.lastAddedScenario = newLastAddedScenario
    '''


def updateBoundsWithNewTSPs(Bab, node):

    n = Bab.instance.NR_CUST
    setGivenDiscount = node.setGivenDiscount
    setMayVary = node.setGivenDiscount
    p_home = Bab.instance.p_home
    continue_flag = True
    ubScenario = Bab.instance.routeCost[0]
    added = 0
    gap_old = (node.ubRoute - node.lbRoute) / node.ubRoute
    # limit the number of deviated since function may be too slow for small probabilities
    #num_may_deviate = Bab.instance.maxNumDeviated[len(setGivenDiscount)]
    if (2 ** n - 1 - node.noDiscountID) in Bab.instance.routeCost:
        node.updateLbScenario(Bab.instance.routeCost[2 ** n - 1 - node.noDiscountID], Bab.instance.p_home, n)

    if node.exactValueProb < 1:
        num_dev_start = node.lastNumDeviatedNewTSP
        # alpha is the number of deviated customers:
        # for alpha in range(num_dev_start, n+1):
        #     node.lastNumDeviatedNewTSP = alpha
        #     for combination in itertools.combinations(list(range(1, n+ 1)), alpha):
        #         scenario = node.withDiscountID
        #         # make a scenario given policy and combination of deviated nodes
        #         for offset in combination:
        #             mask = (1 << (offset - 1))
        #             if node.withDiscountID & mask:
        #                 scenario = scenario & (~mask)
        #             else:
        #                 scenario = scenario | mask
        # if probability.scenarioPossible_2segm(scenario, node.withDiscountID, n, n):
        #     # check if scenario is already in node
        #     gamma = n - bitCount(scenario)
        #     if not node.tspDict.get(gamma):
        #         node.tspDict[gamma] = []
        #     if scenario not in node.tspDict[gamma]:
        #
        #         if scenario not in Bab.instance.routeCost:
        #             scenarioCost = Bab.TSPSolver.tspCost(scenario)
        #             Bab.instance.routeCost[scenario] = scenarioCost
        #             added += 1
        #         else:
        #             scenarioCost = Bab.instance.routeCost[scenario]
        # scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, n, n,
        #                                               Bab.instance.p_pup_delta)
        # node.tspProbDict[scenario] = scenarioProb
        # node.tspDict[gamma].append(scenario)
        # node.exactValueProb += scenarioProb
        # node.exactValue += scenarioCost * scenarioProb
        # lbCoveredProbNew = max(0, node.lbCoveredProb - scenarioProb)
        # node.lbRoute += scenarioProb * (scenarioCost - node.lbScenario[1]) + (
        #         node.lbDensityCovered - node.lbScenario[1]) * (lbCoveredProbNew - node.lbCoveredProb)
        # node.lbCoveredProb = lbCoveredProbNew
        # node.ubRoute -= scenarioProb * (ubScenario - scenarioCost)
        #
        # # add new scenario to the dict of scenarios in the lbCovered, iff the scenario has high SEtWEIGHTCOVERED
        # if node.lbCoveredWeightBest < setWeightCovered(node.withDiscountID, scenario, n, p_home) * scenarioCost:
        #     node.lbCoveredWeightBest = setWeightCovered(node.withDiscountID, scenario, n, p_home) * scenarioCost
        #     node.lbScenarios[scenario] = scenarioCost
        for gamma in range(n - len(setMayVary), n + 1):
            if gamma not in node.tspDict:
                node.tspDict[gamma] = []
            if len(node.tspDict[gamma]) == comb(len(node.setGivenDiscount), gamma - len(node.setNotGivenDiscount)):
                continue

            # scenarioProb = (1 - p_dev) ** (n - gamma) * p_dev ** (gamma - len(node.setNotGivenDiscount))
            num_scenarios_gamma = 0
            cum_cost_gamma = 0
            for combination in itertools.combinations(setMayVary, gamma - len(node.setNotGivenDiscount)):
                scenario = node.withDiscountID
                # make a scenario given policy and combination of deviated nodes
                for offset in combination:
                    mask = ~(1 << (offset - 1))
                    scenario = scenario & mask
                # check if scenario is already in node
                if scenario not in node.tspDict[gamma]:
                    if scenario not in Bab.instance.routeCost:
                        scenarioCost = Bab.TSPSolver.tspCost(scenario)
                        Bab.instance.routeCost[scenario] = scenarioCost
                    else:
                        scenarioCost = Bab.instance.routeCost[scenario]

                    if gamma == n - len(setMayVary):
                        node.updateLbScenario(scenarioCost, Bab.instance.p_home, n)

                    node.tspDict[gamma].append(scenario)
                    added += 1

                    scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, n, n,
                                                                        Bab.instance.p_pup_delta)
                    node.tspProbDict[scenario] = scenarioProb
                    node.exactValueProb += scenarioProb
                    node.exactValue += scenarioCost * scenarioProb
                    lbCoveredProbNew = max(0, node.lbCoveredProb - scenarioProb)
                    node.lbRoute += scenarioProb * (scenarioCost - node.lbScenario[1]) + (
                                            node.lbDensityCovered - node.lbScenario[1]) * (lbCoveredProbNew - node.lbCoveredProb)
                    node.lbCoveredProb = lbCoveredProbNew
                    node.ubRoute -= scenarioProb * (ubScenario - scenarioCost)

                    #add new scenario to the dict of scenarios in the lbCovered, iff the scenario has high SEtWEIGHTCOVERED
                    if node.lbCoveredWeightBest < setWeightCovered(node.withDiscountID, scenario, n, p_home)*scenarioCost:
                        node.lbCoveredWeightBest =  setWeightCovered(node.withDiscountID, scenario, n, p_home)*scenarioCost
                        node.lbScenarios[scenario] = scenarioCost

                if False:
                    #print("added new scenario",bin(scenario))
                    true_exact_cost = 0
                    true_exact_prob = 0
                    for gamma in node.tspDict:
                        for scenario in node.tspDict[gamma]:
                            scenario_prob = probability.scenarioProb(scenario, node.withDiscountID,
                                                                     node.layer, Bab.instance.NR_CUST,
                                                                     Bab.instance.p_home, Bab.instance.p_pup)
                            true_exact_prob += scenario_prob
                            true_exact_cost += scenario_prob * Bab.instance.routeCost[scenario]
                            # print("here", bin(scenario), "true scenario_prob", scenario_prob,"in dict", nextNode.tspProbDict[scenario])
                    if true_exact_prob > node.exactValueProb + constants.EPS or true_exact_prob < node.exactValueProb - constants.EPS:
                        print(" node.exactValueProb", node.exactValueProb, true_exact_prob)


                if (node.ubRoute - node.lbRoute) / node.ubRoute <= 0.9 * gap_old or \
                        (node.lbRoute + node.lbExpDiscount >= Bab.bestUb) or (
                        node.ubRoute + node.ubExpDiscount < Bab.bestNode.lbVal()) or (added > 100):
                    continue_flag = False
                    #node.lastNumDeviatedNewTSP -=1
                    break
            if not continue_flag:
                break

        # for gamma in range( n-len(setMayVary), n+1):
        #    if gamma not in node.tspDict:
        #        node.tspDict[gamma] = []
        #    if len(node.tspDict[gamma]) == comb(len(node.setGivenDiscount), gamma-len(node.setNotGivenDiscount)):
        #        continue
        # alpha is the number of deviated customers,
        # f customers are added to the route,
        # (alpha-f) customers are removed from the route


def updateBoundsWithNewTSPsHeuristic(Bab, node):
    #if node.withDiscountID==Bab.rs_policy or  node.withDiscountID==0 or Bab.bestNode.withDiscountID==Bab.rs_policy or Bab.bestNode.withDiscountID==0:
    #    sample_size = constants.HEURISTIC_SAMPLE_SIZE_RS
    #else:
    sample_size = constants.HEURISTIC_SAMPLE_SIZE
    estimation_cost_by_sampling, Bab.instance.routeCost  = sampleAverageApproximation_PoissonBinomial_1sample(instance = Bab.instance,
                     policy = node.withDiscountID,  solverType =Bab.TSPSolver.solverType, sample_size =  sample_size, routeCosts=Bab.instance.routeCost, solver = Bab.TSPSolver)

    #if 2 ** Bab.instance.NR_CUST < constants.HEURISTIC_SAMPLE_SIZE:
    #    estimation_cost_by_sampling, Bab.instance.routeCost = one_policy_cost_estimation(instance = Bab.instance, policy = node.withDiscountID, routeCosts=Bab.instance.routeCost)
    #else:
    #    estimation_cost_by_sampling, Bab.instance.routeCost  = sampleAverageApproximation_PoissonBinomial_1sample(instance = Bab.instance,
    #                                        policy = node.withDiscountID, sample_size = constants.HEURISTIC_SAMPLE_SIZE, routeCosts=Bab.instance.routeCost)
    node.exactValueProb = 1

    node.lbCoveredProb = 0
    node.lbRoute = estimation_cost_by_sampling[0]-node.lbExpDiscount
    node.ubRoute = node.lbRoute
    node.exactValue = node.lbRoute
    node.lbScenario[1]  = node.lbRoute


def updateBoundsWithImportanceSampling(Bab, node,sampleScenarioID, sampleScenarioCost):
    estimation_cost_by_sampling = importanceSampling( Bab.instance, node.withDiscountID,0, sampleScenarioID, sampleScenarioCost)

    node.exactValueProb = 1

    node.lbCoveredProb = 0
    node.lbRoute = estimation_cost_by_sampling-node.lbExpDiscount
    node.ubRoute = node.lbRoute
    node.exactValue = node.lbRoute
    node.lbScenario[1]  = node.lbRoute


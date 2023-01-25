import os
import sys
from pathlib import Path
import copy
import heapq
import math
from itertools import combinations

from time import process_time
path_to_model = os.path.join((Path(os.path.abspath(__file__)).parents[2]), "model")
sys.path.insert(1, path_to_model)
from Node import Node

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]), "util")
sys.path.insert(3, path_to_util)
import constants
import probability

path_to_stochastic = os.path.join((Path(os.path.abspath(__file__)).parents[2]), "algorithms", "stochastic")
sys.path.insert(4, path_to_stochastic)
from sample_average import samplingDeviated
from sample_average import average_error
from sample_average import sampleAverageApproximation

path_to_bab = os.path.join((Path(os.path.abspath(__file__)).parents[2]), "algorithms", "exact", "bab")
sys.path.insert(2, path_to_bab)
from BABTree import BranchAndBound
from BranchAndBound import timer

from BoundsCalculation import updateByInsertionCost

import BoundsCalculation
from BAB_super_class import BAB_super_class
from BoundsCalculation import updateBoundsFromDictionary
from BoundsCalculation import recalculateLbCovered

prefix = "tag: "

class BranchAndBoundSampling(BranchAndBound):

    def runBranchAndBound(self, useTheorem):
        # Save the current time to a variable ('t')
        t = process_time()
        openNodes = self.openNodes

        # Visit all nodes until no nodes are left to branch on ( or a time limit is reached)
        start_time = process_time()
        time, lbPrint, ubPrint = [], [], []
        first_meet_time = {}

        best_old = self.bestNode
        num_tspSpecial = []
        optimalityFlag = True

        while (self.bestNode.ubVal() - self.bestNode.lbVal()) / self.bestNode.lbVal() > constants.EPSILON + constants.EPS:
            if self.bestNode.withDiscountID not in first_meet_time:
                first_meet_time[self.bestNode.withDiscountID] = process_time()
            if (process_time() - start_time) > constants.HEURISTIC_TIME_LIMIT:
                optimalityFlag = False
                break
            self.openNodes.push(self.bestNode, self.bestNode.ubRoute)
            while not openNodes.empty():

                if self.bestNode.withDiscountID not in first_meet_time:
                    first_meet_time[self.bestNode.withDiscountID] = process_time()
                if (process_time() - start_time) > constants.HEURISTIC_TIME_LIMIT:
                    optimalityFlag = False
                    break
                nextNode = openNodes.pop()

                #print(nextNode, round(nextNode.lbRoute, 4), round(nextNode.ubRoute, 4), round(nextNode.lbVal(), 4),
                #      round(nextNode.ubVal(), 4),nextNode.exactValueProb, nextNode.lbScenario, '       ',
                #      self.bestNode, round(self.bestNode.lbVal(), 4), round(self.bestNode.ubVal(), 4),  self.bestNode.exactValueProb)

                #assert nextNode.exactValueProb < 1 + constants.EPS, 'next node P > 1'
                #assert self.bestNode.exactValueProb < 1 + constants.EPS, 'best node P>1'
                if self.isTerminalNode(nextNode):
                    print(nextNode, " Terminated as this is the best policy")
                    break
                # elif useTheorem:
                elif self.canFathom(nextNode):
                    nextNode.fathomed()

                elif self.canBranch(nextNode):
                    # num_tspSpecial.append(nextNode.tspSpecial)
                    # Else, we branch on the node
                    self.branch(nextNode, self.setCustomerToBranch(nextNode))
                #assert nextNode.exactValueProb < 1 + constants.EPS, 'next node P > 1'


        print(prefix, "2** number of customers ", 2 ** self.instance.NR_CUST)
        print(prefix, "Number of calculates TSPs: ", len(self.instance.routeCost))

        if optimalityFlag:
            print(prefix, "Best Policy by Branch and Bound: ", self.bestNode, self.bestNode.lbVal(),
                  self.bestNode.ubVal())
            print(prefix, 'optimal: True')
        else:
            babExpValueLB, babExpValueUB = sampleAverageApproximation(self.instance, self.bestNode.withDiscountID)
            print(prefix, "Best Policy by Branch and Bound: ", self.bestNode, babExpValueLB, babExpValueUB)
            print(prefix, 'optimal: False')

        print(prefix)
        print(prefix, "Number of nodes in BAB tree: ", self.nrNodes)
        print(prefix)
        print(prefix, "Time of First Meet of Optimum) ",
              timer(start_time, first_meet_time[self.bestNode.withDiscountID]))
        time.append(process_time() - start_time)
        ubPrint.append(self.bestUb)
        return self.bestNode.withDiscountID, time, num_tspSpecial, ubPrint

    # Check whether the node can be fathomed, i.e.whether we can prune the node
    # skip node
    def canFathom(self, node):
        def exploreNode():
            #print("in explore node")
            nonlocal node
            n = self.instance.NR_CUST
            p_dev = self.instance.deviationProbability
            discount = self.instance.FLAT_RATE_SHIPPING_FEE
            oldNumDiscToAdd = n

            numCustToVisit = max(len(node.setNotGivenDiscount),
                                 n - math.floor((self.bestUb - node.lbScenario) / (discount * (1 - p_dev))))
            if numCustToVisit == 0:
                numCustToVisit += 1

            numVisitTemp = numCustToVisit

            while True:
                if numCustToVisit < n:
                    possibleDiscounts = list(range(n, node.layer, -1)) + list(node.setGivenDiscount)[::-1]
                    for combination in combinations(possibleDiscounts, n - numVisitTemp):
                        newLb = 0
                        for i in combination:
                            newLb += 2 ** (i - 1)
                        if newLb not in self.instance.routeCost:
                            self.instance.routeCost[newLb] = self.TSPSolver.tspCost(newLb)

                        else:
                            if  self.instance.routeCost[newLb] == node.lbScenario:
                                continue

                        if newLb == (2 ** n - 1 - node.noDiscountID):
                            lbImprove = node.updateLbScenario(self.instance.routeCost[newLb],
                                                              self.instance.deviationProbability)
                        else:
                            node.lbCalculated[newLb] = self.instance.routeCost[newLb]
                            coveredProbNew, costCoveredNew = recalculateLbCovered(p_dev, node)
                            lbImprove = node.updateLbCovered(coveredProbNew, costCoveredNew)
                        #print("newLb", bin(newLb)[2:], self.instance.routeCost[newLb] )
                        if lbImprove > max(0.05 * (node.ubRoute - node.lbRoute),
                                           constants.EPSILON * self.bestNode.lbVal()):
                            oldnumCustToVisit = numCustToVisit
                            numCustToVisit = max(
                                n - math.floor((self.bestUb - node.lbScenario) / (discount * (1 - p_dev))),
                                len(node.setNotGivenDiscount))
                            if numCustToVisit > oldnumCustToVisit:
                                numVisitTemp = numCustToVisit
                                break
                        else:
                            return False
                    numVisitTemp += 1
                    if numVisitTemp >= n:
                        return False
                else:
                    return False


        if self.bestNode == node:
            if (node.ubVal() - node.lbVal()) > constants.EPSILON * self.bestNode.lbVal():
                updateBoundsFromDictionary(self, node)
                if self.isLeaf(node):
                    updateBoundsWithNewTSPs(self, node)
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                else:
                    exploreNode()
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
            return False

        elif node.lbVal() > self.bestUb:
            return True

        else:
            if node.layer <= self.instance.NR_CUST:
                self.fathomByLayerSibling(node)
                if node.fathomedState:
                    return True
            updateBoundsFromDictionary(self, node)
            if node.fathomedState:
                return True

            while True:
                if node.lbVal() >= self.bestUb:
                    return True

                # fathom by bound (found new Best Node): if current precision is small, then fathom, else - branch on the best node
                elif (node.ubVal() < self.bestNode.ubVal()):
                    self.openNodes.push(self.bestNode, self.bestNode.priority())

                    self.bestNode = node
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                    if not self.isLeaf(node):
                        exploreNode()
                        self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                        return False
                    else:  # HERE we change from True to False
                        return False

                # epsilon optimality
                elif (self.bestNode.ubVal()- node.lbVal()) <= constants.EPSILON*node.lbVal() :
                #elif (node.ubVal() - node.lbVal()) <= constants.EPSILON * node.lbVal():

                    return True


                elif (node.ubVal() - node.lbVal()) < (self.bestNode.ubVal() - self.bestNode.lbVal()):
                    self.openNodes.push(self.bestNode, 0)
                    if self.isLeaf(node):
                        self.openNodes.push(node, node.priority())
                    return False
                else:
                    if not self.isLeaf(node):
                        if not exploreNode():
                            return False
                    else:
                        updateBoundsWithNewTSPs(self, node)

    def canFathomByTheoremCliques(self, node):
        discount = self.instance.FLAT_RATE_SHIPPING_FEE
        if node is self.root:
            diff_customer = 0
        else:
            diff_customer = (node.withDiscountID + node.noDiscountID).bit_length() - 1

        if node.withDiscountID & (1 << diff_customer):
            for other_customer in range(diff_customer):
                if not node.withDiscountID & (1 << other_customer):
                    if self.instance.distanceMatrix[(diff_customer + 1, other_customer + 1)] <= discount / 2:

                        return True
        else:
            for other_customer in range(diff_customer):
                if node.withDiscountID & (1 << other_customer):
                    if self.instance.distanceMatrix[(diff_customer + 1, other_customer + 1)] <= discount / 2:
                        return True
        return False




def samplingDeviatedScenarios(Bab, gamma, node):
    #print("here")
    p_dev = Bab.instance.deviationProbability
    average_cost = 0
    v = gamma - len(node.setNotGivenDiscount)
    sample = []
    limit_ss = comb(len(node.setGivenDiscount), v)
    weight = ((1 - p_dev) ** (len(node.setGivenDiscount) - v) * p_dev ** v) * comb(len(node.setGivenDiscount), v)

    cumulative_sum = 0
    sampleCost = []
    for i in range(constants.NUMBER_OF_SAMPLES):
        sampleCostTemp, routeCost = samplingDeviated(Bab.instance.routeCost, node.setGivenDiscount, v, node.withDiscountID, Bab.TSPSolver)
        sampleCost.append(sampleCostTemp)
    average, error_mean = average_error(sampleCost, constants.NUMBER_OF_SAMPLES)
    average_cost = average * weight
    average_cost_error = error_mean * weight

    return average_cost, weight

def updateBoundsWithNewTSPs(Bab, node):
    #print("in update by new tsp", node, node.exactValueProb)
    #for gamma in node.tspDict:
        #print(gamma, [bin(i)[2:] for i in node.tspDict[gamma]])
    n = Bab.instance.NR_CUST
    setMayVary = node.setGivenDiscount
    p_dev = Bab.instance.deviationProbability
    ubScenario = Bab.instance.routeCost[0]
    added = 0
    gap_old = (node.ubRoute - node.lbRoute) / node.ubRoute

    if (2 ** n - 1 - node.noDiscountID) in Bab.instance.routeCost:
        node.updateLbScenario(Bab.instance.routeCost[2 ** n - 1 - node.noDiscountID], Bab.instance.deviationProbability)

    if node.exactValueProb < 1:
        for gamma in range(n - len(setMayVary), n + 1):
            #print("gamma", gamma)
            if gamma not in node.tspAverageCost:
                #print("here need to calculate average")
                if gamma in node.tspDict:
                    #if len(node.tspDict[gamma]) == comb(len(node.setGivenDiscount), n - gamma):
                    if comb(len(node.setGivenDiscount), gamma - len(
                            node.setNotGivenDiscount)) ==  len(node.tspDict[gamma]):
                        #print("already have full from dict")
                        #print([bin(i)[2:] for i in node.tspDict[gamma]])
                        continue

                # probability of any scenario that deviate in gamma customers
                scenarioProb = (1 - p_dev) ** (n - gamma) * p_dev ** (gamma - len(node.setNotGivenDiscount))
                cost_calculated = 0
                # check what we already calculated and substract this probability and weight from node metrics
                num_calculated = 0

                if comb(len(node.setGivenDiscount), gamma - len(node.setNotGivenDiscount)) < constants.SAMPLE_SIZE * constants.NUMBER_OF_SAMPLES:
                    if gamma not in node.tspDict:
                        node.tspDict[gamma] = []

                    for combination in itertools.combinations(setMayVary, gamma - len(node.setNotGivenDiscount)):
                        scenario = node.withDiscountID
                        # make a scenario given policy and combination of deviated nodes
                        for offset in combination:
                            mask = ~(1 << (offset - 1))
                            scenario = scenario & mask
                        # if node.withDiscountID==4:
                        # print("scenario NEW", bin(scenario)[2:])
                        # check if scenario is already in node
                        if scenario not in node.tspDict[gamma]:
                            if scenario not in Bab.instance.routeCost:
                                scenario_cost = Bab.TSPSolver.tspCost(scenario)
                                Bab.instance.routeCost[scenario] = scenario_cost
                            else:
                                scenario_cost = Bab.instance.routeCost[scenario]
                            node.tspDict[gamma].append(scenario)
                            added += 1
                            node.exactValueProb += scenarioProb
                            node.lbRoute += scenarioProb * (scenario_cost - node.lbScenario)
                            node.ubRoute -= scenarioProb * (ubScenario - scenario_cost)

                            coveredProbNew, costCoveredNew = recalculateLbCovered(Bab.instance.deviationProbability,
                                                                                      node)
                            lbImprove = node.updateLbCovered(coveredProbNew, costCoveredNew)

                            if (node.ubRoute - node.lbRoute) / node.ubRoute <= 0.9 * gap_old or \
                                        (node.lbRoute + node.lbExpDiscount >= Bab.bestUb) or (
                                        node.ubRoute + node.ubExpDiscount < Bab.bestNode.lbVal()) or added > 100:
                                break

                else:
                    # weighted cost and weight of all scenarios that deviate in gamma customers
                    node.tspAverageCost[gamma], weight = samplingDeviatedScenarios(Bab, gamma, node)
                    if gamma in node.tspDict:
                        for scenario in node.tspDict[gamma]:
                            cost_calculated += Bab.instance.routeCost[scenario]
                        cost_calculated *= scenarioProb
                        num_calculated = len(node.tspDict[gamma])

                    node.exactValueProb += -num_calculated*scenarioProb+weight

                    coveredProbNew, costCoveredNew = recalculateLbCovered(Bab.instance.deviationProbability, node)
                    lbImprove = node.updateLbCovered(coveredProbNew, costCoveredNew)
                    #print(node.lbRoute, node.lbExpDiscount, coveredProbNew, costCoveredNew, node.exactValueProb)
                    node.lbRoute += node.tspAverageCost[gamma] - cost_calculated + node.lbScenario*( num_calculated *scenarioProb - weight)
                    node.ubRoute += node.tspAverageCost[gamma] - cost_calculated + ubScenario * ( num_calculated *scenarioProb - weight)
                    #print(node.lbRoute,node.lbVal(), node.ubRoute, node.ubVal(),  node.lbExpDiscount)

            if (node.ubRoute - node.lbRoute) / node.ubRoute <= 0.9 * gap_old or \
                    (node.lbRoute + node.lbExpDiscount >= Bab.bestUb) or (
                    node.ubRoute + node.ubExpDiscount < Bab.bestNode.lbVal()) or added > 100:
                break


def updateBoundsFromDictionary(Bab, node):
    #print("in dict before", node, node.exactValueProb)
    #for gamma in node.tspDict:
    #    print(gamma, [bin(i)[2:] for i in node.tspDict[gamma]])

    if node.parent is not None:
        if node is not Bab.bestNode:
            status_not_changed = updateByInsertionCost(node, Bab)
        else:
            status_not_changed = True

        if status_not_changed:
            routeCost = Bab.instance.routeCost
            #newLastAddedScenario = node.lastAddedScenario
            p_dev = Bab.instance.deviationProbability
            n = Bab.instance.NR_CUST

            # limit the number of deviated since function may be too slow for small probabilities
            num_may_deviate = Bab.instance.maxNumDeviated[len(node.setGivenDiscount)]

            if  (2 ** n - 1 - node.noDiscountID) in routeCost:
                node.updateLbScenario( routeCost[2 ** n - 1 - node.noDiscountID],Bab.instance.deviationProbability)
            #print(node)
            # gamma is the number of visited customers
            for gamma in range( n-len(node.setGivenDiscount), min(n+1, num_may_deviate+len(node.setNotGivenDiscount)+n-node.layer)):
                if gamma not in node.tspAverageCost:
                    #print("gamma in dict", gamma)
                    if gamma not in node.tspDict:
                        node.tspDict[gamma] = []
                    if len(node.tspDict[gamma]) == comb(len(node.setGivenDiscount), n - gamma):
                        continue

                    num_scenarios_gamma = 0
                    cum_cost_gamma = 0

                    for combination in itertools.combinations(node.setGivenDiscount, gamma-len(node.setNotGivenDiscount)-n+node.layer):
                        scenario = node.withDiscountID
                        #make a scenario given policy and combination of deviated nodes
                        for offset in combination:
                            mask = ~(1 << (offset-1))
                            scenario = scenario & mask
                        if scenario not in node.tspDict[gamma]:
                            scenarioCost = routeCost.get(scenario)
                            if scenarioCost:

                                node.tspDict[gamma].append(scenario)
                                num_scenarios_gamma += 1
                                cum_cost_gamma +=scenarioCost


                    scenarioProbLb = (1 - p_dev) ** (n-gamma) * p_dev ** (gamma - len(node.setNotGivenDiscount))

                    node.lbRoute += (cum_cost_gamma - num_scenarios_gamma*node.lbScenario) * scenarioProbLb
                    node.ubRoute -= (routeCost[0]*num_scenarios_gamma - cum_cost_gamma) * scenarioProbLb
                    node.exactValueProb += num_scenarios_gamma* scenarioProbLb

        #print("in dict after", node, node.exactValueProb)
        #for gamma in node.tspDict:
        #    print(gamma, [bin(i)[2:] for i in node.tspDict[gamma]])

        coveredProbNew, costCoveredNew = recalculateLbCovered( Bab.instance.deviationProbability, node)
        lbImprove = node.updateLbCovered(coveredProbNew, costCoveredNew)

        #print("in dict after 2", node, node.exactValueProb)
        #for gamma in node.tspDict:
        #    print(gamma, [bin(i)[2:] for i in node.tspDict[gamma]])


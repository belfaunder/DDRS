import os
import sys
from pathlib import Path
import copy

import math
from itertools import combinations
from src.main.discount_strategy.algorithms.exact.bab import BoundsCalculation
from src.main.discount_strategy.algorithms.exact.bab.BAB_super_class import BAB_super_class
from BoundsCalculation import updateBoundsFromDictionary
from BoundsCalculation import updateBoundsFromLayer
from BoundsCalculation import recalculateLbCovered

import numpy as np
from time import process_time

from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.util.bit_operations import bitCount

path_to_data = constants.PATH_TO_DATA

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

prefix="tag: "

class BABExact(BAB_super_class):

    def runBranchAndBound(self):
        self.instance.set_enlarged_neighbourhood()
        # Save the current time to a variable ('t')
        t  = process_time()
        openNodes = self.openNodes

        #print("upperBoundNumberDiscount", self.upperBoundNumberDiscount)

        # Visit all nodes until no nodes are left to branch on ( or a time limit is reached)
        start_time = process_time()
        ubValRoot = self.root.ubVal()
        time, lbPrint, ubPrint= [], [], []

        first_meet_time = {}

        best_old = self.bestNode
        optimalityFlag = True
        self.openNodes.push(self.bestNode, self.bestNode.ubRoute)

        while (not openNodes.empty()):
            if self.bestNode.withDiscountID not in first_meet_time:
                first_meet_time[self.bestNode.withDiscountID] = process_time()
            if (process_time() - start_time) > constants.TIME_LIMIT:
                optimalityFlag = False
                break

            if (self.nrNodes % 50 == 0):
                if self.bestNode.exactValueProb < 0.95:
                    if best_old is self.bestNode:
                        if  self.isLeaf(self.bestNode):
                            BoundsCalculation.updateBoundsWithNewTSPs(self, self.bestNode)
                self.bestUb = min(self.bestUb,self.bestNode.ubVal() )
                best_old = self.bestNode

            # Select the next node to branch on. We assume that the priority queue
            # containing the open nodes maintains the nodes in sorted order
            nextNode = openNodes.pop()

            if self.isTerminalNode(nextNode):
                continue
            elif self.canFathom(nextNode):
                nextNode.fathomed()
            elif self.canBranch(nextNode):
                self.branch(nextNode, self.setCustomerToBranch(nextNode))
            if  openNodes.empty() and (not self.isLeaf(self.bestNode)):
                self.openNodes.push(self.bestNode, self.bestNode.ubRoute)

        print(prefix+ "Time_running,s: ", (process_time()-start_time))
        print(prefix+ "Number_nodes: ", self.nrNodes)
        print(prefix+ "Number_calculated_TSPs: ", len(self.instance.routeCost))
        if optimalityFlag:
            print(prefix+ 'Optimal: 1')
        else:
            print(prefix+ 'Optimal: 0')
        print(prefix+ 'Obj_val(ub): ', self.bestNode.ubVal(), 'Obj_val(lb): ', self.bestNode.lbVal())
        print(prefix+ 'Gap: ', max(0,round((self.bestNode.ubVal() - self.bestNode.lbVal())/self.bestNode.ubVal(),2)))

        try:
            print(prefix+ "Time_first_meet_optimum,s: ", (first_meet_time[self.bestNode.withDiscountID]-start_time))
        except:
            print(prefix + "Time_first_meet_optimum,s: ", (process_time()))
        print(prefix+ 'BestPolicy_bab_ID: ', self.bestNode.withDiscountID)
        print(prefix+ 'BestPolicy)bab: ', bin(self.bestNode.withDiscountID)[2:].zfill(self.instance.NR_CUST))

        return self.bestNode.withDiscountID, time, lbPrint, ubPrint
    # Check whether the node can be fathomed, i.e.whether we can prune the node
    # skip node
    def canFathom(self, node):
        #TODO: finish this function
        def exploreNode():

            nonlocal node
            n = self.instance.NR_CUST
            p_home = self.instance.p_home
            numCustToVisit = max(n - max(0,math.floor((self.bestUb- node.lbScenario[1]) / (max(0.001, np.amin(self.instance.shipping_fee[1:])
                        * np.amin(self.instance.p_pup_delta[1:]))))), len(node.setNotGivenDiscount))
            if numCustToVisit == 0:
                numCustToVisit += 1

            #calculate the smallest num of customers who was not visited s.t. the tspDisc does not have it
            numVisitTemp = numCustToVisit

            # retrun False if we cannot fathom the node,
            # retrun True if we can fathom
            # max possible numer of discounts ->n-node.layer + len(node.setGivenDiscount)
            # while numDiscTemp>= n-node.layer + len(node.setGivenDiscount):
            while True:
                if numCustToVisit < n:
                    # the list of customers who might get a discount, i.e. those who got a discount in current policy
                    # plus those about whom we do not have a decision yet
                    possibleDiscounts = list(range(n, node.layer ,-1)) + list(node.setGivenDiscount)[::-1]
                    #possibleDiscounts |= node.setGivenDiscount
                    #combinations = random_combination(possibleDiscounts, numDiscTemp)
                    #combinations_set = combinations(possibleDiscounts, numDiscTemp)

                    for combination in combinations(possibleDiscounts, n-numVisitTemp):
                        # create a scenario that offer a discount to  " n-numVisitTemp" number of customers.
                        # The idea is that this scenario may be not possible for some of child nodes,
                        # but it will give a true LB anyway
                        newLbScenario = 0
                        for i in combination:
                            newLbScenario += 1 << (i-1)
                        if newLbScenario not in self.instance.routeCost:
                            self.instance.routeCost[newLbScenario]= self.TSPSolver.tspCost(newLbScenario)
                        else:
                            if self.instance.routeCost[newLbScenario] == node.lbScenario:
                                continue
                        # the following  is true only for 2 segment model:
                        # if the new scenario is a true LB_route, then update Lb_route
                        # (initially all nodes have LB_route = 2**n - 1 or "visit all customers")
                        if newLbScenario == (2 ** n - 1 - node.noDiscountID):
                            lbImprove = node.updateLbScenario(self.instance.routeCost[newLbScenario],
                                                             self.instance.p_home,  n)
                        else:
                            node.lbScenarios[newLbScenario] = self.instance.routeCost[newLbScenario]
                            lbCoveredProbNew, lbDensityCoveredNew = recalculateLbCovered(p_home, node, n)
                            lbImprove = node.updateLbCovered(lbCoveredProbNew, lbDensityCoveredNew)

                        if lbImprove > max(0.005*(node.ubRoute - node.lbRoute), constants.EPSILON*self.bestNode.lbVal()):
                            oldnumCustToVisit = numCustToVisit

                            # it may happen that with new LB_route giving many discounts is not profitable,
                            # so we calculate the max number of discounts that is still may be profitable
                            numCustToVisit = max(
                                n - max(0,math.floor((self.bestUb- node.lbScenario[1]) / (max(0.001, np.amin(self.instance.shipping_fee[1:]) * np.amin(self.instance.p_pup_delta[1:]))))), len(node.setNotGivenDiscount))
                            if numCustToVisit > oldnumCustToVisit:
                                numVisitTemp = numCustToVisit
                                break
                        else:
                            #if increase is not suffucuent, stop adding new scenario and finish the function "explore the node"
                            return False
                    numVisitTemp += 1
                    if numVisitTemp >=n:
                        return False
                else:
                    return False

        # in node is the current BestNode, then update the bounds, and return False (negative answer to canFathom)
        if self.bestNode == node:
            if (node.ubVal() - node.lbVal()) > constants.EPSILON * self.bestNode.lbVal():
                updateBoundsFromDictionary(self, node)
                if self.isLeaf(node):
                    BoundsCalculation.updateBoundsWithNewTSPs(self, node)
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                else:
                    exploreNode()
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
            return False
        elif node.lbVal() > self.bestUb * (1 - constants.EPSILON):
            return True
        else:
            updateBoundsFromDictionary(self, node)
            if node.fathomedState:
                return True
            # compare the cost of a node with the best node
            cycle_iteration = 0
            while cycle_iteration<10:
                cycle_iteration+=1
                if node.lbVal() > self.bestUb * (1 - constants.EPSILON):
                    return True

                # fathom by bound (found new Best Node): if current precision is small, then fathom, else - branch on the best node
                elif (node.ubVal() < self.bestNode.ubVal()):
                    self.openNodes.push(self.bestNode,  self.bestNode.priority())
                    self.bestNode = node
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                    if not self.isLeaf(node):
                        # add info by new tsp and should branch
                        exploreNode()
                        self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                        return False
                    else:
                        return False

                # epsilon optimality
                elif (self.bestNode.ubVal() - node.lbVal()) <= constants.EPSILON * node.lbVal():
                    return True

                # if we cannot distinguish current node with the best node AND
                # current node has more presice bounds than the best node, then push BestNode in OpenNodes and
                # make current node to be the new BestNode
                elif (node.ubVal() - node.lbVal()) < (self.bestNode.ubVal() - self.bestNode.lbVal()):

                    if (self.lastCurrentNodes.count(node.withDiscountID) == 3) and (self.lastCurrentNodes.count(self.bestNode.withDiscountID) == 2) and self.isLeaf(node):
                        self.openNodes.push(node, node.priority()*2)
                    else:
                        self.openNodes.push(self.bestNode, 0)
                        if self.isLeaf(node):
                            self.openNodes.push(node, node.priority())
                    #self.bestNode = node
                    #self.bestUb = min(self.bestUb, self.bestNode.ubVal())

                    return False
                else:
                    if not self.isLeaf(node):
                        if not exploreNode():
                            return False
                    else:
                        # update the bounds for current node and repeat check
                        BoundsCalculation.updateBoundsWithNewTSPs(self, node)
            if cycle_iteration == 10:
                self.openNodes.push(node, node.priority())


from time import process_time
from src.main.discount_strategy.algorithms.exact.bab.BAB_super_class import BAB_super_class
from src.main.discount_strategy.algorithms.exact.bab import BoundsCalculation
from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import updateBoundsFromDictionary
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util.bit_operations import bitCount

prefix=constants.PREFIX
class BABHeuristic(BAB_super_class):
    def runBranchAndBound(self):
        self.upperBoundNumberDiscount = bitCount(self.rs_policy)
        openNodes = self.openNodes
        # Visit all nodes until no nodes are left to branch on ( or a time limit is reached)
        start_time = process_time()
        self.openNodes.push(self.bestNode, self.bestNode.ubRoute)
        self.instance.set_enlarged_neighbourhood()

        while (not openNodes.empty()):
            if (process_time() - start_time) > constants.HEURISTIC_TIME_LIMIT:
                break

            nextNode = openNodes.pop()
            if self.bestNode.fathomedState:
                self.bestNode = nextNode
                self.bestUb = self.bestNode.ubVal()
                self.openNodes.push(self.bestNode, self.bestNode.priority())

            elif self.isTerminalNode(nextNode):
                continue
            elif self.canFathomByTheoremCliques_Heuristic(nextNode):
                nextNode.fathomed()
            elif self.canFathom(nextNode):
                nextNode.fathomed()
            elif self.canBranch(nextNode):
                self.branch(nextNode, self.setCustomerToBranch(nextNode))
            if  openNodes.empty() and (not self.isLeaf(self.bestNode)):
                self.openNodes.push(self.bestNode, self.bestNode.ubRoute)

        print(prefix+ "Time_running,s: ", (process_time() - start_time))
        print(prefix+ "Number_nodes: ", self.nrNodes)
        print(prefix+ "Number_calculated_TSPs: ", len(self.instance.routeCost))
        print(prefix+ 'Obj_val(ub): ', self.bestNode.ubVal()/constants.SCALING_FACTOR, 'Obj_val(lb): ', self.bestNode.lbVal()/constants.SCALING_FACTOR)
        print(prefix+ 'BestPolicy_ID: ', self.bestNode.withDiscountID)
        print(prefix+ 'BestPolicy: ', bin(self.bestNode.withDiscountID)[2:].zfill(self.instance.NR_CUST))
        return self.bestNode.withDiscountID

    # Check whether the node can be fathomed, i.e.whether we can prune the node
    # skip node
    def canFathom(self, node):
        def exploreNode():
            nonlocal node
            if node.setNotGivenDiscount:
                for id in reversed(node.lbScenarios):
                    newScenario = node.lbScenarios[id][2] & ~node.noDiscountID
                    if newScenario != node.lbScenarios[id][2]:
                        if newScenario not in self.instance.routeCost:
                            routingCost = self.TSPSolver.tspCost(newScenario)
                            self.instance.routeCost[newScenario] = routingCost
                        else:
                            routingCost = self.instance.routeCost[newScenario]
                        node.lbRoute += node.lbScenarios[id][1] * (routingCost - node.lbScenarios[id][0])
                        node.lbScenarios[id][0] = routingCost
                        node.lbScenarios[id][2] = newScenario
            return False

        # in node is the current BestNode, then update the bounds, and return False (negative answer to canFathom)
        if self.bestNode == node:
            if (node.ubVal() - node.lbVal()) > constants.EPSILON_H * self.bestNode.lbVal():
                updateBoundsFromDictionary(self, node)
                if self.isLeaf(node):
                    BoundsCalculation.updateBoundsWithNewTSPsHeuristic(self, node)
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                else:
                    exploreNode()
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
            return False

        elif node.lbVal() > self.bestUb*(1-constants.EPSILON_H):
            return True
        else:

            updateBoundsFromDictionary(self, node)
            if node.fathomedState:
                return True
            cycle_iteration = 0
            #while True:
            while cycle_iteration<10:
                cycle_iteration+=1
                if node.lbVal() > self.bestUb*(1-constants.EPSILON_H):
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
                elif (self.bestNode.ubVal() - node.lbVal()) <= constants.EPSILON_H * node.lbVal():
                    return True

                # if we cannot distinguish current node with the best node AND
                # current node has more presice bounds than the best node, then push BestNode in OpenNodes and
                # make current node to be the new BestNode
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
                        # update the bounds for current node and repeat check
                        BoundsCalculation.updateBoundsWithNewTSPsHeuristic(self, node)
            if cycle_iteration == 10:
                self.openNodes.push(node, node.priority())



from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import updateBoundsFromDictionary
from src.main.discount_strategy.algorithms.exact.bab.BAB_super_class import BAB_super_class
from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import updateBoundsWithNewTSPs
from time import process_time
from src.main.discount_strategy.util import constants

path_to_data = constants.PATH_TO_DATA

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

prefix=constants.PREFIX

class BABExact(BAB_super_class):

    def runBranchAndBound(self):
        self.instance.set_enlarged_neighbourhood()
        # Save the current time to a variable ('t')
        t  = process_time()
        openNodes = self.openNodes

        # Visit all nodes until no nodes are left to branch on ( or a time limit is reached)
        start_time = process_time()
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
                            updateBoundsWithNewTSPs(self, self.bestNode)
                self.bestUb = min(self.bestUb,self.bestNode.ubVal() )
                best_old = self.bestNode

            # Select the next node to branch on. We assume that the priority queue
            # containing the open nodes maintains the nodes in sorted order
            nextNode = openNodes.pop()

            #lbPrint.append( self.bestNode.lbVal() )
            #ubPrint.append(self.bestNode.ubVal())
            #time.append(process_time()-start_time)
            #print("\nnextNode", nextNode.withDiscountID, bin(nextNode.withDiscountID), nextNode.layer,
            #      nextNode.exactValueProb, nextNode.exactValue, nextNode.lbRoute,
            #   nextNode.lbVal(), nextNode.ubVal())
            #print("bestNode", bin(self.bestNode.withDiscountID), self.bestNode.withDiscountID,
            #      self.bestNode.exactValueProb, self.bestNode.lbRoute, self.bestNode.lbVal() ,self.bestNode.ubVal()  )

            # print(len(self.instance.routeCost))
            #for id in nextNode.lbScenarios:
            #    print(id, bin(id), nextNode.lbScenarios[id])
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
        print(prefix+ 'BestPolicy_bab: ', bin(self.bestNode.withDiscountID)[2:].zfill(self.instance.NR_CUST))

        return self.bestNode.withDiscountID, time, lbPrint, ubPrint
    # Check whether the node can be fathomed, i.e.whether we can prune the node
    # skip node
    def canFathom(self, node):
        def exploreNode():

            #DOMINANCE_CHECK_REMOVED
            #return False
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

                        #check if the new lbScenario costs less than the lbScenarios, where less pups are visited
                        new_lbScenarioRoutingCost = routingCost
                        for pup in  self.instance.pups:
                            if (1 << pup.number) & id:
                                min_probability_to_visit_pup = 0
                                for cust_id in pup.closest_cust_id:
                                   if  (1<<cust_id) & node.withDiscountID:
                                       min_probability_to_visit_pup = 1
                                       break
                                if not min_probability_to_visit_pup:
                                    id_less_pups_visited = id & ~(1 << pup.number)
                                    new_lbScenarioRoutingCost = min(new_lbScenarioRoutingCost, node.lbScenarios[id_less_pups_visited][0])
                        node.lbRoute +=  node.lbScenarios[id][1] *(new_lbScenarioRoutingCost - node.lbScenarios[id][0])
                        node.lbScenarios[id][0] = new_lbScenarioRoutingCost
                        node.lbScenarios[id][2] = newScenario
                cost_should_be = 0
                for id in node.lbScenarios:
                    cost_should_be+= node.lbScenarios[id][0]*node.lbScenarios[id][1]
            return False
        # in node is the current BestNode, then update the bounds, and return False (negative answer to canFathom)
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
        elif node.lbVal() > self.bestUb :
            #if node.layer < self.instance.NR_CUST:
            #    self.pruned_bounds_nonleaf += 1
            return True
        else:
            updateBoundsFromDictionary(self, node)

            if node.fathomedState:
                return True
            # compare the cost of a node with the best node
            cycle_iteration = 0
            while cycle_iteration<10:
                cycle_iteration+=1
                if node.lbVal() > self.bestUb:
                    #if node.layer < self.instance.NR_CUST:
                    #    self.pruned_bounds_nonleaf += 1
                    return True

                # fathom by bound (found new Best Node): if current precision is small, then fathom, else - branch on the best node
                elif (node.ubVal() < self.bestNode.ubVal()):
                    self.openNodes.push(self.bestNode,  self.bestNode.priority())
                    self.bestNode = node
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                    #if not self.isLeaf(node):
                        # add info by new tsp and should branch
                    exploreNode()
                    self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                    #    return False
                    #else:
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
                        updateBoundsWithNewTSPs(self, node)
            if cycle_iteration == 10:
                self.openNodes.push(node, node.priority())


#tree of BAB - the structure is used in both Exact algorithm and in heuristics
import collections
import numpy as np
import heapq
from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import recalculateLbCovered
#from BoundsCalculation import updateBoundsFromDictionary

from src.main.discount_strategy.model.Node import Node
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW

from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.util.bit_operations import bitCount

# Representation of a BB tree
# @author Joris Kinable

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def empty(self):
        if (len(self._queue) is 0):
            return True
        else:
            return False

    def length(self):
        return len(self._queue)



#shallow version of deepcopy: https://stackoverflow.com/questions/45858084/what-is-a-fast-pythonic-way-to-deepcopy-just-data-from-a-python-dict-or-list
_dispatcher = {}

def _copy_list(l, dispatch):
    ret = l.copy()
    for idx, item in enumerate(ret):
        cp = dispatch.get(type(item))
        if cp is not None:
            ret[idx] = cp(item, dispatch)
    return ret

def _copy_dict(d, dispatch):
    ret = d.copy()
    for key, value in ret.items():
        cp = dispatch.get(type(value))
        if cp is not None:
            ret[key] = cp(value, dispatch)

    return ret

_dispatcher[list] = _copy_list
_dispatcher[dict] = _copy_dict

def deepcopy(sth):
    cp = _dispatcher.get(type(sth))
    if cp is None:
        return sth
    else:
        return cp(sth, _dispatcher)



class BAB_super_class:

    def __init__(self, instance, solverType):
        self.instance = instance

        # gurobi model for TSP is stored in TSP object and reused
        # thus to enable warm start
        self.TSPSolver = TSPSolver(instance=instance, solverType=solverType)

        # add two scnenarios which function as a LB and UB
        scenario_0 = 0
        self.instance.routeCost[scenario_0] = self.TSPSolver.tspCost(scenario_0)

        scenario_1 = 2 ** instance.NR_CUST - 1
        self.instance.routeCost[scenario_1] = self.TSPSolver.tspCost(scenario_1)

        ubRoute = self.instance.routeCost[scenario_0]
        lbRoute = self.instance.routeCost[scenario_1]
        lbScenario = self.instance.routeCost[scenario_1]

        tspDict = {instance.NR_CUST:[scenario_0]}
        tspProbDict = {scenario_0:probability.scenarioProb_2segm(scenario_0, 0, 0, instance.NR_CUST, instance.p_pup_delta)}
        # Root node of the BB tree
        root = Node(parent=None,
                    lbScenario=[scenario_1,lbScenario],
                    lbRoute=lbRoute,
                    ubRoute=ubRoute,
                    withDiscountID=0,
                    noDiscountID=0,
                    lbExpDiscount=0,
                    ubExpDiscount=(instance.p_pup_delta.dot(instance.shipping_fee) ),#*constants.PROB_PLACE_ORDER,
                    tspDict=tspDict,
                    tspProbDict = tspProbDict,
                    lbScenarios={scenario_1:lbScenario},
                    ubScenarios = None,
                    lbCoveredProb=0,
                    lbDensityCovered = lbScenario,
                    ubDensityCovered = ubRoute,
                    exactValueProb = tspProbDict[scenario_0],
                    exactValue = tspProbDict[scenario_0]*self.instance.routeCost[scenario_0],
                    layer=0,
                    priorityCoef = 1,
                    lastEnteranceDictionary = scenario_0
                    )

        self.nodeLayers = {}
        self.root = root
        self.nrNodes = 1
        self.openNodes = PriorityQueue()
        self.lastCurrentNodes = collections.deque(maxlen=5)

        self.bestNode = root
        self.listNodes = []
        self.bestUb = ubRoute

        # in the worst case we can do nothing (do not offer any discount), which will cost route visiting all customers

        rsPolicyID, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)
        self.rs_policy = rsPolicyID
        self.upperBoundNumberDiscount = bitCount(self.rs_policy)
        #print(bin(rsPolicyID)[2:], "RS cost", rsValue)

        # TODO: this theorem is now not used
        #self.upperBoundNumerDiscount = bitCount(rsPolicyID)
        #self.rsCost = {scenario_0:ubRoute, instance.NR_CUST:lbScenario[1]}

        #print(rsPolicyID, bin(rsPolicyID))
        #for i in range(1, instance.NR_CUST):
        #    self.rs_policy[i], self.rsCost[i] = ring_star_deterministic_no_TW(instance= instance,
        #                                max_number_offered_discounts=i, discount=np.zeros(instance.NR_CUST + 1))

    def canBranch(self, node):
        if (node.ubVal() - node.lbVal()) <= constants.EPSILON * self.bestNode.lbVal():
            return False
        else:
            if self.isLeaf(node):
                return False
        return True

    def isLeaf(self, node):
        return node.layer == self.instance.NR_CUST

    def setCustomerToBranch(self, parent):
        if parent is self.root:
            diff_customer = 1
        else:
            diff_customer = (parent.withDiscountID + parent.noDiscountID).bit_length()+1
        return (diff_customer)


    # Method that implements the branching logic
    def branch(self, parent, diff_customer):
        #diff_customer starts form 1
        shipping_fee = self.instance.shipping_fee
        p_home = self.instance.p_home
        p_pup = self.instance.p_pup
        n = self.instance.NR_CUST
        # decide how to branch
        # calculate the new policy information
        # currently branch in alphabetical order, left branch gives discount, right - nodiscout

        noDiscountIDLeft = parent.noDiscountID
        withDiscountIDLeft = parent.withDiscountID + 2 ** (diff_customer-1)
        withDiscountIDRight = parent.withDiscountID
        noDiscountIDRight = parent.noDiscountID + 2 ** (diff_customer-1)

        lbScenarioLeft = deepcopy(parent.lbScenario)
        lbScenarioRight = [parent.lbScenario[0]- 2 ** (diff_customer-1)]

        ubRoute = self.instance.routeCost[0]

        if lbScenarioRight[0]  not in self.instance.routeCost:
            self.instance.routeCost[lbScenarioRight[0]] = self.TSPSolver.tspCost(lbScenarioRight[0])
        lbScenarioRight.append(self.instance.routeCost[lbScenarioRight[0]])
        lastEnteranceDictionary = self.root.lastEnteranceDictionary

        # CHANGESPROBABILITY
        lbExpDiscountLeft = parent.lbExpDiscount + (1 - p_home[diff_customer]) * shipping_fee[diff_customer]#*constants.PROB_PLACE_ORDER
        ubExpDiscountLeft = parent.ubExpDiscount
        lbExpDiscountRight = parent.lbExpDiscount
        ubExpDiscountRight = parent.ubExpDiscount - (1 - p_home[diff_customer]) * shipping_fee[diff_customer]#*constants.PROB_PLACE_ORDER

        #tspDictLeft = copy.deepcopy(parent.tspDict)
        #tspDictRight = copy.deepcopy(parent.tspDict)

        #TODO: seems that we do not use it when calculating the bounds inside brunching
        lbCoveredProbRight = 0
        lbCoveredProbLeft = 0
        lbDensityCoveredRight = 0
        lbDensityCoveredLeft = 0

        lbRouteLeft = parent.lbRoute
        ubRouteLeft = parent.ubRoute
        lbRouteRight = parent.lbRoute

        lbScenariosRight = deepcopy(parent.lbScenarios)
        lbScenariosLeft = deepcopy(parent.lbScenarios)
        # lbScenariosRight = {k: v for k, v in parent.lbScenarios.items() if parent.lbScenario <= v}
        # lbScenariosLeft = {k: v for k, v in parent.lbScenarios.items() if parent.lbScenario <= v}

        tspDictLeft = deepcopy(parent.tspDict)
        tspDictRight = deepcopy(parent.tspDict)
        tspProbDictRight = {}

        exactValProbRight = 0
        exactValueRight = 0

        tspProbDictLeft = deepcopy(parent.tspProbDict)
        exactValProbLeft = parent.exactValueProb
        exactValueLeft = parent.exactValue

        # thus to reduce the problem with multiplication of small numbers
        if parent.exactValueProb > constants.EPS:
            lastEnteranceDictionary = self.root.lastEnteranceDictionary
            exactValProbRight = parent.exactValueProb / p_home[diff_customer]
            exactValueRight = parent.exactValue / p_home[diff_customer]

            for i in parent.tspProbDict:
                tspProbDictRight[i] = parent.tspProbDict[i] / p_home[diff_customer]
        else:
            # lbRouteRight = lbScenarioRight * (1 - clbCveredProbRight) + lbCoveredProb *lbDensityCoveredRight

            for gamma in tspDictRight:
                # all elements up to and including the diff_customer equal 1
                # scenarioProb = (1 - p_dev) ** (n - gamma) * p_dev ** (gamma - len(parent.setNotGivenDiscount) - 1)
                for scenario in tspDictRight[gamma]:
                    if probability.scenarioPossible_2segm(scenario, withDiscountIDRight,  parent.layer + 1, n):
                        scenarioProbRight = probability.scenarioProb_2segm(scenario, withDiscountIDRight, parent.layer + 1, n,
                                                                  self.instance.p_pup_delta)
                        exactValProbRight += scenarioProbRight
                        exactValueRight += self.instance.routeCost[scenario] * scenarioProbRight
                        tspProbDictRight[scenario] = scenarioProbRight
                    # scenarioProbLeft = probability.scenarioProb_2segm(scenario, withDiscountIDLeft,
                    #                                             parent.layer + 1, n,
                    #                                             self.instance.p_pup_delta)
                    # exactValProbLeft += scenarioProbLeft
                    # exactValueLeft += self.instance.routeCost[scenario] * scenarioProbLeft
                    #
                    # tspProbDictLeft[scenario] = scenarioProbLeft

        lbRouteRight = exactValueRight + lbCoveredProbRight*lbDensityCoveredRight+ (1-exactValProbRight - lbCoveredProbRight)*lbScenarioRight[1]
        ubRouteRight = exactValueRight + (1-exactValProbRight) * ubRoute

        lbRouteLeft = exactValueLeft + lbCoveredProbLeft * lbDensityCoveredLeft + (
                    1 - exactValProbLeft - lbCoveredProbLeft) * lbScenarioLeft[1]
        ubRouteLeft = exactValueLeft + (1 - exactValProbLeft) * ubRoute


        leftChild = self.addNode(
            parent, lbScenarioLeft, lbRouteLeft, ubRouteLeft, withDiscountIDLeft, noDiscountIDLeft,
            lbExpDiscountLeft, ubExpDiscountLeft, tspDictLeft, tspProbDictLeft, lbScenariosLeft, lbCoveredProbLeft, lbDensityCoveredLeft,
            exactValProbLeft, exactValueLeft, lastEnteranceDictionary)
        rightChild = self.addNode(
            parent, lbScenarioRight, lbRouteRight, ubRouteRight, withDiscountIDRight, noDiscountIDRight,
            lbExpDiscountRight, ubExpDiscountRight, tspDictRight, tspProbDictRight, lbScenariosRight, lbCoveredProbRight, lbDensityCoveredRight,
            exactValProbRight, exactValueRight, lastEnteranceDictionary)

        parent.children = [leftChild, rightChild]
        # TODO: this lbRecalculate can be moved in the line above
        for node in parent.children:
            lbCoveredProbNew, lbDensityCoveredNew = recalculateLbCovered(p_home, node, n)
            node.updateLbCovered(lbCoveredProbNew, lbDensityCoveredNew)

        if parent is self.bestNode:
            if leftChild.fathomedState:
                self.bestNode = rightChild
                self.bestUb = min(self.bestUb, self.bestNode.ubVal())
            else:
                self.bestNode = leftChild
                self.bestUb = min(self.bestUb, self.bestNode.ubVal())
                if not rightChild.fathomedState:
                    self.openNodes.push(rightChild, rightChild.priority())
        else:
            if not leftChild.fathomedState:
                self.openNodes.push(leftChild, leftChild.priority())

            if not rightChild.fathomedState:
                self.openNodes.push(rightChild, rightChild.priority())


    # Adds a new node to the BB tree
    # @param parent parent node of the node that is added to the tree
    # @param bound bound information (or any other information
    # @return the node newly added to the tree.

    # if new node coinside with rs_policy(fully coinsude) then set high priority coef
    def setPriorityCoef(self, withDiscountID, layer):
        coincide1 = True
        coincide2 = True

        for offset in range(layer):
            #compare with RS
            if ((1 << offset) & self.rs_policy) != ((1 << offset) & withDiscountID):
                coincide1 = False
            if not coincide1:
                break

        # compare with NODISC
        if withDiscountID > 0:
            coincide2 = False
        if coincide1 or coincide2:
            return 0.5
        else:
            return 1

    def addNode(self, parent, lbScenario, lbRoute, ubRoute, withDiscountID, noDiscountID, lbExpDiscount, ubExpDiscount,
                tspDict,tspProbDict, lbScenarios, lbCoveredProb, lbDensityCovered, exactValueProb, exactValue, lastEnteranceDictionary):
        # Create a new node
        layer = parent.layer + 1

        #priorityCoef is the coef defined by ring-star branching policy
        priorityCoef = self.setPriorityCoef(withDiscountID, layer)
        node = Node(parent, lbScenario, lbRoute, ubRoute, withDiscountID, noDiscountID, lbExpDiscount, ubExpDiscount,
                    tspDict,tspProbDict, lbScenarios, None,lbCoveredProb, lbDensityCovered,lbDensityCovered, exactValueProb, exactValue, layer,
                    priorityCoef, lastEnteranceDictionary)


        #assert lbRoute >= 0, 'loop4 new node next node lbRoute <0'
        #mostProbScenario = 2 ** self.instance.NR_CUST - 1 - noDiscountID
        #if mostProbScenario in self.instance.routeCost:
        #    node.updateLbScenario(self.instance.routeCost[mostProbScenario], self.instance.p_home, self.instance.NR_CUST )


        if node.lbRoute + node.lbExpDiscount > self.bestNode.ubVal():
            node.fathomedState = True
            #print(node.lbRoute ,  node.lbExpDiscount,  self.bestNode.ubVal())
            #print(self.instance.instance.p_pup_home, self.instance.shipping_fee)

        elif self.canFathomByTheoremCliques(node):
            node.fathomedState = True
        elif self.canFathomByTheoremUpperBound(node):
            node.fathomedState = True
        if layer in self.nodeLayers:
            self.nodeLayers[layer][1].nextNodeInLayer = node
            node.prevNodeInLayer = self.nodeLayers[layer][1]
        else:
            self.nodeLayers[layer] = [node, node]

        self.nodeLayers[layer][1] = node
        # nodeLayers.put(layer, node)
        self.nrNodes += 1

        # add the node to the open list of nodes
        return node

    # Check whether the node is a terminal node.If so, do something with it
    def isTerminalNode(self, node):
        if self.isLeaf(node) and node is self.bestNode:
            if  self.openNodes.empty():
                return True
        return False

    def canFathomByTheoremCliques(self, node):
        diff_customer = node.layer-1

        for other_cust in self.instance.neighbour[diff_customer+1]:
            cust = other_cust - 1
            if cust<diff_customer:
                if (node.withDiscountID & (1 << diff_customer) and not node.withDiscountID & (1 << cust)) or \
                        (not node.withDiscountID & (1 << diff_customer) and  node.withDiscountID & (1 << cust)):
                    return True
        return False

    def canFathomByTheoremCliques_Heuristic(self, node):
        diff_customer = node.layer-1

        for other_cust in self.instance.neighbourhood_enlarged[diff_customer+1]:
            cust = other_cust - 1
            if cust<diff_customer:
                if (node.withDiscountID & (1 << diff_customer) and not node.withDiscountID & (1 << cust)) or \
                        (not node.withDiscountID & (1 << diff_customer) and  node.withDiscountID & (1 << cust)):
                    node.lbRoute = node.ubVal()
                    node.ubRoute = node.lbRoute
                    node.lbExpDiscount = 0
                    node.ubExpDiscount = 0
                    node.fathomedState = True
                    return True
        return False


    def canFathomByTheoremUpperBound(self, node):
        if bitCount(node.withDiscountID) > self.upperBoundNumberDiscount:
            return True
        else:
            return False

    def printTree(self, node, file=None, _prefix="", _last=True):

        print(_prefix, "`- " if _last else "|- ", str(node.formPolicy(self.instance.NR_CUST)) + " " + str(round(node.lbVal(), 2)) + " " + \
              str(round(node.ubVal(), 2)) + "   " + str(node.fathomedState), sep="", file=file)
        _prefix += "   " if _last else "|  "
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            _last = i == (child_count - 1)
            self.printTree(child, file, _prefix, _last)

    def numLeaves(self, current_root):
        if current_root is None:
            return 0
        else:
            try:
                leftChild = current_root.children[0]
                try:
                    rightChild = current_root.children[1]
                except:
                    rightChild = None
            except:
                leftChild = None
                rightChild = None
            if self.isLeaf(current_root):
                return self.numLeaves(leftChild) + self.numLeaves(rightChild) + 1
            else:
                return self.numLeaves(leftChild) + self.numLeaves(rightChild)

    # Return the number of layers in this BB tree
    # @return number of layers in this BB tree

    def getNrLayers(self):
        return len(self.nodeLayers)

    def getNrNodes(self):
        return self.nrNodes

    # Returns an iterator over all nodes in a given layer. Nodes in a layer are stored in a * linked list to efficiently perform this operation.
    # @param layer layer
    # @return iterator over all nodes in a given layer

    def nodeIterator(self, layer):

        current = None
        if layer in self.nodeLayers:
            next = self.nodeLayers[layer][0]
        else:
            next = None
        while next is not None:
            current = next
            next = current.nextNodeInLayer
            yield current

            # @Override
            # public void remove():
            #    removeNode(current)


    '''
    def fathomByLayerSibling(self, node):
        p_dev = self.instance.p_dev
        ubScenario = self.instance.routeCost[0]
        # print("in fathom by layer", node.lbRoute)

        for sibling in self.nodeIterator(node.layer):
            if sibling is not node and sibling.fathomedState:
                # check if sibling is a "dynamic parent"
                mask = (node.withDiscountID & (2 ** node.layer - 1)) - sibling.withDiscountID
                if ((mask & (mask - 1)) == 0) and (node.withDiscountID & mask > 0):

                    # print("parent sibling", node, sibling, sibling.lbRoute)
                    # print(sibling, (sibling.lbRoute - sibling.lbScenario*(1-sibling.exactValueProb))*p_dev + (1-sibling.exactValueProb*p_dev)*node.lbScenario+node.lbExpDiscount)
                    if sibling.lbRoute * p_dev + (1 - p_dev) * node.lbScenario + node.lbExpDiscount > self.bestUb:
                        # node.fathomed()
                        return True
                    diff_customer = mask.bit_length()
                    fathomed = BoundsCalculation.compareDynamicParent(node, sibling, diff_customer, self)
                    # print("here2", node.lbRoute)
                    # TODO carefully get info from sibling with more info (15_C1_12_1.6_30)
                    # if sibling.exactValueProb * p_dev > max(node.exactValueProb,constants.EPS):
                    # TODO maybe the latter is valid not for fathomed
                    if False:
                        # print("HERE2", sibling)
                        # print("before", node.lbRoute, node.ubRoute)
                        node.exactValueProb = sibling.exactValueProb * p_dev
                        node.lbCalculated = sibling.lbCalculated.copy()
                        oldLbScenario = node.lbScenario
                        lbScenarioRight, lbCalculatedRight = recalculateLbCovered(lbCalculatedRight, lbScenarioRight,
                                                                                  p_dev,
                                                                                  withDiscountIDRight,
                                                                                  diff_customer + 1, n)
                        # print("diff_customer + 1", diff_customer + 1)
                        lbImprove = (1 - exactValProbRight) * (lbScenarioRight - oldLbScenario)
                        lbRouteRight += lbImprove

                        node.lbRoute = (sibling.lbRoute - sibling.lbScenario * (1 - sibling.exactValueProb)) * p_dev + (
                                1 - sibling.exactValueProb * p_dev) * node.lbScenario
                        node.ubRoute = (sibling.ubRoute - ubScenario * (1 - sibling.exactValueProb)) * p_dev + (
                                1 - sibling.exactValueProb * p_dev) * ubScenario
                        # print("after",node.lbRoute , node.ubRoute  )

                # check if sigbling is a "dynamic child"
                mask = sibling.withDiscountID - node.withDiscountID & (2 ** node.layer - 1)
                if (mask & (mask - 1)) == 0 and (sibling.withDiscountID & mask > 0):

                    # print("in dynamic child", sibling,sibling.exactValueProb,  node, node.exactValueProb )
                    # print(sibling.lbRoute, sibling.lbVal())
                    # Carefully get info
                    if False:
                        oldLbScenario = node.lbScenario
                        # print("")
                        # print("before", node.lbCalculated )
                        # TODO the length can be larger than 5
                        node.lbCalculated = sibling.lbCalculated.copy()
                        # print("after", node.lbCalculated )
                        node.lbScenario, node.lbCalculated = recalculateLbCovered(node.lbCalculated, node.lbScenario,
                                                                                  self.instance.p_dev,
                                                                                  node.withDiscountID,
                                                                                  node.layer,
                                                                                  self.instance.NR_CUST)

                        lbImprove = (1 - node.exactValueProb) * (node.lbScenario - oldLbScenario)
                        node.lbRoute += lbImprove

                    diff_customer = mask.bit_length()
                    fathomed = BoundsCalculation.compareDynamicChild(node, sibling, diff_customer, self)
                    if fathomed:
                        # print("sibling_comparison", sibling, node, diff_customer)
                        return True
            #
                    if node.exactValueProb < sibling.exactValueProb+10:
                        node.exactValueProb = sibling.exactValueProb
                        exactLbRoute = sibling.lbRoute - sibling.lbScenario*(1-sibling.exactValueProb)
                        exactUbRoute = sibling.ubRoute  - ubScenario*(1-sibling.exactValueProb)
                        for scenario in sibling.tspCalculated:
                            if (scenario & mask) > 0:
                                node.tspCalculated.remove(scenario)
                                scenarioProbLb = (1 - p_dev) ** bitCount(sibling.withDiscountID & scenario) * p_dev ** (
                                        bitCount(sibling.withDiscountID ^ scenario) + self.instance.NR_CUST - (
                                        sibling.withDiscountID + sibling.noDiscountID).bit_length())


                                node.exactValueProb -= scenarioProbLb
                                exactLbRoute -= scenarioProbLb*self.instance.routeCost[scenario]
                                exactUbRoute -= scenarioProbLb * self.instance.routeCost[scenario]
                        node.exactValueProb = node.exactValueProb/p_dev
                        node.lbRoute = exactLbRoute/p_dev + (1-node.exactValueProb)*node.lbScenario
                        node.ubRoute = exactUbRoute / p_dev + (1 - node.exactValueProb) * ubScenario
                    if node.lbVal() > self.bestUb:
                        node.fathomed()

                    #if (sibling.lbRoute - sibling.lbScenario*(1-sibling.exactValueProb))*p_dev + (1-sibling.exactValueProb*p_dev)*node.lbScenario+node.lbExpDiscount > self.bestUb:
                    #    node.fathomed()
                    #    break
                    '''


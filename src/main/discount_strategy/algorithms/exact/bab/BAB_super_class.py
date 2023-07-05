#tree of BAB - the structure is used in both Exact algorithm and in heuristics
import collections
import numpy as np
import heapq
import copy
from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import set_probability_covered
#from BoundsCalculation import updateBoundsFromDictionary
import itertools
from collections import OrderedDict
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
        if (len(self._queue) == 0):
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
        self.pruned_insertionCost_nonleaf = 0
        self.pruned_insertionCost_leaf = 0
        self.pruned_cliques_nonleaf = 0
        self.pruned_cliques_leaf = 0
        self.pruned_rs_nonleaf = 0
        self.pruned_rs_leaf = 0
        self.pruned_bounds_nonleaf = 0
        self.pruned_branching = 0

        # gurobi model for TSP is stored in TSP object and reused
        # thus to enable warm start
        self.TSPSolver = TSPSolver(instance=instance, solverType=solverType)

        # add two scnenarios which function as a LB and UB
        scenario_0 = 0
        self.instance.routeCost[scenario_0] = self.TSPSolver.tspCost(scenario_0)

        lbScenarios = {}
        self.instance.ubScenario, self.instance.lbScenario = self.TSPSolver.tspCostUbScenario()
        lbScenarios, lbScenario = {}, self.instance.lbScenario
        # COVERED_BOUND
        #DOMINANCE_CHECK_BOUNDS_REMOVED
        lbScenarios, self.instance.lbScenario  = self.set_lbScenarios()
        ubRoute = self.instance.ubScenario
        lbRoute = self.instance.lbScenario
        tspDict = {instance.NR_CUST:[scenario_0]}
        tspProbDict = {scenario_0:probability.scenarioProb_2segm(scenario_0, 0, 0, instance.NR_CUST, instance.p_pup_delta)}

        # Root node of the BB tree
        root = Node(parent=None,
                    lbRoute = lbRoute,
                    ubRoute = ubRoute,
                    withDiscountID=0,
                    noDiscountID=0,
                    lbExpDiscount=0,
                    ubExpDiscount=(instance.p_pup_delta.dot(instance.shipping_fee) ),#*constants.PROB_PLACE_ORDER,
                    tspDict=tspDict,
                    tspProbDict = tspProbDict,
                    lbScenarios=lbScenarios,
                    exactValueProb = tspProbDict[scenario_0],
                    exactValue = tspProbDict[scenario_0]*self.instance.routeCost[scenario_0],
                    layer=0,
                    priorityCoef = 1,
                    lastEnteranceDictionary = scenario_0
                    )
        lastEnteranceDictionary = None
        set_probability_covered(lbScenarios,0,  tspProbDict, self.instance)
        # we already added scenario 0 to the exact cost of the root - this is done in set_probability_covered function


        # DOMINANCE_CHECK_BOUNDS_REMOVED
        #root.lbRoute = sum(root.lbScenarios[id][1] * root.lbScenarios[id][0] for id in root.lbScenarios) + root.exactValue + (1-tspProbDict[scenario_0])*lbScenario
        root.lbScenarios[0][1] = 0
        root.lbRoute = sum(root.lbScenarios[id][1] * root.lbScenarios[id][0] for id in root.lbScenarios) + root.exactValue
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

        #self.upperBoundNumberDiscount = self.instance.NR_CUST
        self.upperBoundNumberDiscount = bitCount(rsPolicyID)

    def set_lbScenarios(self):
        lbScenarios = OrderedDict()            #self.instance.NR_PUP
        lbScenario = 10**5
        set_pups = list(range(self.instance.NR_PUP))

        for number_visited_pups in range(self.instance.NR_PUP + 1):
            for combination in itertools.combinations(set_pups, number_visited_pups):
                id = sum(1 << (offset) for offset in combination)
                scenario = 0
                for pup in self.instance.pups:
                    if (1 << pup.number) & id:
                        for cust_id in pup.closest_cust_id:
                            scenario = scenario +  (1 << (cust_id - 1))
                #First element is the cost of lbScenario, the second element is the probabbility of all scenarios that have
                # lower cost than the lbScenario. We initiate with the zero probability
                lbScenarios[id] = [self.TSPSolver.tspCost(scenario), 0 , scenario]
                lbScenario = min(lbScenarios[id][0], lbScenario)

        return lbScenarios, lbScenario

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
        n = self.instance.NR_CUST
        # decide how to branch
        # calculate the new policy information
        # currently branch in alphabetical order, left branch gives discount, right - nodiscout
        noDiscountIDLeft = parent.noDiscountID
        withDiscountIDLeft = parent.withDiscountID + 2 ** (diff_customer-1)
        withDiscountIDRight = parent.withDiscountID
        noDiscountIDRight = parent.noDiscountID + 2 ** (diff_customer-1)

        lastEnteranceDictionary = self.root.lastEnteranceDictionary

        # CHANGESPROBABILITY
        lbExpDiscountLeft = parent.lbExpDiscount + (1 - p_home[diff_customer]) * shipping_fee[diff_customer]
        ubExpDiscountLeft = parent.ubExpDiscount
        lbExpDiscountRight = parent.lbExpDiscount
        ubExpDiscountRight = parent.ubExpDiscount - (1 - p_home[diff_customer]) * shipping_fee[diff_customer]

        lbScenariosRight = copy.deepcopy(parent.lbScenarios)
        lbScenariosLeft = copy.deepcopy(parent.lbScenarios)

        tspDictLeft = deepcopy(parent.tspDict)
        tspDictRight = deepcopy(parent.tspDict)

        tspProbDictRight = {}
        tspProbDictLeft = deepcopy(parent.tspProbDict)

        exactValProbLeft = parent.exactValueProb
        exactValueLeft = parent.exactValue

        # thus to reduce the problem with multiplication of small numbers
        if parent.exactValueProb > 10*constants.EPS:
            lastEnteranceDictionary = self.root.lastEnteranceDictionary
            exactValProbRight = parent.exactValueProb / p_home[diff_customer]
            exactValueRight = parent.exactValue / p_home[diff_customer]
            for i in parent.tspProbDict:
                tspProbDictRight[i] = parent.tspProbDict[i] / p_home[diff_customer]
        else:
            exactValProbRight = 0
            exactValueRight = 0
            for gamma in tspDictRight:
                # all elements up to and including the diff_customer equal 1
                for scenario in tspDictRight[gamma]:
                    if probability.scenarioPossible_2segm(scenario, withDiscountIDRight,  parent.layer + 1, n):
                        scenarioProbRight = probability.scenarioProb_2segm(scenario, withDiscountIDRight, parent.layer + 1, n,
                                                                  self.instance.p_pup_delta )
                        exactValProbRight += scenarioProbRight
                        exactValueRight += self.instance.routeCost[scenario] * scenarioProbRight
                        tspProbDictRight[scenario] = scenarioProbRight

        lbScenariosRight = set_probability_covered(lbScenariosRight, noDiscountIDRight, tspProbDictRight, self.instance)

        #DOMINANCE_CHECK_BOUNDS_REMOVED
        #lbRouteRight = sum(
        #    lbScenariosRight[id][1] * lbScenariosRight[id][0] for id in lbScenariosRight) + exactValueRight + (1-exactValProbRight)*self.instance.lbScenario

        lbRouteRight = sum(
            lbScenariosRight[id][1] * lbScenariosRight[id][0] for id in lbScenariosRight) + exactValueRight

        ubRouteRight = exactValueRight + (1-exactValProbRight) * self.instance.ubScenario

        lbRouteLeft = parent.lbRoute
        ubRouteLeft = parent.ubRoute

        leftChild = self.addNode(
            parent,  lbRouteLeft, ubRouteLeft, withDiscountIDLeft, noDiscountIDLeft,
            lbExpDiscountLeft, ubExpDiscountLeft, tspDictLeft, tspProbDictLeft, lbScenariosLeft,
            exactValProbLeft, exactValueLeft, lastEnteranceDictionary)
        rightChild = self.addNode(
            parent,  lbRouteRight, ubRouteRight, withDiscountIDRight, noDiscountIDRight,
            lbExpDiscountRight, ubExpDiscountRight, tspDictRight, tspProbDictRight, lbScenariosRight,
            exactValProbRight, exactValueRight, lastEnteranceDictionary)

        parent.children = [leftChild, rightChild]

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
        coincide3 = True

        for offset in range(layer):
            #compare with RS
            if ((1 << offset) & self.rs_policy) != ((1 << offset) & withDiscountID):
                coincide1 = False
            if not coincide1:
                break
        for offset in range(layer):
            #compare with RS
            if ((1 << offset) & self.rs_policy) < ((1 << offset) & withDiscountID):
                coincide3 = False
            if not coincide3:
                break

        # compare with NODISC
        if withDiscountID > 0:
            coincide2 = False
        if coincide1 or coincide2:
            return 0.5
        elif coincide3:
            return 0.8
        else:
            return 1

    def addNode(self, parent, lbRoute, ubRoute, withDiscountID, noDiscountID, lbExpDiscount, ubExpDiscount,
                tspDict,tspProbDict, lbScenarios, exactValueProb, exactValue, lastEnteranceDictionary):

        # Create a new node
        layer = parent.layer + 1

        #priorityCoef is the coef defined by ring-star branching policy
        priorityCoef = self.setPriorityCoef(withDiscountID, layer)
        node = Node(parent, lbRoute, ubRoute, withDiscountID, noDiscountID, lbExpDiscount, ubExpDiscount,
                    tspDict,tspProbDict, lbScenarios,  exactValueProb, exactValue, layer,
                    priorityCoef, lastEnteranceDictionary)
        # if withDiscountID== 0:
        #     print("inBranch", layer, exactValueProb, exactValue, lbRoute)
        #     for scenario in lbScenarios:
        #         print(lbScenarios[scenario] )

        if node.lbRoute + node.lbExpDiscount > self.bestNode.ubVal():
            node.fathomedState = True

        #DOMINANCE_CHECK_TOREMOVED
        # elif self.canFathomByTheoremCliques(node):
        #   node.fathomedState = True
        #   if node.layer == self.instance.NR_CUST:
        #      self.pruned_cliques_leaf +=1
        #   else:
        #      self.pruned_cliques_nonleaf += 1
        # elif self.canFathomByTheoremUpperBound(node):
        #    node.fathomedState = True
        #    if node.layer == self.instance.NR_CUST:
        #       self.pruned_rs_leaf += 1
        #    else:
        #       self.pruned_rs_nonleaf += 1
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
        diff_customer = node.layer
        for other_cust in self.instance.neighbour[diff_customer]:
            # if we defined the policy for the other_cust
            if other_cust<diff_customer:
                if (node.withDiscountID & (1 << diff_customer-1) and not node.withDiscountID & (1 << other_cust - 1)) or \
                        (not node.withDiscountID & (1 << diff_customer-1) and  node.withDiscountID & (1 << other_cust - 1)):
                    return True
        return False

    def canFathomByTheoremCliques_Heuristic(self, node):
        diff_customer = node.layer - 1

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

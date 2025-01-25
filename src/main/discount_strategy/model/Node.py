# Intrusive implementation of a BB node
# @author Joris Kinable
class Node:
    #Reference to the next node in the same layer * /
    #Node < S > nextNodeInLayer=null;
    #Reference to the previous node in the same layer * /
    #Node < S > prevNodeInLayer=null;

    #BNode constructor
    #@ param layer layer
    #@ param state state
    __slots__ = ['noDiscountID', 'withDiscountID', 'parent', 'layer','nextNodeInLayer', 'prevNodeInLayer', 'lbRoute',
                 'ubRoute', 'lbScenarios', 'lbExpDiscount', 'ubExpDiscount', 'children','fathomedState',
                 'tspDict','tspProbDict',  'exactValue','exactValueProb', 'setGivenDiscount',
                 'setNotGivenDiscount',  'tspAverageCost','priorityCoef','lastEnteranceDictionary','lastNumDeviatedNewTSP','number_disc']


    def __init__(self, parent,  lbRoute, ubRoute, withDiscountID, noDiscountID,
                 lbExpDiscount, ubExpDiscount, tspDict,tspProbDict, lbScenarios,  exactValueProb, exactValue, layer, priorityCoef,lastEnteranceDictionary):

        self.noDiscountID = noDiscountID
        self.withDiscountID = withDiscountID
        self.parent = parent
        # Layer containing this node
        self.layer = layer
        # Reference to the next node in the same layer * /
        self.nextNodeInLayer = None
        # Reference to the previous node in the same layer * /
        self.prevNodeInLayer = None
        self.lbRoute = lbRoute
        self.ubRoute = ubRoute
        self.lbExpDiscount = lbExpDiscount
        self.ubExpDiscount = ubExpDiscount
        self.priorityCoef = priorityCoef
        self.lastEnteranceDictionary = lastEnteranceDictionary
        self.lastNumDeviatedNewTSP = 0

        # Instance Variables
        # Children
        self.children = []
        self.fathomedState = False

        self.tspDict = tspDict
        self.tspProbDict = tspProbDict
        self.tspAverageCost = {}

        # represents the line of the probability table
        # has a key: "scenario" - scenario tuple, which has dictionaries as values

        # lbScenarios = scenarios that provide part of the true Lb_Route of the node.
        # It is possible that some of the routes will not be possible for some of children
        # (but still will be the TRUE LB_route for them)
        self.lbScenarios = lbScenarios

        self.exactValueProb = exactValueProb
        self.exactValue = exactValue

        givenDiscount = []
        notGivenDiscount = []


        for offset in range((noDiscountID + withDiscountID).bit_length()):
            mask = 1<<offset
            if not withDiscountID&mask:
                notGivenDiscount.append(offset+1)
            else:
                givenDiscount.append(offset+1)

        self.number_disc = len(givenDiscount)
        self.setGivenDiscount = givenDiscount
        self.setNotGivenDiscount = notGivenDiscount

    def lbVal(self):
        return self.lbRoute + self.lbExpDiscount
    def ubVal(self):
        return self.ubRoute + self.ubExpDiscount
    def priority(self):
        #return (self.ubRoute+self.ubExpDiscount+self.lbRoute+self.lbExpDiscount)*self.priorityCoef/(1+self.layer)/(1+self.layer)
        #return (self.ubRoute + self.ubExpDiscount)*self.priorityCoef
        return (self.ubRoute+self.ubExpDiscount + self.lbRoute + self.lbExpDiscount)*self.priorityCoef


    def __str__(self):
        return str(self.withDiscountID)+str(self.noDiscountID)

    def formPolicy( self, NR_CUST):
        lenDefined = (self.withDiscountID + self.noDiscountID).bit_length()
        if self.withDiscountID + self.noDiscountID == 0:
            policy = ['x'] * NR_CUST
        else:
            policy = ['x']*(NR_CUST -lenDefined)
            for i in bin(self.withDiscountID)[2:].zfill(lenDefined):
                policy.append(i)
        return policy

    def isRightChild(self):
        if self.noDiscountID >self.withDiscountID:
            return True
        else:
            return False

    # keeps only 1 fathomed node per layer
    def fathomed(self):
        self.fathomedState = True
        new_parent = self.parent
        while new_parent is not None and (new_parent.children[0].fathomedState) and (new_parent.children[1].fathomedState):

            new_parent.lbRoute = min(new_parent.children[0].lbVal(), new_parent.children[1].lbVal())
            #delete one children (not self)
            new_parent.lbExpDiscount = 0
            new_parent.fathomedState = True
            new_parent.children[1].prevNodeInLayer.nextNodeInLayer = new_parent.children[1].nextNodeInLayer
            del new_parent.children[1]
            new_child = new_parent

            while new_child.children:
                new_child = new_child.children[0]
                new_child.lbExpDiscount = 0
                new_child.lbRoute = new_parent.lbRoute

            new_parent = new_parent.parent

    def __eq__(self, other):
        if not isinstance(other, Node):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return  self.noDiscountID == other.noDiscountID and self.withDiscountID == other.withDiscountID
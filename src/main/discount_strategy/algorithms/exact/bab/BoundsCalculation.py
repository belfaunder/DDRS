import itertools
from scipy.special import comb
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm

def set_probability_covered(lbScenarios,noDiscountID, tspProbDict, instance):
    for id in lbScenarios:
        lbScenarios[id][1] = 1
        for pup in instance.pups:
            # for all those customers who should be visited at home in the scenario id
            if not (1 << pup.number) & id:
                for cust_id in pup.closest_cust_id:
                    # if the discount is either given to the customer or is not defined(becasue is the discount
                    # is not given to the customer, then the probability of selecting home is 1)
                    #if cust_id not in setNotGivenDiscount:
                    if not noDiscountID & (1 << cust_id - 1):
                        lbScenarios[id][1] *= instance.p_home[cust_id]

    for id in lbScenarios:
        visited_pups = [i for i in range(instance.NR_PUP) if id & (1 << i)]
        for number_visited_pups in range(bin(id).count("1")):
            for combination in itertools.combinations(visited_pups,  number_visited_pups):
                id_to_reduce = sum((1<<offset) for offset in combination)
                lbScenarios[id][1] -= lbScenarios[id_to_reduce][1]
    for id in reversed(lbScenarios):
        for pup in instance.pups:
            if (1 << pup.number) & id:
                id_less_pups_visited = id & ~(1 << pup.number)
                lbScenarios[id][0] = min(lbScenarios[id][0], lbScenarios[id_less_pups_visited][0])
    # remove any scenario from lbCoveredProb if this scenario was included in calculation of exactValue(prob)
    for scenario in tspProbDict:
        for id in lbScenarios:
            if not ~lbScenarios[id][2] & (2**(instance.NR_CUST) - 1) & scenario:
                lbScenarios[id][1] -=  tspProbDict[scenario]
                break
    return lbScenarios

def ub_insertion_cost(instance, setNotGivenDiscount,setGivenDiscount, diff_customer):
    p_home = instance.p_home
    ub_insertion = instance.ubInsertion[diff_customer]

    insertion_cost = 0
    p_left = 1
    for i in ub_insertion:
        if i in [0]+setNotGivenDiscount:
            insertion_cost += ub_insertion[i] *  p_left#*constants.PROB_PLACE_ORDER
            p_left = 0
        elif i>instance.NR_CUST:
            probability_temp = 1
            for pup in instance.pups:
                if pup.id ==i:
                    for cust in pup.closest_cust_id:
                        if cust in setGivenDiscount:
                            probability_temp *= instance.p_home[cust]
            insertion_cost += ub_insertion[i] * (1-probability_temp)* p_left
            p_left *=  probability_temp
        else:
            insertion_cost += ub_insertion[i] * p_home[i] * p_left#*constants.PROB_PLACE_ORDER
            p_left *= (1 - p_home[i])

    ub = (insertion_cost  - instance.shipping_fee[diff_customer])* instance.p_pup_delta[diff_customer]
    return ub

def lb_insertion_cost(instance, setGivenDiscount,setNotGivenDiscount, diff_customer):

    lb_insertion = instance.lbInsertion[diff_customer]
    p_home = instance.p_home
    insertion_cost = 0
    previous_insertion = []

    #prob_temp is the probability that non of the closest customers select home
    prob_temp = 1
    for pup in instance.pups:
        if diff_customer in pup.closest_cust_id:
            for cust in pup.closest_cust_id:
                if cust in setGivenDiscount:
                    prob_temp *= instance.p_home[cust]
            min_insertion_pup = 2 * instance.distanceMatrix[pup.id,0]
            for cust in setNotGivenDiscount:
                if cust is not diff_customer:
                    min_insertion_pup= min(min_insertion_pup,2 * instance.distanceMatrix[pup.id,cust])
    probability_closest_pup_visited = 1 - prob_temp

    #minimum insertion cost for the pickup point of the diff_customer:
    p_left = 1
    for (i, j) in lb_insertion:
        if p_left > 0:
            if [i, j] != [0, 0]:
                prob_current = 1
                if i == j:
                    if i in setGivenDiscount:
                        prob_current = p_home[i]
                    #if i is pup:
                    elif i > instance.NR_CUST:
                        probability_temp = 1
                        for pup in instance.pups:
                            if pup.id == i:
                                for cust in pup.closest_cust_id:
                                    if cust in setGivenDiscount:
                                        probability_temp *= instance.p_home[cust]
                        prob_current = 1 - probability_temp
                else:
                    if i > instance.NR_CUST:
                        probability_temp = 1
                        for pup in instance.pups:
                            if pup.id == i:
                                for cust in pup.closest_cust_id:
                                    if cust in setGivenDiscount:
                                        probability_temp *= instance.p_home[cust]
                        prob_current *= (1 - probability_temp)
                    elif (i in setGivenDiscount):
                        prob_current *= p_home[i]

                    if j > instance.NR_CUST:
                        probability_temp = 1
                        for pup in instance.pups:
                            if pup.id == j:
                                for cust in pup.closest_cust_id:
                                    if cust in setGivenDiscount:
                                        probability_temp *= instance.p_home[cust]
                        prob_current *= (1 - probability_temp)
                    elif (j in setGivenDiscount):
                        prob_current *= p_home[j]

                considered_pair = []
                for (m, k) in previous_insertion:
                    if (m == i or m == j) and k not in considered_pair:
                        considered_pair.append(k)
                        if k ==0 :
                            prob_current = 0
                        elif k > instance.NR_CUST:
                            for pup in instance.pups:
                                if pup.id == k:
                                    for cust in pup.closest_cust_id:
                                        if cust in setGivenDiscount:
                                            prob_current *=  instance.p_home[cust]
                        else:
                            if k in setGivenDiscount:
                                prob_current *= (1 - p_home[k])
                            else:
                                prob_current = 0

                    if k == i or k == j and m not in considered_pair:
                        considered_pair.append(m)
                        if m == 0:
                            prob_current = 0
                        elif m > instance.NR_CUST:
                            for pup in instance.pups:
                                if pup.id == m:
                                    for cust in pup.closest_cust_id:
                                        if cust in setGivenDiscount:
                                            prob_current *= instance.p_home[cust]
                        else:
                            if m in setGivenDiscount:
                                prob_current *= (1 - p_home[m])
                            else:
                                prob_current = 0
                insertion_cost += min(prob_current, p_left) * lb_insertion[i, j]
                if min(prob_current, p_left) > 0:
                    previous_insertion.append([i, j])

                p_left -= prob_current
            else:
                insertion_cost += lb_insertion[i, j] * p_left
        else:
            break


    insertion_cost = probability_closest_pup_visited*insertion_cost + (1-probability_closest_pup_visited)* (insertion_cost - min_insertion_pup)
    lb =  (insertion_cost-instance.shipping_fee[diff_customer]) * instance.p_pup_delta[diff_customer]
    return lb

def updateByInsertionCost(node, Bab):
    status_not_changed = True
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
            lbInsertionCost = lb_insertion_cost(Bab.instance, node.setGivenDiscount,node.setNotGivenDiscount, diff_customer)
            if lbInsertionCost + additionSibling > 0:
                node.lbRoute = node.parent.children[0].lbVal() + lbInsertionCost + additionSibling
                node.ubRoute = node.lbRoute
                node.lbExpDiscount = 0
                node.ubExpDiscount = 0
                node.fathomedState = True
                status_not_changed = False
        else:
            if node.parent.children[1].fathomedState:
                additionSibling = node.parent.children[1].lbVal() - Bab.bestNode.ubVal()

            else:
                additionSibling = 0
            ubInsertionCost = ub_insertion_cost(Bab.instance, node.setNotGivenDiscount, node.setGivenDiscount, diff_customer)

            if -ubInsertionCost + additionSibling > 0:
                node.lbRoute = node.parent.children[1].lbVal() - ubInsertionCost + constants.EPS + additionSibling
                node.ubRoute = node.lbRoute
                node.lbExpDiscount = 0
                node.ubExpDiscount = 0
                node.fathomedState = True
                status_not_changed = False

    return status_not_changed


def updateBoundsFromDictionary(Bab, node):
    n = Bab.instance.NR_CUST
    # calculate the worst UB on the insetion cost given the information about customers with discount
    if node.parent is not None:

        # First try to prune node by insertion criteria (if suceeded - status_not_changed=True), next by bounds
        if node is not Bab.bestNode:
           status_not_changed = updateByInsertionCost(node, Bab)
        else:
           status_not_changed = True

        if status_not_changed:
            for scenario, scenarioCost in itertools .dropwhile(lambda x: x[0] != node.lastEnteranceDictionary,  Bab.instance.routeCost.items()):
                if not ~node.withDiscountID & scenario:
                    #for scenario in routeCost:
                    gamma = n - bitCount(scenario)
                    if not node.tspDict.get(gamma):
                        node.tspDict[gamma] = []
                    if scenario not in node.tspDict[gamma]:
                        node.tspDict[gamma].append(scenario)
                        scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, node.layer,
                                                                n, Bab.instance.p_pup_delta)
                        node.tspProbDict[scenario] = scenarioProb
                        node.exactValueProb += scenarioProb
                        node.exactValue += scenarioCost * scenarioProb
                        for id in node.lbScenarios:
                            if not ~node.lbScenarios[id][2] & (2 ** (Bab.instance.NR_CUST) - 1) & scenario:
                                node.lbScenarios[id][1] -= node.tspProbDict[scenario]
                                node.lbRoute += scenarioProb * (scenarioCost - node.lbScenarios[id][0])
                                break
                        node.ubRoute -= (Bab.instance.ubScenario - scenarioCost) * scenarioProb

            #lastEnteranceDictionary is the last element of Route Cost that was checked for this node
            node.lastEnteranceDictionary = next(reversed(Bab.instance.routeCost))

        else:
            if node.layer == Bab.instance.NR_CUST:
               Bab.pruned_insertionCost_leaf += 1
            else:
               Bab.pruned_insertionCost_nonleaf += 1

def updateBoundsWithNewTSPs(Bab, node):

    n = Bab.instance.NR_CUST
    setMayVary = node.setGivenDiscount
    continue_flag = True
    added = 0
    gap_old = (node.ubRoute - node.lbRoute) / node.ubRoute
    # limit the number of deviated since function may be too slow for small probabilities

    if node.exactValueProb < 1:
        for gamma in range(n - len(setMayVary), n + 1):
            if gamma not in node.tspDict:
                node.tspDict[gamma] = []
            if len(node.tspDict[gamma]) == comb(len(node.setGivenDiscount), gamma - len(node.setNotGivenDiscount)):
                continue
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
                    node.tspDict[gamma].append(scenario)
                    added += 1
                    scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, n, n,
                                                                        Bab.instance.p_pup_delta )

                    node.tspProbDict[scenario] = scenarioProb
                    node.exactValueProb += scenarioProb
                    node.exactValue += scenarioCost * scenarioProb
                    for id in node.lbScenarios:
                        if not ~node.lbScenarios[id][2]&(2**(Bab.instance.NR_CUST) - 1) & scenario:
                            node.lbScenarios[id][1] -= node.tspProbDict[scenario]
                            node.lbRoute += scenarioProb * (scenarioCost - node.lbScenarios[id][0])
                            break
                    node.ubRoute -= scenarioProb * (Bab.instance.ubScenario - scenarioCost)

                if (node.ubRoute - node.lbRoute) / node.ubRoute <= 0.9 * gap_old or \
                        (node.lbRoute + node.lbExpDiscount >= Bab.bestUb) or (
                        node.ubRoute + node.ubExpDiscount < Bab.bestNode.lbVal()) or (added > 100):
                    continue_flag = False
                    break
            if not continue_flag:
                break


def updateBoundsWithNewTSPsHeuristic(Bab, node):
    sample_size = constants.HEURISTIC_SAMPLE_SIZE
    estimation_cost_by_sampling, Bab.instance.routeCost  = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance = Bab.instance, setMayVary = node.setGivenDiscount,
                     policy = node.withDiscountID,  solverType =Bab.TSPSolver.solverType, sample_size =  sample_size, routeCosts=Bab.instance.routeCost, solver = Bab.TSPSolver)
    node.exactValueProb = 1
    node.lbRoute = estimation_cost_by_sampling[0]-node.lbExpDiscount
    node.ubRoute = node.lbRoute
    node.exactValue = node.lbRoute

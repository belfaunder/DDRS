import itertools
from scipy.special import comb
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.util.bit_operations import bitCount

from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample
from src.main.discount_strategy.algorithms.heuristic.sample_average import importanceSampling



def set_probability_covered(lbScenarios,noDiscountID, tspProbDict, instance):
    # DOMINANCE_CHECK_REMOVED
    #return {}

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

    #TODO: carefully remove the probability from the lbScenarios wherever you add a new TSP
    # remove any scenario from lbCoveredProb if this scenario was included in calculation of exactValue(prob)
    for scenario in tspProbDict:
        for id in lbScenarios:
            if not ~lbScenarios[id][2]&(2**(instance.NR_CUST) - 1) & scenario:
                lbScenarios[id][1] -=  tspProbDict[scenario]
                break

    return lbScenarios

#CHANGESPROBABILITY
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

#CHANGESPROBABILITY
#TODO: finish the lb for the case with multiple pups

def lb_insertion_cost(instance, setGivenDiscount,setNotGivenDiscount, diff_customer):

    lb_insertion = instance.lbInsertion[diff_customer]
    p_home = instance.p_home
    insertion_cost = 0
    previous_insertion = []

    prob_temp = 1
    for pup in instance.pups:
        if diff_customer in pup.closest_cust_id:
            for cust in pup.closest_cust_id:
                if cust in setGivenDiscount:
                    prob_temp *= instance.p_home[cust]
            min_insertion_pup = 2 * instance.distanceMatrix[pup.id,0]
            for cust in setNotGivenDiscount:
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
            additionSibling = 0
            if node.parent.children[0].fathomedState:
                additionSibling = node.parent.children[0].lbVal() - Bab.bestNode.ubVal()
            else:
                additionSibling = 0
            lbInsertionCost = lb_insertion_cost(Bab.instance, node.setGivenDiscount,node.setNotGivenDiscount, diff_customer)
            if lbInsertionCost + additionSibling > -constants.EPS*Bab.bestNode.lbVal():
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

            if -ubInsertionCost + additionSibling > -constants.EPS*Bab.bestNode.lbVal():
                node.lbRoute = node.parent.children[1].lbVal() - ubInsertionCost + constants.EPS + additionSibling
                node.ubRoute = node.lbRoute
                node.lbExpDiscount = 0
                node.ubExpDiscount = 0
                node.fathomedState = True
                status_not_changed = False

    return status_not_changed


def updateBoundsFromDictionary(Bab, node):
    n = Bab.instance.NR_CUST
    #print(len(Bab.instance.routeCost))
    # calculate the worst UB on the insetion cost given the information about customers with discount
    if node.parent is not None:
        #DOMINANCE_CHECK_REMOVED
        #status_not_changed = True
        if node is not Bab.bestNode:
            status_not_changed = updateByInsertionCost(node, Bab)
        else:
            status_not_changed = True

        if status_not_changed:
            setGivenDiscount = node.setGivenDiscount
            number_offered_discounts = len(setGivenDiscount)
            # limit the number of deviated since function may be too slow for small probabilities
            #num_may_deviate = Bab.instance.maxNumDeviated[len(setGivenDiscount)]
            # if (2 ** n - 1 - node.noDiscountID) in Bab.instance.routeCost:
            #     node.updateLbScenario(Bab.instance.routeCost[2 ** n - 1 - node.noDiscountID], Bab.instance.p_home, n)
            # limit the number of deviated since function may be too slow for small probabilities

            num_may_deviate = Bab.instance.maxNumDeviated[len(setGivenDiscount)]
            setMayVary = node.setGivenDiscount

            initial_probability = 1

            for i in node.setGivenDiscount:
                initial_probability *= Bab.instance.p_pup_delta[i]
            for i in range(node.layer + 1, n + 1):
                initial_probability *= Bab.instance.p_home[i]

            #gamma is the number of visited customers
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
                    scenarioProb = initial_probability
                    for offset in combination:
                        mask = ~(1 << (offset - 1))
                        scenario = scenario & mask
                        scenarioProb *= Bab.instance.p_home[i]/Bab.instance.p_pup_delta[i]
                    # check if scenario is already in node
                    if scenario not in node.tspDict[gamma]:
                        scenarioCost = Bab.instance.routeCost.get(scenario)
                        if scenarioCost:
                            node.tspDict[gamma].append(scenario)
                            # scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, node.layer, n,
                            #                                               Bab.instance.p_pup_delta )
                            # if (scenarioProb-scenarioProb1)>constants.EPS or (scenarioProb1-scenarioProb)>constants.EPS:
                            #     print(scenarioProb, scenarioProb1)
                            node.tspProbDict[scenario] = scenarioProb
                            node.exactValueProb += scenarioProb
                            node.exactValue += scenarioCost * scenarioProb

                            #DOMINANCE_CHECK_REMOVED
                            #node.lbRoute += scenarioProb * (scenarioCost - Bab.instance.lbScenario)
                            for id in node.lbScenarios:
                                if not ~node.lbScenarios[id][2] & (2 ** (Bab.instance.NR_CUST) - 1) & scenario:
                                    node.lbScenarios[id][1] -= node.tspProbDict[scenario]
                                    node.lbRoute += scenarioProb * (scenarioCost - node.lbScenarios[id][0])
                                    break
                            node.ubRoute -= (Bab.instance.ubScenario- scenarioCost) * scenarioProb

            # for scenario, scenarioCost in dropwhile(lambda x: x[0] != node.lastEnteranceDictionary,  Bab.instance.routeCost.items()):
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
            #             node.exactValueProb += scenarioProb
            #             node.exactValue += scenarioCost * scenarioProb
            #             for id in node.lbScenarios:
            #                 if not ~node.lbScenarios[id][2] & (2 ** (Bab.instance.NR_CUST) - 1) & scenario:
            #                     node.lbScenarios[id][1] -= node.tspProbDict[scenario]
            #                     node.lbRoute += scenarioProb * (scenarioCost - node.lbScenarios[id][0])
            #                     break
            #             node.ubRoute -= (Bab.ubScenario - scenarioCost) * scenarioProb
            #lastEnteranceDictionary is the last element of Route Cost that was checked for this node
            node.lastEnteranceDictionary = next(reversed(Bab.instance.routeCost))
        else:
            if node.layer == Bab.instance.NR_CUST:
                Bab.pruned_insertionCost_leaf += 1
            else:
                Bab.pruned_insertionCost_nonleaf += 1

def updateBoundsWithNewTSPs(Bab, node):

    n = Bab.instance.NR_CUST
    setGivenDiscount = node.setGivenDiscount
    setMayVary = node.setGivenDiscount
    continue_flag = True
    added = 0
    gap_old = (node.ubRoute - node.lbRoute) / node.ubRoute
    # limit the number of deviated since function may be too slow for small probabilities
    #num_may_deviate = Bab.instance.maxNumDeviated[len(setGivenDiscount)]
    # if (2 ** n - 1 - node.noDiscountID) in Bab.instance.routeCost:
    #     node.updateLbScenario(Bab.instance.routeCost[2 ** n - 1 - node.noDiscountID], Bab.instance.p_home, n)

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
                    # if gamma == n - len(setMayVary):
                    #     node.updateLbScenario(scenarioCost, Bab.instance.p_home, n)
                    node.tspDict[gamma].append(scenario)
                    added += 1
                    scenarioProb = probability.scenarioProb_2segm(scenario, node.withDiscountID, n, n,
                                                                        Bab.instance.p_pup_delta )

                    node.tspProbDict[scenario] = scenarioProb
                    node.exactValueProb += scenarioProb
                    node.exactValue += scenarioCost * scenarioProb
                    #DOMINANCE_CHECK_REMOVED
                    #node.lbRoute += scenarioProb * (scenarioCost - Bab.instance.lbScenario)
                    for id in node.lbScenarios:
                        if not ~node.lbScenarios[id][2]&(2**(Bab.instance.NR_CUST) - 1) & scenario:
                            node.lbScenarios[id][1] -= node.tspProbDict[scenario]
                            node.lbRoute += scenarioProb * (scenarioCost - node.lbScenarios[id][0])
                            break
                    node.ubRoute -= scenarioProb * (Bab.instance.ubScenario - scenarioCost)

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
    Bab.lbScenario  = node.lbRoute

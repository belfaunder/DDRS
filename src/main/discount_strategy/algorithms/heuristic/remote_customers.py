import numpy as np
from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW
prefix=constants.PREFIX

def policy_remote_customers(instance):

    farness = {}
    list_cust_ids = [cust.id for cust in instance.customers]
    # 1 Dominance rule 2
    dict_same_decision = {}
    for cust in instance.customers:
        if cust.id in list_cust_ids:
            for cust2 in instance.customers:
                if cust.id != cust2.id:
                    if instance.distanceMatrix[cust.id, cust2.id] <= (cust.shipping_fee + cust2.shipping_fee) / 2:
                        if cust2.id in list_cust_ids:
                            list_cust_ids.remove(cust2.id)
                        if cust.id in dict_same_decision:
                            dict_same_decision[cust.id].append(cust2.id)
                        elif cust.id in list_cust_ids:
                            dict_same_decision[cust.id] = [cust2.id]
                        else:
                            for key in dict_same_decision:
                                if cust.id in dict_same_decision[key] and cust.id != key:
                                    dict_same_decision[key].append(cust2.id)

    size_neighbours = round(len(instance.customers) / 8) + 1

    list_farness = []
    list_discounts = []
    list_farness_all = []
    list_temp = []
    for cust in instance.customers:
        list_farness.append(instance.distanceMatrix[cust.id, cust.closest_pup_id])
        list_discounts.append(cust.shipping_fee)
        distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                           j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [instance.distanceMatrix[cust.id, instance.depot.id]]
        farness[cust.id] = sum(sorted(distancesf)[:size_neighbours])/size_neighbours
        list_farness_all.append(round(sum(sorted(distancesf)[:size_neighbours])/size_neighbours))
        list_temp.append(round((sum(sorted(distancesf)[:size_neighbours]) / size_neighbours )*(1-cust.prob_home)))
    list_temp.reverse()
    list_farness.reverse()
    list_discounts.reverse()
    list_farness_all.reverse()

    #start by removing discounts offered to customers, who are further than dist_1, then remove cusotmer further than dist_2 in the increasing order of cusotmers' distance
    # next, remove customers located closer then dist_1 in the decreasing order
    dist_1 = min(list_farness) + (max(list_farness) - min(list_farness))/4
    dist_1 = np.percentile(list_farness, 33)
    dist_2 = min(list_farness) + (max(list_farness) - min(list_farness))*2/3
    policy = 0
    policy_rs, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)
    for cust in instance.customers:
        if cust.id in list_cust_ids:
            if policy_rs & (1 << int(cust.id-1)):
                if instance.distanceMatrix[cust.id, cust.closest_pup_id] < dist_1:
                    if cust.shipping_fee < farness[cust.id]:
                        policy += (1 << int(cust.id - 1))
                        if cust.id in dict_same_decision:
                            for cust2 in dict_same_decision[cust.id]:
                                policy += (1 << int(cust2 - 1))
                else:
                    if cust.shipping_fee <(1-cust.prob_home)*farness[cust.id]:
                        policy += (1 << int(cust.id - 1))
                        if cust.id in dict_same_decision:
                            for cust2 in dict_same_decision[cust.id]:
                                policy += (1 << int(cust2 - 1))

    return policy

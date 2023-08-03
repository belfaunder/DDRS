from pathlib import Path
import sys
from sys import argv
import math
import os
from time import process_time
import cProfile
import pstats
from src.main.discount_strategy.io.print_functions import Painter
import numpy as np
from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
from src.main.discount_strategy.util.bit_operations import bitCount
from src.main.discount_strategy.algorithms.exact.bab.BAB_exact import BABExact
from src.main.discount_strategy.algorithms.heuristic.sample_average import sampleAverageApproximation_PoissonBinomial_1sample_2segm
from src.main.discount_strategy.algorithms.heuristic.sample_average import one_policy_cost_estimation
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.io import print_functions
from src.main.discount_strategy.algorithms.exact.ring_star_without_TW import ring_star_deterministic_no_TW
#prefix="tag: "
prefix=constants.PREFIX


def policy_insights_Nevin(instance):
    def proportion_closest(delta):
        #print((-0.6*delta + 0.9))
        return -0.6*delta + 0.9
    policy_rs, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)
    deltas = [1-cust.prob_home for cust in instance.customers]
    delta = sum(deltas) / len(deltas)
    number_incentives = math.ceil(bitCount(policy_rs) * (delta*delta*0.8+0.2))
    print(bitCount(policy_rs), number_incentives)

    painter = Painter()
    painter.printVertex(instance)
    solverType = 'Gurobi'
    #farness = {}
    list_farness = []
    list_dist_closest_customer = []
    for cust in instance.customers:
        list_farness.append(instance.distanceMatrix[cust.id, cust.closest_pup_id])
        distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                      j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [
                         instance.distanceMatrix[cust.id, instance.depot.id]]
        list_dist_closest_customer.append(min(distancesf))
    #dist_1 = np.percentile(list_farness, 33)
    #dist_2 = np.percentile(list_farness, 66)
    dist_1 = min(list_farness) + (max(list_farness) - min(list_farness)) / 3
    dist_2 = min(list_farness) + (max(list_farness) - min(list_farness)) * 2 / 3
    list_farness.reverse()
    list_dist_closest_customer.reverse()
    #print("list_farness", list_farness)
    print("list_dist_closest_customer", list_dist_closest_customer)
    dict_customers_ranges = {'closest':[], 'middle':[], 'furthest':[]}

    for cust in instance.customers:
        if instance.distanceMatrix[cust.id, cust.closest_pup_id] <= dist_1:
            dict_customers_ranges['closest'].append(cust.id)
        elif instance.distanceMatrix[cust.id, cust.closest_pup_id] <= dist_2:
            dict_customers_ranges['middle'].append(cust.id)
        else:
            dict_customers_ranges['furthest'].append(cust.id)
    #print("dict_customers_ranges", dict_customers_ranges)
    #print("proportion_closest(delta)", proportion_closest(delta))
    number_incentives_closest = round(proportion_closest(delta) * number_incentives)
    number_incentives_fatherst= round((1-proportion_closest(delta)) * number_incentives/2)
    number_incentives_middle = round((1-proportion_closest(delta)) * number_incentives/2)

    number_incentives = {'closest':number_incentives_closest, 'middle':number_incentives_middle, 'furthest':number_incentives_fatherst}
    print(number_incentives)


    policy = 0
    for key in dict_customers_ranges.keys():
        distances_closest_range = []
        for cust in instance.customers:
            if cust.id in dict_customers_ranges[key]:
                # distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                #               j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [
                #                  instance.distanceMatrix[cust.id, instance.depot.id]]
                distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                              j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [
                                 instance.distanceMatrix[cust.id, instance.depot.id]]
                distances_closest_range.append(sum(sorted(distancesf)[:4]) / 4)
                #distances_closest_range.append(min(distancesf))
        # we will offer discounts to "number_incentives" customers of this range if the distance to closest is among the "number_incentives" largest

        distances_closest_range = sorted(distances_closest_range, reverse= True)
        #print(key, "distances_closest_range", distances_closest_range, distances_closest_range[:min(number_incentives[key], len(dict_customers_ranges['closest']))])

        num_incentives = min(number_incentives[key], len(dict_customers_ranges['closest']))
        if num_incentives>0 and len(distances_closest_range) > 0:
            min_allowed_distance_closest = min(distances_closest_range[:num_incentives])
        else:
            min_allowed_distance_closest = 100000000
        num_given = 0

        for cust in instance.customers:
            if num_given < number_incentives[key] and cust.id in dict_customers_ranges[key]:
                distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                              j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [
                                 instance.distanceMatrix[cust.id, instance.depot.id]]

                if sum(sorted(distancesf)[:4]) / 4 >= min_allowed_distance_closest:
                    policy += (1 << int(cust.id - 1))
                    num_given+=1
    #print(bin(policy))
    return policy

def policy_remote_customers(instance):
    policy_rs, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)
    painter = Painter()
    painter.printVertex(instance)
    solverType = 'Gurobi'
    farness = {}

    list_farness = []
    list_discounts = []
    list_farness_all = []
    for cust in instance.customers:
        list_farness.append(instance.distanceMatrix[cust.id, cust.closest_pup_id])
        list_discounts.append(cust.shipping_fee)
        distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                           j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [instance.distanceMatrix[cust.id, instance.depot.id]]
        #print(cust.id, sorted(distancesf), instance.distanceMatrix[cust.id, cust.closest_pup_id])
        farness[cust.id] = round(sum(sorted(distancesf)[:4])/4)
        list_farness_all.append(round(sum(sorted(distancesf)[:4])/4))
    list_farness.reverse()
    list_discounts.reverse()
    list_farness_all.reverse()
    # print("list_farness", list_farness)
    # print("list_discounts", list_discounts)
    # print("list_farness_all", list_farness_all)

    #start by removing discounts offered to customers, who are further than dist_1, then remove cusotmer further than dist_2 in the increasing order of cusotmers' distance
    # next, remove customers located closer then dist_1 in the decreasing order
    dist_1 = min(list_farness) + (max(list_farness) - min(list_farness))/4
    print(dist_1)
    dist_2 = min(list_farness) + (max(list_farness) - min(list_farness))*2/3
    dict_cust_remove = {'middle':{}, 'farthest':{}, 'closest':{}}
    policy = 0
    for cust in instance.customers:
        if policy_rs & (1 << int(cust.id-1)):
            if instance.distanceMatrix[cust.id, cust.closest_pup_id] < dist_1:
                if cust.shipping_fee < farness[cust.id]:
                    policy += (1 << int(cust.id - 1))
                    #print("here1", cust.id, cust.shipping_fee ,farness[cust.id])
            else:
                if cust.shipping_fee <(1-cust.prob_home)*(1-cust.prob_home)*farness[cust.id]:
                    policy += (1 << int(cust.id - 1))
                    #print("here2", cust.id, cust.shipping_fee,(cust.prob_home),farness[cust.id])
    #print(bin(policy))
    return policy

def policy_remote_rs(instance):
    policy_rs, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)
    # painter = Painter()
    # painter.printVertex(instance)
    # solverType = 'Gurobi'
    # farness = {}
    #
    # list_farness = []
    # list_discounts = []
    # list_farness_all = []
    # for cust in instance.customers:
    #     list_farness.append(instance.distanceMatrix[cust.id, cust.closest_pup_id])
    #     list_discounts.append(cust.shipping_fee)
    #     distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
    #                        j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [instance.distanceMatrix[cust.id, instance.depot.id]]
    #     #print(cust.id, sorted(distancesf), instance.distanceMatrix[cust.id, cust.closest_pup_id])
    #     farness[cust.id] = round(sum(sorted(distancesf)[:4])/4)
    #     list_farness_all.append(round(sum(sorted(distancesf)[:4])/4))
    # list_farness.reverse()
    # list_discounts.reverse()
    # list_farness_all.reverse()
    # # print("list_farness", list_farness)
    # # print("list_discounts", list_discounts)
    # # print("list_farness_all", list_farness_all)
    #
    # #start by removing discounts offered to customers, who are further than dist_1, then remove cusotmer further than dist_2 in the increasing order of cusotmers' distance
    # # next, remove customers located closer then dist_1 in the decreasing order
    # dist_1 = min(list_farness) + (max(list_farness) - min(list_farness))/4
    # print(dist_1)
    # dist_2 = min(list_farness) + (max(list_farness) - min(list_farness))*2/3
    # dict_cust_remove = {'middle':{}, 'farthest':{}, 'closest':{}}
    # policy = 0
    # for cust in instance.customers:
    #     if policy_rs & (1 << int(cust.id-1)):
    #         if instance.distanceMatrix[cust.id, cust.closest_pup_id] < dist_1:
    #             if cust.shipping_fee < farness[cust.id]:
    #                 policy += (1 << int(cust.id - 1))
    #                 #print("here1", cust.id, cust.shipping_fee ,farness[cust.id])
    #         else:
    #             if cust.shipping_fee <(1-cust.prob_home)*(1-cust.prob_home)*farness[cust.id]:
    #                 policy += (1 << int(cust.id - 1))
    #                 #print("here2", cust.id, cust.shipping_fee,(cust.prob_home),farness[cust.id])
    # #print(bin(policy))
    return policy_rs


def policy_remote_customers_old(instance):
    painter = Painter()
    painter.printVertex(instance)
    solverType = 'Gurobi'
    farness = {}


    list_farness = []
    list_discounts = []
    list_farness_all = []
    for cust in instance.customers:
        list_farness.append(instance.distanceMatrix[cust.id, cust.closest_pup_id] / 10)
        list_discounts.append(cust.shipping_fee)
        distancesf = [instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                           j is not cust] + [instance.distanceMatrix[cust.id, j.id] for j in instance.pups] + [instance.distanceMatrix[cust.id, instance.depot.id]]
        print(cust.id, sorted(distancesf))
        list_farness_all.append(round(sum(sorted(distancesf)[:4])/4))
        # list_farness_all.append(sum(instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
        #                                   j is not cust) + \
        #                               sum(instance.distanceMatrix[cust.id, j.id] for j in instance.pups) + \
        #                               instance.distanceMatrix[cust.id, instance.depot.id])
    print(list_farness)
    print(list_discounts)
    print(list_farness_all)
    #start by removing discounts offered to customers, who are further than dist_1, then remove cusotmer further than dist_2 in the increasing order of cusotmers' distance
    # next, remove customers located closer then dist_1 in the decreasing order
    dist_1 = min(list_farness) + (max(list_farness) - min(list_farness))/3
    dist_2 = min(list_farness) + (max(list_farness) - min(list_farness))*2/3
    dict_cust_remove = {'middle':{}, 'farthest':{}, 'closest':{}}
    for cust in instance.customers:
        farness = instance.distanceMatrix[cust.id, cust.closest_pup_id]/10
        if farness >= dist_1 and farness< dist_2:
            dict_cust_remove['middle'][cust.id] = instance.distanceMatrix[cust.id, cust.closest_pup_id] / 10
        elif farness >= dist_2:
            dict_cust_remove['farthest'][cust.id] = instance.distanceMatrix[cust.id, cust.closest_pup_id] / 10
        else:
            dict_cust_remove['closest'][cust.id] = instance.distanceMatrix[cust.id, cust.closest_pup_id] / 10

    policy, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)
    print("rs", bin(policy))

    #policy that gives discount to all csutomers in rs and to the closest customers
    policy = 0
    for cust_id in dict_cust_remove['closest']:
        if not policy & (1 << int(cust_id-1)):
            policy += (1 << int(cust_id-1))


    if 2 ** bitCount(policy) < constants.SAMPLE_SIZE:
        best_value = one_policy_cost_estimation(instance=instance, policy=policy, solverType=solverType)
    else:
        best_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
                                                                            policy=policy, solverType=solverType)
    # Flag is true if we want to check more policies
    print(dict_cust_remove['farthest'])
    print(dict_cust_remove['middle'])
    print(dict_cust_remove['closest'])
    flag = True
    for cutomer_offer_incentive, v in sorted(dict_cust_remove['middle'].items(), key=lambda item: item[1]):

        if flag:
            print("middle", cutomer_offer_incentive, v)
            if True:
            #if policy & (1 << int(cutomer_offer_incentive - 1)):
                new_policy = policy  + (1 << int(cutomer_offer_incentive - 1))
                if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
                    test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
                else:
                    test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
                                                                                          policy=new_policy,
                                                                                          solverType=solverType,
                                                                                          sample_size=constants.REMOTE_HEURISTIC_SAMPLE)
                print("current", best_value, "policy", bin(policy))
                print("test", test_value, "policy", bin(new_policy))
                if test_value[0] <= best_value[0]:
                    best_value = test_value
                    policy = new_policy
                else:
                    flag = False
    flag = True
    for cutomer_offer_incentive, v in sorted(dict_cust_remove['farthest'].items(), key=lambda item: item[1]):
        if flag:
            print("farthest", cutomer_offer_incentive, v)
            #if policy & (1 << int(cutomer_offer_incentive - 1)):
            if True:
                new_policy = policy +(1 << int(cutomer_offer_incentive - 1))
                #new_policy = policy & ~(1 << int(cutomer_offer_incentive - 1))
                if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
                    test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
                else:
                    test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
                                                                                          policy=new_policy,
                                                                                          solverType=solverType,
                                                                                          sample_size=constants.REMOTE_HEURISTIC_SAMPLE)
                print("current", best_value,"policy", bin(policy))
                print("test", test_value,"policy", bin(new_policy))
                if test_value[0] <= best_value[0]:
                    best_value = test_value
                    policy = new_policy

    flag = True
    for cutomer_offer_incentive, v in sorted(dict_cust_remove['closest'].items(), key=lambda item: item[1], reverse=True):
        if flag:
            print("closest", cutomer_offer_incentive, v)
            if policy & (1 << int(cutomer_offer_incentive - 1)):
                new_policy = policy & ~(1 << int(cutomer_offer_incentive - 1))
                if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
                    test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
                else:
                    test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
                                                                                          policy=new_policy,
                                                                                          solverType=solverType,
                                                                                          sample_size=constants.REMOTE_HEURISTIC_SAMPLE)
                # print("current", best_value,"policy", bin(policy))
                # print("test", test_value,"policy", bin(new_policy))
                if test_value[0] <= best_value[0]:
                    best_value = test_value
                    policy = new_policy
                else:
                    flag = False
    print(bin(policy))
    #
    # flag= True
    # for cutomer_offer_incentive, v in sorted(dict_cust_remove[3] .items(), key=lambda item: item[1]):
    #     if flag:
    #         print("here 3", cutomer_offer_incentive, v)
    #         new_policy = policy +  (1 << int(cutomer_offer_incentive-1))
    #         if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
    #             test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
    #         else:
    #             test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
    #                                                             policy=new_policy, solverType=solverType,
    #                                                             sample_size =  constants.REMOTE_HEURISTIC_SAMPLE)
    #         print("current", best_value,"policy", bin(policy))
    #         print("test", test_value,"policy", bin(new_policy))
    #         if test_value[0] <= best_value[0]:
    #             best_value = test_value
    #             policy = new_policy
    #         else:
    #             flag = False
    # flag = True
    # for cutomer_offer_incentive, v in sorted(dict_cust_remove[1] .items(), key=lambda item: item[1]):
    #     if flag:
    #         print("here 2", cutomer_offer_incentive, v)
    #         new_policy = policy +  (1 << int(cutomer_offer_incentive-1))
    #         if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
    #             test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
    #         else:
    #             test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
    #                                                             policy=new_policy, solverType=solverType,
    #                                                             sample_size =  constants.REMOTE_HEURISTIC_SAMPLE)
    #         print("current", best_value,"policy", bin(policy))
    #         print("test", test_value,"policy", bin(new_policy))
    #         if test_value[0] <= best_value[0]:
    #             best_value = test_value
    #             policy = new_policy
    #         else:
    #             flag = False
    # flag = True
    # for cutomer_offer_incentive, v in sorted(dict_cust_remove[2] .items(), key=lambda item: item[1]):
    #     if flag:
    #         print("here 1", cutomer_offer_incentive, v)
    #         new_policy = policy +  (1 << int(cutomer_offer_incentive-1))
    #         if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
    #             test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
    #         else:
    #             test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
    #                                                             policy=new_policy, solverType=solverType,
    #                                                             sample_size =  constants.REMOTE_HEURISTIC_SAMPLE)
    #         print("current", best_value,"policy", bin(policy))
    #         print("test", test_value,"policy", bin(new_policy))
    #         if test_value[0] <= best_value[0]:
    #             best_value = test_value
    #             policy = new_policy
    #         else:
    #             flag = False
    # print(bin(policy))




    return policy
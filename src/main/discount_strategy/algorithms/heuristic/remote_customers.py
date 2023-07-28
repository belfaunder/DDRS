from pathlib import Path
import sys
from sys import argv
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

def policy_remote_customers(instance):
    #painter = Painter()
    #painter.printVertex(instance)
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
        #print(cust.id, sorted(distancesf))
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
    #print(dist_1)
    dist_2 = min(list_farness) + (max(list_farness) - min(list_farness))*2/3
    dict_cust_remove = {'middle':{}, 'farthest':{}, 'closest':{}}
    policy = 0
    for cust in instance.customers:
        if instance.distanceMatrix[cust.id, cust.closest_pup_id] < dist_1:
            if cust.shipping_fee < farness[cust.id]:
                policy += (1 << int(cust.id - 1))
                #print("here1", cust.id, cust.shipping_fee ,farness[cust.id])
        else:
            if cust.shipping_fee <cust.prob_home*farness[cust.id]:
                policy += (1 << int(cust.id - 1))
                #print("here2", cust.id, cust.shipping_fee,(cust.prob_home),farness[cust.id])
    #print(bin(policy))
    return policy


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
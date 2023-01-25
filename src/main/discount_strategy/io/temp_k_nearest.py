#!/usr/bin/python3
from pathlib import Path
import sys
import os
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"io")
sys.path.insert(2, path_to_io)
import OCVRPParser

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"algorithms","exact")
sys.path.insert(3, path_to_exact)

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(1, path_to_util)
import constants
from bit_operations import bitCount
from collections import OrderedDict

def dist(vertex1,vertex2):
    return int(math.sqrt((vertex1.xCoord -vertex2.xCoord ) ** 2 +
                  (vertex1.yCoord - vertex2.yCoord) ** 2) + 0.5)

def check_k_nearest(policy_bab, instance, order_customer):
    num_disc_bab = bitCount(policy_bab)
    folder = os.path.join(constants.PATH_TO_DATA, "data", "i_VRPDO_old")
    file_instance = os.path.join(folder, instance + ".txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()

    number_closest = 2
    #int(OCVRPInstance.NR_CUST / 5)
    #painter = Painter()
    #painter.printVertex(OCVRPInstance)

    offered_disc_bab = []
    for i in range(OCVRPInstance.NR_CUST):
        mask = 1 << i
        if  (policy_bab & mask):
            offered_disc_bab.append(i+1)

    dict_customer = OrderedDict()
    for c in OCVRPInstance.customers:
        dict_customer[c.id] = 0
        #dict_customer[c.id] = dist(OCVRPInstance.pup, c)+dist(OCVRPInstance.depot, c)
        dict_neighbours = OrderedDict()
        for neighbour in OCVRPInstance.customers:
            if neighbour is not c:
                dict_neighbours[neighbour] =  dist(neighbour, c)

        count = 0
        dict_neighbours = sorted(dict_neighbours.items(), key=lambda x: x[1])
        for k,v in dict_neighbours:
            dict_customer[c.id] += v
            count += 1
            if count >= number_closest:
                break

    dict_customer = sorted(dict_customer.items(), key=lambda x: x[1], reverse=True)

    count = 0
    intersection = 0
    group_considered = num_disc_bab
    for k, v in dict_customer:
        count +=1
        if count == order_customer:
            if k in offered_disc_bab:
                return 1
            else:
                return 0
        # if count <= group_considered:
        #     if k in offered_disc_bab:
        #         intersection +=1
    if order_customer > OCVRPInstance.NR_CUST:
        return -1
    elif group_considered:
        return intersection/group_considered
    else:
        return -1

def check_k_nearest_distance(policy_bab, instance,discount_rate,  distance_new, distance_old):
    folder = os.path.join(constants.PATH_TO_DATA, "data", "i_VRPDO_old")
    file_instance = os.path.join(folder, instance + ".txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    number_closest = 2

    offered_disc_bab = []
    for i in range(OCVRPInstance.NR_CUST):
        mask = 1 << i
        if  (policy_bab & mask):
            offered_disc_bab.append(i+1)

    dict_customer = OrderedDict()
    for c in OCVRPInstance.customers:
        dict_customer[c.id] = 0
        dict_customer[c.id] = dist(OCVRPInstance.pup, c)+dist(OCVRPInstance.depot, c)
        #dict_customer[c.id] = dist(OCVRPInstance.depot, c)

        # dict_neighbours = OrderedDict()
        # for neighbour in OCVRPInstance.customers:
        #     if neighbour is not c:
        #         dict_neighbours[neighbour] =  dist(neighbour, c)
        # dict_neighbours[OCVRPInstance.pup] =  dist(OCVRPInstance.pup, c)
        # dict_neighbours[OCVRPInstance.depot] = dist(OCVRPInstance.depot, c)
        # count = 0
        # dict_neighbours = sorted(dict_neighbours.items(), key=lambda x: x[1])
        # for k,v in dict_neighbours:
        #     dict_customer[c.id] += v
        #     count += 1
        #     if count >= number_closest:
        #         break

    dict_customer = sorted(dict_customer.items(), key=lambda x: x[1])

    count = 0
    number_customers_inrange = 0
    for k, v in dict_customer:
        if v >= distance_old and v<distance_new:
            number_customers_inrange += 1
            if k in offered_disc_bab:
                count += 1
    if number_customers_inrange:
        return count/number_customers_inrange
    else:
        return -1

if __name__ == "__main__":
    folder = os.path.join(constants.PATH_TO_DATA, "output", "VRPDO")
    df = pd.read_csv(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_disc.csv"))
    #df = df[(df.discount_rate ==0.2)].copy()
    df['fraction_of_incentive'] = df.apply(lambda x: bitCount(int(x['policy_bab_ID'])) / x['nrCust'], axis=1)
    average_prob = df['fraction_of_incentive'].mean()*100
    prob_customers = []
    order_customers = []
    order_customers_dist = []
    # for order_customer in range(1,15):
    #     df['intersection'] = df.apply(lambda x: check_k_nearest(int(x['policy_bab_ID']) , x['instance'],order_customer ), axis = 1)
    #     df_temp = df[df['intersection'] >=0].copy()
    #     prob_customers.append(df_temp['intersection'].mean()*100)
    #     order_customers.append(order_customer)
    #     #order_customers_dist.append()
    # fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    # sns.lineplot(ax=axes, x=order_customers, y=prob_customers)
    # plt.axhline(y=average_prob, color='r', linestyle='--', label='% of customers with inc.')
    # # plt.axhline(y = rsValue, color='r', linestyle='--', label = 'expected value of deterministic ring-star')
    # plt.title("Probability that offering an incentive to the customer is optimal")
    # # plt.xlabel("Customer in the descending order of the distance to pup+depot")
    # plt.xlabel("Customer in the descending order of the distance to 2 nearest neighbours")

    step = 3
    distance_old = step
    for distance in range(step, 40, step):
        order_customers.append(distance_old)
        df['intersection'] = df.apply(lambda x: check_k_nearest_distance(int(x['policy_bab_ID']) , x['instance'],x['discount_rate'], distance, distance_old ), axis = 1)
        distance_old = distance
        df_temp = df[df['intersection'] >=0].copy()
        prob_customers.append(df_temp['intersection'].mean()*100)

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    sns.set()
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    sns.barplot(ax=axes, x = order_customers, y = prob_customers, color='b')
    #sns.lineplot(ax=axes, x = order_customers, y = prob_customers)
    plt.axhline(y=average_prob, color='r', linestyle='--', label='% of customers with inc.')
    plt.title("Probability that offering an incentive to the customer is optimal")
    plt.xlabel("Distance to 2 nearest neighbours")
    plt.xlabel("Distance to PUP+DEPOT")
    plt.ylabel("Probability")
    plt.legend()
    plt.ylim(0, )
    plt.show()
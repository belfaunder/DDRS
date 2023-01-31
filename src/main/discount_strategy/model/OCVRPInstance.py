from pathlib import Path
import sys
import os
from collections import OrderedDict

import numpy as np

from src.main.discount_strategy.util import constants
prefix="tag: "

class OCVRPInstance:
    '''
    /* Vehicle source depot (where all vehicles depart) */
    public final Depot sourceDepot;
    /* Vehicle source depot (where all vehicles return) */
    public final Depot targetDepot;
    /* Pickup points */
    public final List<PickupPoint> pickupPoints;

    /* Vehicle capacity */
    public final int VEHIC_CAP;

    /* Flat rate shipping fee */
    public final int FLAT_RATE_SHIPPING_FEE;

    /* Undirected routing graph. Edge weights are distances between nodes */
    public final Graph<Node, DefaultWeightedEdge> routingGraph;
    /* Directed assignment graph. An arc c -> v indicates that customer c can be assigned to node v, where is either a pickup point or v=c (the customer is assigned to itself). Edge weights are the assignment costs*/
    public final Graph<Node, DefaultWeightedEdge> assignmentGraph;
    '''
    # Dictionaries of the upper bound and lower bound on the insertion cost of each vertex
    ubInsertion = {}
    lbInsertion = {}
    painter = None
    routeCost = OrderedDict()


    def __init__(self, name, customers, depot, pups, distanceMatrix):
        # Instance name
        self.name = name
        # Customers
        self.customers = customers
        self.depot = depot
        self.pups = pups
        # Number of customers
        self.NR_CUST = len(customers)
        self.NR_PUP = len(pups)

        self.distanceMatrix = distanceMatrix

        #TODO:where do we use it? Rethink
        maxNumDeviated = {}
        for i in range(self.NR_CUST+1):
            maxNumDeviated[i] = self.NR_CUST
        self.maxNumDeviated =  maxNumDeviated

        # call for customer is i
        self.p_home = np.zeros(self.NR_CUST+1)
        self.p_pup = np.zeros(self.NR_CUST + 1)
        self.p_delta = np.zeros(self.NR_CUST + 1)
        self.p_pup_delta = np.zeros(self.NR_CUST + 1)
        self.shipping_fee = np.zeros(self.NR_CUST+1)
        self.neighbour = {}
        self.neighbourhood_enlarged = {}
        for cust in customers:
            self.p_home[cust.id] = cust.prob_home
            self.p_pup[cust.id] = cust.prob_pup
            self.p_delta[cust.id] = 1 - cust.prob_home - cust.prob_pup
            self.p_pup_delta[cust.id] = 1 - cust.prob_home
            self.shipping_fee[cust.id] = cust.shipping_fee
            self.neighbour[cust.id] = []

            for other_cust in customers:
                if other_cust is not cust:
                    if  cust.shipping_fee == min([cust.shipping_fee, other_cust.shipping_fee]):
                        if distanceMatrix[cust.id,other_cust.id] < (cust.shipping_fee*(1+cust.prob_pup/(1-cust.prob_pup-cust.prob_home+constants.EPS)) -
                                            other_cust.shipping_fee*other_cust.prob_pup*(1-other_cust.prob_home)/
                                                                    ((1-other_cust.prob_pup-other_cust.prob_home+constants.EPS)*(1-cust.prob_home+constants.EPS)) )/2:
                            self.neighbour[cust.id].append(other_cust.id)
                    else:
                        if distanceMatrix[cust.id,other_cust.id] < (other_cust.shipping_fee*(1+other_cust.prob_pup/(1-other_cust.prob_pup-other_cust.prob_home+constants.EPS)) -
                                            cust.shipping_fee*cust.prob_pup*(1-cust.prob_home)/
                                                                    ((1-cust.prob_pup-cust.prob_home+constants.EPS)*(1-other_cust.prob_home+constants.EPS)) )/2:
                            self.neighbour[cust.id].append(other_cust.id)

    def __str__(self):
        s = prefix +"Instance: " + self.name + "\n" + \
            prefix+ "Number_of_customers: " + str(self.NR_CUST) + "\n" + \
            prefix + "Number_of_pups: " + str(self.NR_PUP) + "\n" + \
            prefix + "Epsilon_precision: " + str(constants.EPSILON* 100)  + " %\n"


        # "Number of pickup points: "+pickupPoints.size()+"\n"+
        # "Vehicle capacity: "+VEHIC_CAP+"\n"+
        # prefix + "Deviation probability: " + str(self.deviationProbability*100) + " %" + "\n" + \
        # prefix + "Flat rate shipping fee: " + str(self.FLAT_RATE_SHIPPING_FEE) + "\n" +\
        return s

    def set_enlarged_neighbourhood(self):
        self.neighbourhood_enlarged[0]=[]
        for cust in self.customers:
            self.neighbourhood_enlarged[cust.id] = []

            for other_cust in self.customers:
                if other_cust is not cust:
                    if  cust.shipping_fee == min([cust.shipping_fee, other_cust.shipping_fee]):
                        if self.distanceMatrix[cust.id,other_cust.id] < constants.NEIGHBOURHOOD_HEURISTIC *(cust.shipping_fee*(1+cust.prob_pup/(1-cust.prob_pup-cust.prob_home+constants.EPS)) -
                                            other_cust.shipping_fee*other_cust.prob_pup*(1-other_cust.prob_home)/
                                                                    ((1-other_cust.prob_pup-other_cust.prob_home+constants.EPS)*(1-cust.prob_home+constants.EPS)) )/2:
                            self.neighbourhood_enlarged[cust.id].append(other_cust.id)
                    else:
                        if self.distanceMatrix[cust.id,other_cust.id] < constants.NEIGHBOURHOOD_HEURISTIC *(other_cust.shipping_fee*(1+other_cust.prob_pup/(1-other_cust.prob_pup-other_cust.prob_home+constants.EPS)) -
                                            cust.shipping_fee*cust.prob_pup*(1-cust.prob_home)/
                                                                    ((1-cust.prob_pup-cust.prob_home+constants.EPS)*(1-other_cust.prob_home+constants.EPS)) )/2:
                            self.neighbourhood_enlarged[cust.id].append(other_cust.id)



    def calculateInsertionBounds(self):
        ubInsertionDict = {}
        lbInsertionDict = {}

        for customer in self.customers:
            ubInsertionDict[customer.id] = self.sort_ub(customer)
            lbInsertionDict[customer.id] = self.sort_lb(customer)

        self.ubInsertion = ubInsertionDict
        self.lbInsertion = lbInsertionDict
        # list of all possible ub insertion cost that are cheapper than min_default = dist[0,node] or dist[n+1,node]

    def sort_ub(self, customer):
        dist = self.distanceMatrix
        min_default_ub = 2*dist[0, customer.id]
        min_insertion = {}

        min_insertion[0] = min_default_ub
        for i in range(1, self.NR_CUST + 1):
        #for i in range(1, self.NR_CUST  + self.NR_PUP + 1):
            if i != customer.id:
                if dist[i, customer.id] * 2 < min_default_ub:
                    min_insertion[i] = dist[i, customer.id]*2
        sorted_insertion = {k: v for k, v in sorted(min_insertion.items(), key=lambda item: item[1])}
        return sorted_insertion

    # list of all possible lb insertion cost that are cheaper than min_default = dist[0,node] or dist[n+1,node]
    def sort_lb(self, customer):
        dist = self.distanceMatrix
        min_insertion = {}
        min_insertion[0, 0] = 2 * dist[0, customer.id]
        for i in range(0, self.NR_CUST + self.NR_PUP+1):
            if i != customer.id:
                for j in range(i, self.NR_CUST + self.NR_PUP+1):
                    if j != customer.id:
                        if dist[i, customer.id] + dist[customer.id, j] - dist[i, j] < min_insertion[0, 0]:
                            min_insertion[i, j] = max(0, dist[i, customer.id] + dist[customer.id, j] - dist[i, j])
        sorted_insertion = {k: v for k, v in sorted(min_insertion.items(), key=lambda item: item[1])}

        p_home, p_pup = {}, {}
        for cust in self.customers:
            p_home[cust.id] = max(cust.prob_home, constants.EPS)
            p_pup[cust.id] = max(cust.prob_pup, constants.EPS)
        p_home[0] = constants.EPS
        p_pup[0] = constants.EPS

        # in the worst case we to not visit pickup points
        for i in range(self.NR_CUST + 1, self.NR_CUST + self.NR_PUP+1):
            prob_temp = 1
            for pup in self.pups:
                for cust in pup.closest_cust_id:
                    prob_temp *= p_home[cust]

            p_home[i] = prob_temp
            p_pup[i] = 1 - prob_temp

        # We want to delete any benchmanrk that will not be used (because there is a better one)
        previous_insertion = {}
        p_left = 1
        for (i, j) in sorted_insertion:
            if p_left > 0:
                if [i, j] != [0, 0]:
                    prob_current = 1
                    if i == j:
                        prob_current *= p_home[i]
                    else:
                        if i !=0:
                            prob_current *= p_home[i]
                        if j !=0:
                            prob_current *= p_home[j]

                    for (m, k) in previous_insertion:
                        if m == i or m == j:
                            if (k !=0):
                                prob_current *= (1 - p_home[k])
                            else:
                                prob_current *= 0
                        if k == i or k == j:
                            if m !=0:
                                prob_current *= (1 - p_home[m])
                            else:
                                prob_current *= 0
                    # here can check if prob_current > epsilon
                    if prob_current > 0:
                        previous_insertion[i, j] = sorted_insertion[i,j]
                        p_left -= prob_current
                else:
                    previous_insertion[i, j] = sorted_insertion[i,j]
                    p_left = 0
        return previous_insertion

    def setMaxNumDeviated(self, p_dev, n):
        # in case of  different probavilities have to approximate ( probably with the smallest prob)
        MaxNumDevList = []
        for i in range(n+1):
            # the max of binomial distribution
            #for larger instance Moivre Laplace approximation can be used
            minNumDev = max(int((i+1)*p_dev - 1),0)
            numDev = minNumDev
            #for numDev in range(minNumDev, i+1):
            #    if ((1 - p_dev) ** (i - numDev) * p_dev ** numDev) * comb(i,numDev) <= max(constants.EPSILON*5, constants.EPS*5) :
            #        break
            MaxNumDevList.append(numDev)

        return MaxNumDevList



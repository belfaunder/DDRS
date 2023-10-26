from pathlib import Path
import sys
import os
import math
from collections import defaultdict
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.model.OCVRPInstance import OCVRPInstance
from src.main.discount_strategy.model.Customer import Customer
from src.main.discount_strategy.model.Depot import Depot
from src.main.discount_strategy.model.PUP import PUP
from src.main.discount_strategy.util import distance

# module method
# parse the csv file given the file name, number of customer, deviation probability
def parse(file_instance):
    # nr_cust,  flat_rate_shipping_fee, deviationProb
    #file_name, pup_node
    #OCVRPParser.parse(file_name="R101.csv", nr_cust=8,
    #                                  pup_node = 16, flat_rate_shipping_fee = 9/0.7, deviationProb = 30)
    def read_file(file_instance):
        #file_path = os.path.join((Path(os.path.abspath(__file__)).parents[6]),file_instance)
        with open(file_instance, 'r',  encoding='utf-8') as file:
            lines = file.readlines()
            name = lines[0].split()[1]
            nr_cust = int(lines[1].split()[1])
            nr_pup = int(lines[2].split()[1])

            #flat_rate_shipping_fee = float(general_info[1])
            #deviationProb = float(general_info[2])/100

            vertices = {}
            for line in lines[12+nr_pup:12+nr_pup+nr_cust]:
                custInfo = line.split()
                vertices[int(custInfo[0])] = {}
                vertices[int(custInfo[0])]['x'] = float(custInfo[1])
                vertices[int(custInfo[0])]['y'] = float(custInfo[2])
                vertices[int(custInfo[0])]['prob_home'] = 1 - float(custInfo[3])
                vertices[int(custInfo[0])]['prob_pup'] = 0
                vertices[int(custInfo[0])]['shipping_fee'] = float(custInfo[4])

            vertices[0] = {}
            vertices[0]['x'] = float(lines[7].split()[1])
            vertices[0]['y'] = float(lines[7].split()[2])

            for iter in range(nr_pup):
                vertices[nr_cust + iter+1] = {}
                vertices[nr_cust + iter+1]['x'] = float(lines[7 + 3 + iter].split()[1])
                vertices[nr_cust + iter+1]['y'] = float(lines[7 + 3 + iter].split()[2])

        return nr_cust, nr_pup, vertices, name

    # retrurn matrix of distances with depot in the node 0 and many vitual pups
    def setDistances(nodes, n, nr_pup):
        d = {}
        i = 0
        while i < n + 1+nr_pup:
            d[(i, i)] = 0
            j = i + 1
            while j < n + 1 + nr_pup:
                #d[(i, j)] = int(10 * math.sqrt((nodes[i]['x'] - nodes[j]['x']) ** 2 +
                #                               (nodes[i]['y'] - nodes[j]['y']) ** 2)) / 10
                #d[(i, j)] = math.sqrt((nodes[i]['x'] - nodes[j]['x']) ** 2 +
                #                               (nodes[i]['y'] - nodes[j]['y']) ** 2)
                #EUC_2D : rounded Euclidean distances metric from TSPLIN format
                d[(i, j)] = distance.set_distance(nodes[i]['x'],nodes[i]['y'], nodes[j]['x'],nodes[j]['y'])
                j += 1
            # while j < n + constants.FLEET_SIZE + 1:
            #     print("here", i,j)
            #     d[(i, j)] = d[(i, j - 1)]
            #     j += 1
            j = 0
            while j < i:
                d[(i, j)] = d[(j, i)]
                j += 1
            i += 1
        j = 0
        while i < n + constants.FLEET_SIZE + 1:
            d[i, i] = 0
            while j < n + constants.FLEET_SIZE + 1:
                d[(i, j)] = d[(i - 1, j)]
                j += 1
            j = 0
            i += 1
        #check triangle inequality
        # count = 0
        # for i in range(n + 1+nr_pup):
        #     for j in range(n + 1+nr_pup):
        #         for k in range(n + 1+nr_pup):
        #             if (d[(i, j)] > d[(i, k)] + d[(k, j)]+0.1):
        #                 count+=1
        #                 print("triangle inequality not satisfied", i,j,k, nodes[i]['x'],nodes[i]['y'],nodes[j]['x'],nodes[j]['y'], nodes[k]['x'],nodes[k]['y'])
        #
        # print("TOTAL", count)
        #while i < n + constants.FLEET_SIZE + 1:
        #    j = 0
        #    s = ""
        #    while j < n + constants.FLEET_SIZE + 1:
        #        s += str((d[(i, j)]))
        #        s += str('   ')
        #        j += 1
            #    print(s)
         #   i += 1
        # for i in range(n+2):
        #    del d[(i,i)]
        return d

    nr_cust, nr_pup, vertices, name = read_file(file_instance)
    depot = Depot(vertices[0]['x'], vertices[0]['y'], 0)
    pups = []
    for iter in range(nr_pup):
        closest_cust_id = []
        pup = PUP(vertices[nr_cust + 1 + iter]['x'], vertices[nr_cust + 1 + iter]['y'], nr_cust + 1 + iter, closest_cust_id, iter)
        pups.append(pup)

    verticesReenum = vertices

    customers = []
    for i in range(1, nr_cust+1):
        closest_pup_id = nr_cust + 1
        distance_to_closest_pup = 10**5
        for pup in pups:

            distance_temp = math.sqrt((pup.xCoord - verticesReenum[i]['x'] ) ** 2 +
                                               (pup.yCoord - verticesReenum[i]['y'] ) ** 2)
            if distance_temp < distance_to_closest_pup:
                distance_to_closest_pup = distance_temp
                closest_pup_id = pup.id


        customer = Customer(verticesReenum[i]['x'] , verticesReenum[i]['y'], i,verticesReenum[i]['prob_home'],verticesReenum[i]['prob_pup'],
                            verticesReenum[i]['shipping_fee'], closest_pup_id  )

        customers.append(customer)
    for pup in pups:
        for customer in customers:
            if customer.closest_pup_id==pup.id:
                pup.closest_cust_id.append(customer.id)
        #print(pup.id, pup.closest_cust_id)
    distanceMatrix = setDistances(verticesReenum, nr_cust, nr_pup)

    instance = OCVRPInstance(name, customers,depot, pups, distanceMatrix)

    return instance
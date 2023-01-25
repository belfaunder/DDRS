from pathlib import Path
import sys
import os
import math
from collections import defaultdict
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(1, path_to_util)
import constants

path_to_model = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"model")
sys.path.insert(2, path_to_model)
from OCVRPInstance import OCVRPInstance
from Customer import Customer
from Depot import Depot
from PUP import PUP
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
            #flat_rate_shipping_fee = float(general_info[1])
            #deviationProb = float(general_info[2])/100
            vertices = {}

            for line in lines[12:12+nr_cust]:
                custInfo = line.split()
                vertices[int(custInfo[0])] = {}
                vertices[int(custInfo[0])]['x'] = float(custInfo[1])
                vertices[int(custInfo[0])]['y'] = float(custInfo[2])
                vertices[int(custInfo[0])]['prob_home'] = max(float(custInfo[3]), constants.EPS/2)
                vertices[int(custInfo[0])]['prob_pup'] = max(float(custInfo[4]), constants.EPS/2)
                vertices[int(custInfo[0])]['shipping_fee'] = float(custInfo[5])


            vertices[0] = {}
            vertices[0]['x'] = float(lines[6].split()[1])
            vertices[0]['y'] = float(lines[6].split()[2])

            vertices[nr_cust + 1] = {}
            vertices[nr_cust + 1]['x'] = float(lines[6 + 3].split()[1])
            vertices[nr_cust + 1]['y'] = float(lines[6 + 3].split()[2])

        return nr_cust, vertices, name

    # retrurn matrix of distances with depot in the node 0 and many vitual pups
    def setDistances(nodes, n):
        d = {}
        i = 0
        while i < n + 2:
            d[(i, i)] = 0
            j = i + 1
            while j < n + 2:
                #d[(i, j)] = int(10 * math.sqrt((nodes[i]['x'] - nodes[j]['x']) ** 2 +
                #                               (nodes[i]['y'] - nodes[j]['y']) ** 2)) / 10
                #d[(i, j)] = math.sqrt((nodes[i]['x'] - nodes[j]['x']) ** 2 +
                #                               (nodes[i]['y'] - nodes[j]['y']) ** 2)
                #EUC_2D : rounded Euclidean distances metric from TSPLIN format
                d[(i, j)] = int(math.sqrt((nodes[i]['x'] - nodes[j]['x']) ** 2 +
                                               (nodes[i]['y'] - nodes[j]['y']) ** 2) +0.5)

                j += 1
            while j < n + constants.FLEET_SIZE + 1:
                d[(i, j)] = d[(i, j - 1)]
                j += 1
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
        i = 0
        j = 0

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

    nr_cust, vertices, name = read_file(file_instance)
    depot = Depot(vertices[0]['x'], vertices[0]['y'], 0)
    pup = PUP(vertices[nr_cust + 1]['x'], vertices[nr_cust + 1]['y'], nr_cust + 1)
    customersDistance = defaultdict(int)

    for i in range(1, nr_cust+1):
        customersDistance[i] = math.sqrt((vertices[i]['x'] - vertices[0]['x']) ** 2 +
                                               (vertices[i]['y'] - vertices[0]['y']) ** 2)

    verticesReenum = {}
    for i in [0, nr_cust+1]:
        verticesReenum[i] = vertices[i]
    id = 1
    for i in sorted(customersDistance, key=customersDistance.get, reverse=True):
        verticesReenum[id] = vertices[i]
        id += 1

    customers = []
    for i in range(1, nr_cust+1):
        customer = Customer(verticesReenum[i]['x'] , verticesReenum[i]['y'], i,verticesReenum[i]['prob_home'],verticesReenum[i]['prob_pup'],
                            verticesReenum[i]['shipping_fee']  )
        customers.append(customer)

    distanceMatrix = setDistances(verticesReenum, nr_cust)

    instance = OCVRPInstance(name, customers,depot, pup, distanceMatrix)

    return instance
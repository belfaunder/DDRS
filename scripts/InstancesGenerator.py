from pathlib import Path
import sys
import os
import csv
#import tsplib95
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gzip
import random
import pickle
import math
import matplotlib.pyplot as plt
path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"src","main","discount_strategy","io")
sys.path.insert(1, path_to_io)
import OCVRPParser
path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"src","main","discount_strategy", "algorithms", "exact")
sys.path.insert(2, path_to_exact)
from ring_star_without_TW import ring_star_deterministic_no_TW

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(3, path_to_exact)
from TSPSolver import TSPSolver

from random import sample

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"src", "main","discount_strategy","util")
sys.path.insert(4, path_to_util)
import constants
path_to_data = constants.PATH_TO_DATA

def bitCount(int_type):
    count = 0
    while (int_type):
        int_type &= int_type - 1
        count += 1
    return (count)

# for SolomonC100: DepotNode = 100, PupNode = 2
# for SolomonR100: DepotNode = 100, PupNode = 2
# for SolomonRC100: DepotNode = 100, PupNode = 2: but only for small instances ( <30 customers)
# for berlin52:  DepotNode = 52, PupNode = 1
def generate_3_segments_instance(file_name, instance_type):
    problem = tsplib95.load(file_name)
    nr_custs = [15]
    nr_custs = [10, 11, 12, 13, 14,  16, 17, 18, 19, 20, 21, 22, 23,24,25]
    if instance_type == 'berlin52':
        pupnode, depotnode = 1, 52
        #nr_custs = [20]
        disc_sizes = [180]

    elif  instance_type == 'solomonC100':
        #pupnode, depotnode = 2, 100  #the best alternative of nodes
        pupnode, depotnode = 1, 52
        #disc_sizes = [1, 4, 12]
        #disc_sizes = [2,4,6]
        disc_sizes = [2, 3, 4]
        u_s = [0.2, 0.5, 0.1, 0.12, 0.18, 0.24, 0.4, 0.6, 0.8, 1,2]  #discout parameters relative to the routing cost
        #disc_sizes = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    elif instance_type == 'solomonR100':
        #pupnode, depotnode = 2, 100 #the best alternative of nodes
        pupnode, depotnode = 1, 52
        #disc_sizes = [2, 8, 13]
        disc_sizes = [4,8,10]
        u_s = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7,1,1.5, 2,2.5, 3,3.5, 4, 4.5 ,5]  # discout parameters relative to the routing cost
        #disc_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    elif instance_type == 'solomonRC100':
        #pupnode, depotnode = 2, 100 #the best alternative of nodes
        pupnode, depotnode = 2, 1
        #disc_sizes = [0.5, 2, 10]
        disc_sizes = [2,3,4]
        u_s = [0.25]
        #u_s = [ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,0.4, 0.45,0.5,0.55, 0.6,0.65,0.7, 0.75]  # discout parameters relative to the routing cost
        #disc_sizes = [0.5, 1,2,3,4,5,6,7,8,9,10,11]

    elif instance_type == 'eil51':
        pupnode, depotnode = 47, 1
        disc_sizes = [6]

    elif instance_type == 'eil76':
        pupnode, depotnode = 76, 1
        disc_sizes = [6]

    elif instance_type == 'eil101':
        pupnode, depotnode = 100, 1
        disc_sizes = [6.5]

    # check the cost of TSP per customer(thus to plan appropriate discount size)
    # disc_sizes(%) = [0.05, 0.1, 0.15, 0.25, 0.5, 1]
    #disc_sizes =  [0.5, 1,2,3,4,5,6,7,8,9,10,11]
    #dict_probabilities = {  0.0: [0.0, 0.2, 0.4, 0.6, 0.8],
    ##                        0.2: [0.0, 0.2, 0.4, 0.6],
     #                       0.4: [0.0, 0.2, 0.4],
     #                       0.6: [0.0, 0.2]}
    dict_probabilities = {0.2: [0.2]}
    #dict_probabilities = {0.05: [0.05],
    #                      0.1: [0.1],
    #                      0.35: [0.35]}
    #p_homes = [0.0, 0.2, 0.4, 0.6]
    #p_pups = [0.0, 0.2, 0.4]
    #p_homes = [0.1]
    #p_pups = [0.2]

    #dict_probabilities = {0.2: [0.2]}

    mainDirStorage = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "i_SolomonRC")
    instanceList = os.path.join(mainDirStorage, 'list.txt')
    shuffled_cust_list = os.path.join(mainDir, 'shuffled_customers.txt')

    # for disc_size in disc_sizes:



    for nr_cust in nr_custs:
        #sample_customers = list(range(1,nr_cust + 3))
        #if pupnode in sample_customers:
        #    sample_customers.remove(pupnode)
        #if depotnode in sample_customers:
        #    sample_customers.remove(depotnode)
        #sample_customers = sample_customers[:nr_cust]
        #for disc_size in disc_sizes:
        #    customer_disc_coef = set_disc_coef(problem, sample_customers, nr_cust, pupnode, depotnode, disc_size)
        #    print(disc_size, customer_disc_coef)
        #for p_pup in p_pups:
        for p_pup in dict_probabilities:
            #for p_home in p_homes:
            for p_home in dict_probabilities[p_pup]:
                for u in u_s:
                    with open(shuffled_cust_list, "rb") as file_shuffled:
                        for id_instance in range(10):
                            shuffled_cust_id = pickle.load(file_shuffled)
                            depotnode = shuffled_cust_id[0]
                            pupnode = shuffled_cust_id[1]
                            sample_customers = shuffled_cust_id[2:nr_cust + 2]

                            instanceName = instance_type+'_size_'+str(nr_cust) + '_phome_' + str(p_home) + '_ppup_' + str(p_pup) +'_incrate_'  + str(u) +'_'+str(id_instance)+ '.txt'
                            instanceDir = os.path.join(mainDirStorage, instanceName)
                             # Unpickling
                            with open(instanceList, 'a+', encoding='utf-8') as file:
                                file.write("{}\n".format(instanceName.split('.txt')[0]))

                            with open(instanceDir, 'w+', encoding='utf-8') as file:
                                file.write("NAME: {}\n".format(instanceName))
                                file.write("SIZE: {}\n\n".format(nr_cust))

                                file.write("LOCATION COORDINATES:\n\n")

                                file.write("DEPOT V. XCOORD. YCOORD.\n")
                                file.write(
                                    "{} {} {}\n\n".format(0, problem.node_coords[depotnode][0], problem.node_coords[depotnode][1]))

                                file.write("PUP V. XCOORD. YCOORD.\n")
                                file.write("{} {} {}\n\n".format(nr_cust + 1, problem.node_coords[pupnode][0],
                                                                 problem.node_coords[pupnode][1]))


                                file.write("CUST V. XCOORD. YCOORD. PROB_ALWAYS_HOME PROB_ALWAYS_PUP SHIPPING_FEE\n")
                                customer_ID = 1
                                for i in sample_customers:
                                    customer_distance_to_pup = round(
                                        math.sqrt((problem.node_coords[pupnode][0] - problem.node_coords[i][0]) ** 2 +
                                                  (problem.node_coords[pupnode][1] - problem.node_coords[i][1]) ** 2),2)
                                    #file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
                                    #                                        problem.node_coords[i][1], p_home, p_pup,disc_size*cusomter_disc_coef[i] ))
                                    #file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
                                    #                                        problem.node_coords[i][1], round(random.uniform( max(p_home-0.1, 0), min(p_home+0.1,1)),2),
                                    #                                        round(random.uniform( max(0,p_pup - 0.1), min(1,p_pup+0.1)),2),
                                    #                                        round(customer_distance_to_pup*u)))

                                    file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
                                                                            problem.node_coords[i][1], p_home, p_pup,
                                                                            round(customer_distance_to_pup * u,2)))
                                    customer_ID += 1

def generate_3_segments_instance_zhou(instance_type ):
    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    dict_depot, dict_pickup, dict_customer = adapt_zhou(instance_type)

    #generate a txt file with 10 reshuffled ids of 100 customers
    for id in [0,1,2,3,4]:
        depots = list(range(1, len(dict_depot)))
        customers = list(range(1, len(dict_customer)))
        shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers_'+str(id)+'.txt')
        with open(shuffled_cust_list, 'wb') as file:
            #np.random.shuffle(depots)
            #np.random.shuffle(customers)
            depots = pickle.load(file)
            customers = pickle.load(file)
            nr_cust = 30
            pup_id = set_pickup_point(dict_pickup, dict_customer, customers[:nr_cust], 15)

            pickle.dump(depots, file)
            pickle.dump(customers, file)

    # instance_type = "VRPDO"
    # mainDirStorage = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "i_VRPDO")
    # #nr_custs = [15]
    # #nr_custs = [10, 11, 12, 13, 14,15,  16, 17, 18, 19, 20]
    # #nr_custs = [40]
    #
    # #dict_probabilities = {  0.0: [0.0, 0.2, 0.4, 0.6, 0.8],
    # #                        0.2: [0.0, 0.2, 0.4, 0.6],
    # #                        0.4: [0.0, 0.2, 0.4],
    # #                        0.6: [0.0, 0.2]}
    # dict_probabilities = {0.3:[0.3]}
    # disc_rates = [0.1, 0.07]
    # #disc_rates =  [ 0.05, 0.1, 0.2, 0.3, 0.4,0.5, 0.6,0.7,0.8, 0.9, 1, 1.2,1.5]
    #
    #
    #
    # instanceList = os.path.join(mainDirStorage, 'list.txt')
    # shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers.txt')
    #
    # for id_instance in list(range(5)):
    #     shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers_' + str(id_instance) + '.txt')
    #     for nr_cust in nr_custs:
    #         for p_pup in dict_probabilities:
    #             for p_home in dict_probabilities[p_pup]:
    #                 for u in disc_rates:
    #                     with open(shuffled_cust_list, "rb") as file_shuffled:
    #                         depots_id = pickle.load(file_shuffled)
    #                         customers_id = pickle.load(file_shuffled)
    #                         pup_id = set_pickup_point(dict_pickup, dict_customer, customers_id, nr_cust)
    #
    #                         instanceName = instance_type+'_size_'+str(nr_cust) + '_phome_' + str(p_home) + '_ppup_' + str(p_pup) +'_incrate_'  + str(u) +'_'+str(id_instance)+ '.txt'
    #                         instanceDir = os.path.join(mainDirStorage, instanceName)
    #
    #                         with open(instanceList, 'a+', encoding='utf-8') as file:
    #                             file.write("{}\n".format(instanceName.split('.txt')[0]))
    #
    #                         with open(instanceDir, 'w+', encoding='utf-8') as file:
    #                             file.write("NAME: {}\n".format(instanceName))
    #                             file.write("SIZE: {}\n\n".format(nr_cust))
    #
    #                             file.write("LOCATION COORDINATES:\n\n")
    #
    #                             file.write("DEPOT V. XCOORD. YCOORD.\n")
    #                             file.write(
    #                                 "{} {} {}\n\n".format(0, dict_depot[depots_id[0]]["x"],dict_depot[depots_id[0]]["y"]))
    #
    #                             file.write("PUP V. XCOORD. YCOORD.\n")
    #                             file.write("{} {} {}\n\n".format(nr_cust + 1, dict_pickup[pup_id]["x"],
    #                                                              dict_pickup[pup_id]["y"]))
    #
    #
    #                             file.write("CUST V. XCOORD. YCOORD. PROB_ALWAYS_HOME PROB_ALWAYS_PUP SHIPPING_FEE\n")
    #                             for i in range(1,nr_cust+1):
    #                                 customer_distance_to_pup = math.sqrt(( dict_pickup[pup_id]["x"]- dict_customer[customers_id[i-1]]["x"]) ** 2 +
    #                                               ( dict_pickup[pup_id]["y"] - dict_customer[customers_id[i-1]]["y"]) ** 2)
    #                                 #file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
    #                                 #                                        problem.node_coords[i][1], p_home, p_pup,disc_size*cusomter_disc_coef[i] ))
    #                                 #file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
    #                                 #                                        problem.node_coords[i][1], round(random.uniform( max(p_home-0.1, 0), min(p_home+0.1,1)),2),
    #                                 #                                        round(random.uniform( max(0,p_pup - 0.1), min(1,p_pup+0.1)),2),
    #                                 #                                        round(customer_distance_to_pup*u)))
    #
    #                                 file.write("{} {} {} {} {} {}\n".format(i,  dict_customer[customers_id[i-1]]["x"],
    #                                                                         dict_customer[customers_id[i-1]]["y"], p_home, p_pup,
    #                                                                         round(customer_distance_to_pup * u,2)))

def generate_3_segments_instance_zhou_constant_density(instance_type ):
    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    dict_depot, dict_pickup, dict_customer = adapt_zhou(instance_type)
    instance_type = "VRPDO"
    mainDirStorage = os.path.join(path_to_data, "data", "i_VRPDO_const_density_temp")
    #nr_custs = [15]
    #nr_custs = [10, 11, 12, 13, 14,15,  16, 17, 18, 19, 20, 25,30,35,40,45,50]
    nr_custs = [40]

    # dict_probabilities = {  0.0: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                         0.2: [0.0, 0.2, 0.4, 0.6, 0.8],
    #                         0.4: [0.0, 0.2, 0.4, 0.6],
    #                         0.6: [0.0, 0.2,0.4],
    #                         0.8: [0.0, 0.2],
    #                         1.0: [0.0]}
    dict_probabilities = {0.2:[0.2]}
    #disc_rates = [0.3]

    disc_rates =  [ 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    instanceList = os.path.join(mainDirStorage, 'list.txt')
    shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers.txt')

    for id_instance in list(range(6)):
        shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers_' + str(id_instance) + '.txt')
        for nr_cust in nr_custs:
            for p_pup in dict_probabilities:
                for p_home in dict_probabilities[p_pup]:
                    for u in disc_rates:
                        with open(shuffled_cust_list, "rb") as file_shuffled:
                            nr = 15
                            depots_id = pickle.load(file_shuffled)
                            customers_id = pickle.load(file_shuffled)
                            pup_id = set_pickup_point(dict_pickup, dict_customer, customers_id, 15)

                            # find min and max coordinates
                            minx15 = min(dict_pickup[pup_id]["x"], dict_depot[depots_id[0]]["x"])
                            miny15 = min(dict_pickup[pup_id]["y"], dict_depot[depots_id[0]]["y"])
                            maxx15 = max(dict_pickup[pup_id]["x"], dict_depot[depots_id[0]]["x"])
                            maxy15 = max(dict_pickup[pup_id]["y"], dict_depot[depots_id[0]]["y"])
                            for i in range(1, nr + 1):
                                minx15 = min(minx15, dict_customer[customers_id[i - 1]]["x"])
                                maxx15 = max(maxx15, dict_customer[customers_id[i - 1]]["x"])
                                miny15 = min(miny15, dict_customer[customers_id[i - 1]]["x"])
                                maxy15 = max(maxy15, dict_customer[customers_id[i - 1]]["x"])

                            instanceName = instance_type+'_size_'+str(nr_cust) + '_phome_' + str(p_home) + '_ppup_' + str(p_pup) +'_incrate_'  + str(u) +'_'+str(id_instance)+ '.txt'
                            instanceDir = os.path.join(mainDirStorage, instanceName)

                            with open(instanceList, 'a+', encoding='utf-8') as file:
                                file.write("{}\n".format(instanceName.split('.txt')[0]))
                            # find min and max coordinates
                            minx = min(dict_pickup[pup_id]["x"], dict_depot[depots_id[0]]["x"])
                            miny = min(dict_pickup[pup_id]["y"], dict_depot[depots_id[0]]["y"])
                            maxx = max(dict_pickup[pup_id]["x"], dict_depot[depots_id[0]]["x"])
                            maxy = max(dict_pickup[pup_id]["y"], dict_depot[depots_id[0]]["y"])
                            for i in range(1, nr_cust + 1):
                                minx = min(minx, dict_customer[customers_id[i - 1]]["x"])
                                miny = min(miny, dict_customer[customers_id[i - 1]]["x"])
                                maxx = max(maxx, dict_customer[customers_id[i - 1]]["x"])
                                maxy = max(maxy, dict_customer[customers_id[i - 1]]["x"])

                            #coefficient to multiply side sizes thus to get the same density as the instance with size 15
                            coef = math.sqrt(((nr_cust)*(maxx15 - minx15)*(maxy15 - miny15))/(15*(maxx - minx)*(maxy-miny)))
                            with open(instanceDir, 'w+', encoding='utf-8') as file:
                                file.write("NAME: {}\n".format(instanceName))
                                file.write("SIZE: {}\n\n".format(nr_cust))

                                file.write("LOCATION COORDINATES:\n\n")

                                file.write("DEPOT V. XCOORD. YCOORD.\n")
                                file.write(
                                    "{} {} {}\n\n".format(0, (dict_depot[depots_id[0]]["x"]-minx)*coef,(dict_depot[depots_id[0]]["y"]-miny)*coef))

                                file.write("PUP V. XCOORD. YCOORD.\n")
                                file.write("{} {} {}\n\n".format(nr_cust + 1, (dict_pickup[pup_id]["x"]-minx)*coef,
                                                                 (dict_pickup[pup_id]["y"]-miny)*coef))


                                file.write("CUST V. XCOORD. YCOORD. PROB_ALWAYS_HOME PROB_ALWAYS_PUP SHIPPING_FEE\n")
                                for i in range(1,nr_cust+1):
                                    customer_distance_to_pup = math.sqrt(( dict_pickup[pup_id]["x"]- dict_customer[customers_id[i-1]]["x"]) ** 2 +
                                                  ( dict_pickup[pup_id]["y"] - dict_customer[customers_id[i-1]]["y"]) ** 2)
                                    #file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
                                    #                                        problem.node_coords[i][1], p_home, p_pup,disc_size*cusomter_disc_coef[i] ))
                                    #file.write("{} {} {} {} {} {}\n".format(customer_ID, problem.node_coords[i][0],
                                    #                                        problem.node_coords[i][1], round(random.uniform( max(p_home-0.1, 0), min(p_home+0.1,1)),2),
                                    #                                        round(random.uniform( max(0,p_pup - 0.1), min(1,p_pup+0.1)),2),
                                    #                                        round(customer_distance_to_pup*u)))

                                    file.write("{} {} {} {} {} {}\n".format(i,  round((dict_customer[customers_id[i-1]]["x"]-minx)*coef,2),
                                                                            round((dict_customer[customers_id[i-1]]["y"]-miny)*coef,2), p_home, p_pup,
                                                                            round(customer_distance_to_pup * u,2))) #this devision by 10 is because we multiply all coordinates by 10

def generate_3_segments_instance_zhou_discount_proportional_tsp(instance_type ):
    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    dict_depot, dict_pickup, dict_customer = adapt_zhou(instance_type)
    for dict in [dict_depot, dict_pickup, dict_customer]:
        for i in dict:
            dict[i]['x'] *= 10
            dict[i]['y'] *= 10
    #instance_type = "VRPDODistDepAccept"
    instance_type = "VRPDO"
    mainDirStorage = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP")
    #nr_custs = [30]
    #nr_custs = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    nr_custs = [10, 11,12,13,14,15,16,17,18,19,20]
    disc_rates = [  0.03, 0.06, 0.12]
    #disc_rates = [0.015]
    #nr_custs = [30]
    #dict_probabilities = {0.0:[0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85]}
    dict_probabilities = {0.0: [ 0.1,   0.4,   0.7 ]}
    #dict_probabilities = {0.0: [0.4]}
    #disc_rates = [0.005, 0.01,0.015, 0.02,0.025, 0.03,0.035, 0.04,0.045, 0.05, 0.06, 0.07, 0.08, 0.09]
    instanceList = os.path.join(mainDirStorage, 'list.txt')

    for id_instance in range(10):
        print(id_instance)
        shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers_' + str(id_instance) + '.txt')
        for nr_cust in nr_custs:
            #p_pup, p_home = 0,0
            for p_pup in dict_probabilities:
               for p_home in dict_probabilities[p_pup]:
            #for (l_min, l_max) in [(1,4)]:
            #    for p_min in [0.9]:
            #        for p_av in [ 0.5 ]:
            #            for p_max in [0.1]:
                    if True:
                        if True:
                            for u in disc_rates:
                                for nr_pup in [3]:
                                    #p_pup, p_home = 0, 0
                                    with open(shuffled_cust_list, "rb") as file_shuffled:
                                        depots_id = pickle.load(file_shuffled)
                                        customers_id = pickle.load(file_shuffled)
                                        #pup_ids = set_pickup_point(dict_pickup, dict_customer, customers_id[:nr_cust], 15, nr_pup)
                                        pup_ids = set_pickup_point_preselected( nr_pup,nr_cust, id_instance)
                                        instanceName = instance_type+'_size_'+str(nr_cust) + '_phome_' + str(p_home) + '_ppup_' + \
                                                       str(p_pup) +'_incrate_'  + str(u) +'_nrpup'+str(nr_pup)+'_'+\
                                                       str(id_instance)+ '.txt'

                                        # instanceName = instance_type + '2_size_' + str(nr_cust) + '_pmin_' + str(
                                        #     p_min) + '_pav_' + str(p_av)+'_pmax_' + str(p_max)+'_lmin_' + str(l_min)+'_lmax_' + str(l_max) +'_incrate_'  + str(u) +'_nrpup'+str(nr_pup)+'_'+\
                                        #                 str(id_instance)+ '.txt'

                                        instanceDir = os.path.join(mainDirStorage, instanceName)
                                        with open(instanceList, 'a+', encoding='utf-8') as file:
                                            file.write("{}\n".format(instanceName.split('.txt')[0]))
                                        TSP_cost_per_customer = temp_instance(dict_customer, nr_cust, nr_pup, customers_id, [dict_pickup[pup_id]for pup_id in pup_ids], dict_depot[depots_id[0]])
                                        with open(instanceDir, 'w+', encoding='utf-8') as file:
                                            file.write("NAME: {}\n".format(instanceName))
                                            file.write("SIZE: {}\n".format(nr_cust))
                                            file.write("NUMBER_PUPs: {}\n\n".format(nr_pup))

                                            file.write("LOCATION COORDINATES:\n\n")

                                            file.write("DEPOT V. XCOORD. YCOORD.\n")
                                            file.write(
                                                "{} {} {}\n\n".format(0, dict_depot[depots_id[0]]["x"],dict_depot[depots_id[0]]["y"]))

                                            file.write("PUP V. XCOORD. YCOORD.\n")
                                            iter = nr_cust
                                            for pup_id in pup_ids:
                                                iter +=1
                                                file.write("{} {} {}\n".format(iter, (dict_pickup[pup_id]["x"]),
                                                                             (dict_pickup[pup_id]["y"])))
                                            file.write("\n")

                                            file.write("CUST V. XCOORD. YCOORD. PROB_ALWAYS_HOME PROB_ALWAYS_PUP SHIPPING_FEE\n")

                                            total_discount = 0
                                            for i in range(1,nr_cust+1):
                                                distance_to_the_first_pup = int(math.sqrt((dict_pickup[pup_ids[0]]["x"] - dict_customer[customers_id[i-1]]["x"])** 2 +\
                                                                     (dict_pickup[pup_ids[0]]["y"] - dict_customer[customers_id[i-1]]["y"]) ** 2) + 0.5)

                                                total_discount += round(distance_to_the_first_pup * u * TSP_cost_per_customer,2)

                                            for i in range(1,nr_cust+1):
                                                distance_to_closest_pup = 10 ** 5
                                                for pup_id in pup_ids:
                                                    distance_temp = math.sqrt((dict_pickup[pup_id]["x"] - dict_customer[customers_id[i-1]]["x"])** 2 +\
                                                                     (dict_pickup[pup_id]["y"] - dict_customer[customers_id[i-1]]["y"]) ** 2)
                                                    distance_to_closest_pup = min(distance_to_closest_pup, distance_temp)

                                                customer_distance_to_pup = math.sqrt(( dict_pickup[pup_id]["x"]- dict_customer[customers_id[i-1]]["x"]) ** 2 +
                                                              ( dict_pickup[pup_id]["y"] - dict_customer[customers_id[i-1]]["y"]) ** 2)
                                                #
                                                # if distance_to_closest_pup*4/TSP_cost_per_customer>l_max:
                                                #     p_home = 1-p_max
                                                # elif distance_to_closest_pup*4/TSP_cost_per_customer < l_min:
                                                #     p_home = 1-p_min
                                                # else:
                                                #     p_home = 1-p_av
                                                #print(i, distance_to_closest_pup*4/TSP_cost_per_customer , p_home)
                                                file.write("{} {} {} {} {} {}\n".format(i,  round(dict_customer[customers_id[i-1]]["x"],2),
                                                                                        round(dict_customer[customers_id[i-1]]["y"],2), p_home, p_pup,
                                                                                         round(distance_to_closest_pup * u * TSP_cost_per_customer/10,2) )) #  round(total_discount/ nr_cust,2)

def generate_3_segments_instance_zhou_saturation(instance_type ):
    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    dict_depot, dict_pickup, dict_customer = adapt_zhou(instance_type)
    instance_type = "VRPDO"
    mainDirStorage = os.path.join(path_to_data, "data", "i_VRPDO_saturation_manyPup")
    #nr_custs = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    nr_custs = [15]
    dict_disc_prob = saturation()
    instanceList = os.path.join(mainDirStorage, 'list.txt')

    for id_instance in [1]:
        shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers_' + str(id_instance) + '.txt')
        for nr_cust in nr_custs:
            for u in dict_disc_prob:
                p_pup =  0
                p_home =  round(dict_disc_prob[u][1],3)
                nr_pup = 3
                with open(shuffled_cust_list, "rb") as file_shuffled:
                    depots_id = pickle.load(file_shuffled)
                    customers_id = pickle.load(file_shuffled)
                    pup_ids = set_pickup_point_preselected(nr_pup, nr_cust, id_instance)

                    instanceName = instance_type+'_size_'+str(nr_cust) + '_phome_' + str(p_home) + '_ppup_' + str(p_pup) +'_incrate_'  + str(round(u,3)) +'_'+str(id_instance)+ '.txt'
                    instanceDir = os.path.join(mainDirStorage, instanceName)
                    with open(instanceList, 'a+', encoding='utf-8') as file:
                        file.write("{}\n".format(instanceName.split('.txt')[0]))

                        TSP_cost_per_customer = temp_instance(dict_customer, nr_cust, nr_pup, customers_id,
                                                              [dict_pickup[pup_id] for pup_id in pup_ids],
                                                              dict_depot[depots_id[0]])

                        with open(instanceDir, 'w+', encoding='utf-8') as file:
                            file.write("NAME: {}\n".format(instanceName))
                            file.write("SIZE: {}\n".format(nr_cust))
                            file.write("NUMBER_PUPs: {}\n\n".format(nr_pup))

                            file.write("LOCATION COORDINATES:\n\n")

                            file.write("DEPOT V. XCOORD. YCOORD.\n")
                            file.write("{} {} {}\n\n".format(0, dict_depot[depots_id[0]]["x"],dict_depot[depots_id[0]]["y"]))

                            file.write("PUP V. XCOORD. YCOORD.\n")
                            iter = nr_cust
                            for pup_id in pup_ids:
                                iter += 1
                                file.write("{} {} {}\n".format(iter, (dict_pickup[pup_id]["x"]),
                                                               (dict_pickup[pup_id]["y"])))
                            file.write("\n")

                            file.write("CUST V. XCOORD. YCOORD. PROB_ALWAYS_HOME PROB_ALWAYS_PUP SHIPPING_FEE\n")

                            total_discount = 0
                            for i in range(1, nr_cust + 1):
                                distance_to_the_first_pup = int(math.sqrt(
                                    (dict_pickup[pup_ids[0]]["x"] - dict_customer[customers_id[i - 1]]["x"]) ** 2 + \
                                    (dict_pickup[pup_ids[0]]["y"] - dict_customer[customers_id[i - 1]][
                                        "y"]) ** 2) + 0.5)
                                total_discount += round(distance_to_the_first_pup * u * TSP_cost_per_customer, 2)

                            for i in range(1, nr_cust + 1):
                                distance_to_closest_pup = 10 ** 5
                                for pup_id in pup_ids:
                                    distance_temp = math.sqrt(
                                        (dict_pickup[pup_id]["x"] - dict_customer[customers_id[i - 1]]["x"]) ** 2 + \
                                        (dict_pickup[pup_id]["y"] - dict_customer[customers_id[i - 1]]["y"]) ** 2)
                                    distance_to_closest_pup = min(distance_to_closest_pup, distance_temp)

                                customer_distance_to_pup = math.sqrt(
                                    (dict_pickup[pup_id]["x"] - dict_customer[customers_id[i - 1]]["x"]) ** 2 +
                                    (dict_pickup[pup_id]["y"] - dict_customer[customers_id[i - 1]]["y"]) ** 2)

                                file.write("{} {} {} {} {} {}\n".format(i,  (dict_customer[customers_id[i-1]]["x"]),
                                                                            (dict_customer[customers_id[i-1]]["y"]), p_home, p_pup,
                                                                            round(customer_distance_to_pup * u * TSP_cost_per_customer,2)))



def temp_check_position_pup(file_name):
    problem = tsplib95.load(file_name)
    num_cust = 100
    pupnode = 1
    depotnode = 50

    nodes = problem.get_nodes()
    sample_customers = range(1, num_cust+1)

    plt.figure(0)
    ax, fig = plt.gca(), plt.gcf()
    ax.set_aspect(1.0)

    for customer in sample_customers:
        ax.scatter(problem.node_coords[customer][0], problem.node_coords[customer][1], marker='o', s=20, color='red')
        ax.text(problem.node_coords[customer][0]+ 0.35,problem.node_coords[customer][1] + 0.35, customer, fontsize=7)

    ax.scatter(problem.node_coords[pupnode][0], problem.node_coords[pupnode][1], marker='^', s=60, color='blue')
    ax.text(problem.node_coords[pupnode][0]+ 0.35,problem.node_coords[pupnode][1] + 0.35, 'P', fontsize=12)

    ax.scatter(problem.node_coords[depotnode][0], problem.node_coords[depotnode][1], marker='s', s=60, color="blue")
    ax.text(problem.node_coords[depotnode][0]+ 0.35,problem.node_coords[depotnode][1] + 0.35, 'D', fontsize=12)
    plt.show()
    print()


def set_disc_coef(problem, sample_customers, nr_cust, pupnode ,depotnode, disc_size):
    customer_disc_coef = []
    sum_distances_to_depot = 0
    for i in sample_customers:
        customer_distance_to_pup = int(math.sqrt((problem.node_coords[pupnode][0]-problem.node_coords[i][0]) ** 2 +
                                                    (problem.node_coords[pupnode][1]- problem.node_coords[i][1]) ** 2))
        customer_disc_coef.append(disc_size / customer_distance_to_pup)
        #print(i, disc_size / customer_distance_to_pup)
    return sum(customer_disc_coef)/len(customer_disc_coef)

#temp fucntion - to write an empty instance that contains only coordinates: thus to calculate TSP cost
def temp_instance(dict_customer, nr_cust,nr_pups,customers_id, pupnodes ,depotnode):
    instance_name = 'temp.txt'
    with open(instance_name, 'w+', encoding='utf-8') as file:
        file.write("NAME: {}\n")
        file.write("SIZE: {}\n".format(nr_cust))
        file.write("NUMBER_PUPs: {}\n\n".format(nr_pups))

        file.write("LOCATION COORDINATES:\n\n")

        file.write("DEPOT V. XCOORD. YCOORD.\n")
        file.write("{} {} {}\n\n".format(0, depotnode["x"],depotnode["y"]))

        file.write("PUP V. XCOORD. YCOORD.\n")
        for pupnode in pupnodes:
            file.write("{} {} {}\n".format(nr_cust + 1, pupnode["x"], pupnode["y"]))
        file.write("\n")

        file.write("CUST V. XCOORD. YCOORD. PROB_ALWAYS_HOME PROB_ALWAYS_PUP SHIPPING_FEE\n")

        for i in range(1, nr_cust + 1):
            file.write("{} {} {} {} {} {}\n".format(i, (dict_customer[customers_id[i - 1]]["x"]),
                                                    (dict_customer[customers_id[i - 1]]["y"]), 0.0, 0.0,0.0))
    OCVRPInstance = OCVRPParser.parse(instance_name)
    tspSolver = TSPSolver(instance=OCVRPInstance, solverType='Gurobi')
    nominalTSPCost = tspSolver.tspCost(0) / nr_cust
    print("nominalTSPCost(per customer: ",nominalTSPCost)
    os.remove("temp.txt")
    return nominalTSPCost



def generate_examples_concorde(file_example):
    for size in range(9, 60):
        problem = tsplib95.load(file_example)
        problem.dimension = size
        problem.name = "berlin"+str(size)
        node_coords = {key:problem.node_coords[key] for key in range(1, size + 1)}
        problem.node_coords = node_coords
        problem.save(os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "data_for_pyconcorde","berlin"+str(size)+".tsp"))


def set_pickup_point(dict_pickup, dict_customer, customers_id, nr_cust, nr_pups):
    pup_id_best = 1
    distance_best = 10000000
    for id in range(1, len(dict_pickup) + 1):
        distance = 0
        for i in range(1, nr_cust + 1):
            distance += math.sqrt((dict_pickup[id]["x"] - dict_customer[customers_id[i - 1]]["x"]) ** 2 +
                                      (dict_pickup[id]["y"] - dict_customer[customers_id[i - 1]]["y"]) ** 2)
        if distance < distance_best:
            pup_id_best = id
            distance_best = distance


    #pups that are the closest for at least one customer
    closest_pups= []
    for cust_id in customers_id:
        closest_pup_id = 1
        distance_closest = 1000000
        for pup in dict_pickup:
            distance_pup = math.sqrt((dict_pickup[pup]["x"] - dict_customer[cust_id]["x"]) ** 2 +
                          (dict_pickup[pup]["y"] - dict_customer[cust_id]["y"]) ** 2)
            if distance_pup < distance_closest:
                distance_closest = distance_pup
                closest_pup_id = pup
        if closest_pup_id not in closest_pups:
            if closest_pup_id != pup_id_best:
                closest_pups.append(closest_pup_id)
    pup_ids = [pup_id_best] + sample(closest_pups, nr_pups-1)
    return pup_ids

def set_pickup_point_preselected(nr_pups, nr_cust, shuffle_id):
    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    pup_list = os.path.join(mainDirZhou, 'PUP_id.txt')
    if nr_cust <= 20:
        nr_cust_example = 15
    else:
        nr_cust_example = 30
    with open(pup_list, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if str(shuffle_id) in  line.split(' ')[0]:
                if str(nr_cust_example) in line.split(' ')[1]:
                    pup_ids = [int(x) for x in line.split('[')[1].split(']')[0].split(',')[:nr_pups]]
    return pup_ids



def adapt_zhou(instance_type):

    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    file_instance = os.path.join(mainDirZhou, instance_type + '.txt')

    with open(file_instance, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        ndep = int(lines[1].split()[1])
        nsat = int(lines[1].split()[2])
        npic = int(lines[1].split()[3])
        ncus = int(lines[1].split()[4])
        dict_depot, dict_pickup, dict_customer = {},{},{}
        dict_main_depots = {}
        for line in lines[4 :4 + ndep]:
            info = line.split()
            id = int(info[0])
            dict_main_depots[id] = {}
            dict_main_depots[id]['x'] = float(info[5])
            dict_main_depots[id]['y'] = float(info[6])

        #depots
        for line in lines[6+ndep:6+ndep + nsat]:
            info = line.split()
            id = int(info[0])
            dict_depot[id] = {}
            dict_depot[id]['x'] = float(info[4])
            dict_depot[id]['y'] = float(info[5])
        #pups
        for line in lines[8+ndep+nsat:8+ndep + nsat+npic]:
            info = line.split()
            id = int(info[0])
            dict_pickup[id] = {}
            dict_pickup[id]['x'] = float(info[1])
            dict_pickup[id]['y'] = float(info[2])
        #custs
        for line in lines[10+ndep+nsat+npic:10+ndep + nsat+npic+ncus]:
            info = line.split()
            id = int(info[0])
            dict_customer[id] = {}
            dict_customer[id]['x'] = float(info[2])
            dict_customer[id]['y'] = float(info[3])

    return dict_depot, dict_pickup, dict_customer

import matplotlib.pyplot as plt

def print_zhou():
    instance_type =  'I1-12-30-200'
    dict_depot, dict_pickup, dict_customer,dict_main_depots = adapt_zhou(instance_type)

    plt.figure(0)
    ax, fig = plt.gca(), plt.gcf()
    ax.set_aspect(1.0)

    for i in dict_depot:
        ax.scatter(dict_depot[i]["x"], dict_depot[i]["y"], marker='s', s=40, color='black')

    for i in dict_pickup:
        ax.scatter(dict_pickup[i]["x"], dict_pickup[i]["y"], marker='^', s=40, color='blue')

    for i in dict_main_depots:
        ax.scatter(dict_main_depots[i]["x"], dict_main_depots[i]["y"], marker='s', s=80, color='black')

    for i in dict_customer:
        ax.scatter(dict_customer[i]["x"], dict_customer[i]["y"], marker='o', s=40, color='red')
    #ax.text(customer.xCoord + 0.35, customer.yCoord + 0.35, customer, fontsize=12)
    plt.show()

def saturation():
    def rate_from_disc(discount):
        return discount*0.06/average_discount
    def prob_pup(beta, discount):
        prob = 1 - math.exp(-beta*discount)
        return prob

    def set_beta(discount_rate, p_delta):
        return -math.log(1 - p_delta )/ discount_rate

    file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP",
                                 "VRPDO_size_15_phome_0.4_ppup_0.0_incrate_0.06_1.txt")
    instance = OCVRPParser.parse(file_instance)
    tspSolver = TSPSolver(instance=instance, solverType='Gurobi')
    nominalTSPCost = tspSolver.tspCost(0) / 15
    print("nominalTSPCost(per customer: ", nominalTSPCost)
    discount_value_av = []
    for cust in instance.customers:
        discount_value_av.append(instance.shipping_fee[cust.id])
    average_discount = sum(discount_value_av) / len(discount_value_av)
    print('average_discount value:', average_discount)

    p_pup = 0
    u_disc_0 = rate_from_disc(0.5)
    p_delta_0 = 0.5
    beta = set_beta(u_disc_0, p_delta_0)
    print("beta: ", beta, beta*0.05/average_discount)


    discounts_print = [0.33, 0.66, 1,1.33, 1.66,  2,  2.33, 2.66, 3, 3.33, 3.66, 4, 4.33, 4.66, 5, 5.33, 5.66, 6]
    dict_probabilities = {}
    for disc in discounts_print:
        dict_probabilities[rate_from_disc(disc)] = [p_pup, 1 - p_pup - prob_pup(beta, rate_from_disc(disc)),disc]
        print(disc,dict_probabilities[rate_from_disc(disc)][1] )
    return dict_probabilities

def generate_artificial(template_name, instance_base):
    instance_template = os.path.join(path_to_data, "data", "solomon", template_name+".csv")
    nr_custs = [15]
    # elif instance_template == 'solomonC100':
    #     # pupnode, depotnode = 2, 100  #the best alternative of nodes
    #     pupnode, depotnode = 1, 52
    #     # disc_sizes = [1, 4, 12]
    #     # disc_sizes = [2,4,6]
    #     disc_sizes = [2, 3, 4]
    #     u_s = [0.2, 0.5, 0.1, 0.12, 0.18, 0.24, 0.4, 0.6, 0.8, 1, 2]  # discout parameters relative to the routing cost
    #     # disc_sizes = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

if __name__ == "__main__":
    #print_zhou()
    #generate_examples_concorde(os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data","data_for_pyconcorde", "berlin70.tsp"))
    mainDirTSPLIB = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "TSPLIB_all_instances")
    instance_example =  os.path.join(path_to_data, "data", "i_VRPDO_saturation_manyPup",
                                     "VRPDO_size_15_phome_0.4_ppup_0.0_incrate_0.06_nrpup5_0.txt")
    #generate_artificial('C101', instance_example)
    #instance_types =['berlin52', 'solomonR100', 'solomonRC100', 'solomonC100']
    #instance_types = ['berlin52', 'solomonR100', 'solomonRC100', 'solomonC100']
    #instance_types = ['berlin52_clustered3']

    #instance_types = ['eil101']
    #generate_3_segments_instance_zhou_constant_density(instance_type='I1-12-30-200')
    #generate_3_segments_instance_zhou_saturation(instance_type='I1-12-30-200')
    generate_3_segments_instance_zhou_discount_proportional_tsp(instance_type='I1-12-30-200')

    #for instance_type in instance_types:
    #    file_instance_basic = os.path.join(mainDirTSPLIB, instance_type + '.tsp')
        # 1. check positions of PUP and Depot
        #temp_check_position_pup(file_instance_basic)
        #generate_3_segments_instance(file_instance_basic, instance_type)
    #files_tsp = [f for f in os.listdir(mainDirTSPLIB) if f.endswith(".tsp")]

   #read_file(file_instance_basic)

    #for file_name in ['eil76.tsp', 'eil51.tsp','eil101.tsp','berlin52.tsp']:
    #for file_name in ['berlin52.tsp']:
    #    readTSPLIB(file_name)
    #read_file('RC1_19')
    #read_file('R1_11')
    #read_file('C1_12')

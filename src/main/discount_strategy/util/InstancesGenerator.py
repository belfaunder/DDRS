import os
import pickle
import math
from src.main.discount_strategy.io import OCVRPParser
from src.main.discount_strategy.algorithms.exact.TSPSolver import TSPSolver
from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA

def generate_3_segments_instance_zhou_discount_proportional_tsp(instance_type ):
    mainDirZhou = os.path.join(path_to_data, "data", "zhou-et-al-2017")
    dict_depot, dict_pickup, dict_customer = adapt_zhou(instance_type)
    for dict in [dict_depot, dict_pickup, dict_customer]:
        for i in dict:
            dict[i]['x'] *= constants.SCALING_FACTOR
            dict[i]['y'] *= constants.SCALING_FACTOR
    instance_type = "DDRS"
    mainDirStorage = os.path.join(path_to_data, "data", "i_DDRS")
    nr_custs = [18]
    dict_parameters = {0.06:[0.3, 0.6, 0.9], 0.03:[0.6], 0.12:[0.6]}
    instanceList = os.path.join(mainDirStorage, 'list.txt')

    for id_instance in range(10):
        print("id", id_instance)
        shuffled_cust_list = os.path.join(mainDirZhou, 'shuffled_customers_' + str(id_instance) + '.txt')
        for nr_cust in nr_custs:
            for u in dict_parameters:
                for p_delta in dict_parameters[u]:
                    for nr_pup in [1,5]:
                        with open(shuffled_cust_list, "rb") as file_shuffled:
                            depots_id = pickle.load(file_shuffled)
                            customers_id = pickle.load(file_shuffled)
                            pup_ids = set_pickup_point_preselected( nr_pup,nr_cust, id_instance)
                            instanceName = instance_type+'_nrcust_'+str(nr_cust)+'_nrpup'+str(nr_pup) + '_delta_' + str(p_delta) +'_u_'  + str(u) +'_'+\
                                                       str(id_instance)+ '.txt'

                            instanceDir = os.path.join(mainDirStorage, instanceName)
                            with open(instanceList, 'a+', encoding='utf-8') as file:
                                file.write("{}\n".format(instanceName.split('.txt')[0]))
                            TSP_cost_per_customer = temp_instance(dict_customer, nr_cust, nr_pup, customers_id, [dict_pickup[pup_id] for pup_id in pup_ids], dict_depot[depots_id[0]])
                            with open(instanceDir, 'w+', encoding='utf-8') as file:
                                file.write("NAME: {}\n".format(instanceName))
                                file.write("NUMBER_CUSTOMERs: {}\n".format(nr_cust))
                                file.write("NUMBER_PUPs: {}\n\n".format(nr_pup))

                                file.write("LOCATION COORDINATES:\n\n")

                                file.write("DEPOT V. XCOORD. YCOORD.\n")
                                file.write("{} {} {}\n\n".format(0, round(dict_depot[depots_id[0]]["x"]),round(dict_depot[depots_id[0]]["y"])))

                                file.write("PUP V. XCOORD. YCOORD.\n")
                                iter = nr_cust
                                for pup_id in pup_ids:
                                    iter +=1
                                    file.write("{} {} {}\n".format(iter, round(dict_pickup[pup_id]["x"]),
                                                                             round(dict_pickup[pup_id]["y"])))
                                file.write("\n")

                                file.write("CUST V. XCOORD. YCOORD. INCENTIVE_EFFECTIVENESS INCENTIVE_VALUE\n")
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
                                    file.write("{} {} {} {} {}\n".format(i,  round(dict_customer[customers_id[i-1]]["x"]),
                                                                                round(dict_customer[customers_id[i-1]]["y"]), round(p_delta,1),
                                                                             round(distance_to_closest_pup * u * TSP_cost_per_customer/1000),2 ))


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

if __name__ == "__main__":
    generate_3_segments_instance_zhou_discount_proportional_tsp(instance_type='I1-12-30-200')

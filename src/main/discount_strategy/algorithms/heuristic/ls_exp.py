'''
#Local search
import numpy as np
import gurobipy as grb
from pathlib import Path
import sys
import os
import random
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)
from init_VRP import initialise_VRP
import constants
from small_functions import inpup_quantity
from small_functions import tolist
from small_functions import format_strategy

path_to_heuristic= os.path.join((Path(os.path.abspath(__file__)).parents[1]),"src","main","discount_strategy","algorithms","heuristic")
sys.path.insert(2, path_to_heuristic)
from evaluate_bounds import two_strategy_comparison

def insert_operator(old_strategy):
    new_strategy = dict(old_strategy)
    notvisited_customers = []
    success = False
    for i in old_strategy:
        if old_strategy[i] > constants.EPS:
            notvisited_customers.append(i)
    try:
        insert_customer = random.choice(notvisited_customers)
        new_strategy[insert_customer] = 0
        success = True
    except:
        pass
    return new_strategy, success

def delete_operator(old_strategy):
    new_strategy = dict(old_strategy)
    visited_customers = []
    success = False
    for i in old_strategy:
        if old_strategy[i] == 0:
            visited_customers.append(i)
    try:
        delete_customer = random.choice(visited_customers)
        new_strategy[delete_customer] = 1
        success = True
    except:
        pass
        #print("Cannot delete customer")
    return new_strategy, success

def local_search_exp(t_no_discount, input_data, sets, cost_K, routes, dict_scenarios):
    c = input_data["distances"]
    n = input_data["n"]
    discount_e = input_data["discount_e"]
    old_t = dict(t_no_discount)
    ls_ub_list = []
    ls_lb_list = []
    best_known_dict = dict(old_t)
    best_known_ub = cost_K[-1]
    best_known_lb = c[0, n + 1] * 2
    iter_constant_t = 0

    iterations = 1
    while iterations < constants.ITERATION_LS:
        iterations += 1
        #print(iterations,best_known_ub,best_known_lb)
        operator = np.random.choice(["insert", "delete"], p=[0.5, 0.5])
        if operator == "insert":
            new_t, success = insert_operator(old_t)
            if not success:
                new_t, success = delete_operator(old_t)
        else:
            new_t, success = delete_operator(old_t)
            if not success:
                new_t, success = insert_operator(old_t)

        lb1, ub1, lb2, ub2, dict_scenarios = two_strategy_comparison(old_t, new_t, input_data, sets, cost_K, routes,
                                                                     dict_scenarios)

        discount_difference = discount_e * (inpup_quantity(old_t) - inpup_quantity(new_t))
        #print(discount_difference, lb1, ub1, lb2, ub2, format_strategy(old_t), format_strategy(new_t))
        if (ub2 - lb1 - discount_difference < constants.EPS):
            best_known_dict = dict(new_t)
            best_known_ub = ub2
            best_known_lb = lb2
            if np.random.choice([0, 1], p=[0.1, 0.9]) == 1:
                old_t = dict(new_t)
                iter_constant_t = 0
            else:
                iter_constant_t += 1
        elif (ub1 - lb2 + discount_difference < constants.EPS):
            if np.random.choice([0, 1], p=[0.9, 0.1]) == 1:
                old_t = dict(new_t)
                iter_constant_t = 0
            iter_constant_t += 1
        else:
            if np.random.choice([0, 1], p=[0.5, 0.5]) == 1:
                old_t = dict(new_t)
                iter_constant_t = 0
            else:
                iter_constant_t += 1
        if iter_constant_t > 50:
            print("stack in", old_t)
            break
    return best_known_dict, best_known_lb, best_known_ub, dict_scenarios
'''
'''
#random strategy
from pathlib import Path
import numpy as np
import sys
import os
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)
import constants

def random_strategy(old_strategy, x, sets, input_data):
    new_strategy ={}

    istance = input_data['distances']
    set_N = sets['N']
    set_P = range(input_data['n'] + 1, input_data['n'] + 1 + constants.FLEET_SIZE)
    set_0N1 = set(set_N).union([0], set_P)
    prob = 0.5
    for i in set_N:
        new_strategy[i] = np.random.choice([0,input_data['lowest_discount'][1]], p=[ 1-prob, prob])
    return new_strategy
'''
'''
#Remove customer
from pathlib import Path
import sys
import os
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)
import constants

def insertion_small_enough(old_strategy, x, sets, input_data):
    flag = 0
    new_strategy = old_strategy.copy()
    number_given_discount = 0
    for i in old_strategy:
        if old_strategy[i] > 0.5:
            number_given_discount += 1
    if number_given_discount == 0:
        print("no customers_removed")
        flag = 1
    else:
        distance = input_data['distances']
        set_N = sets['N']
        set_P = range(input_data['n'] + 1, input_data['n'] + 1 + constants.FLEET_SIZE)
        set_0N1 = set(set_N).union([0], set_P)

        treshold_insertion_cost = input_data['lowest_discount'][1]*(
            input_data['segment_probability'][0]/input_data['segment_probability'][1] + 1)
        insertion_cost = {}
        for i in set_N:
            if old_strategy[i] > 0.5:
                insertion_temp = []
                start=0
                finish = 0
                for arc in x:
                    if x[arc] > 0.5:
                        insertion_temp.append(distance[arc[0],i]+distance[i,arc[1]] - distance[arc[0],arc[1]])

                insertion_cost[i] = min(insertion_temp)
            else:
                insertion_cost[i] = treshold_insertion_cost+constants.EPS
        print(insertion_cost)
        print(treshold_insertion_cost)
        candidate = min(insertion_cost, key=insertion_cost.get)
        if insertion_cost[candidate] < treshold_insertion_cost:
            new_strategy[candidate] = 0
        else:
            flag = 1
    return new_strategy,flag
'''

print(bin(30720))
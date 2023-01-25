# Set scenarios - combinatorial enumeration of all possible variations of 1 and 0 with length N
# The customers numbers starts from 1 therefore the scenarios [0] ==0 in all scenario(its done in order not to change the notation in VRP algorithm
# scenario as an array:
# T = set_scenarios(n, lowest_discount[0], lowest_discount[1])
from itertools import combinations

def set_scenarios(size, first_param, second_param):
    K = {}
    k = 0
    while k < 2 ** size:
        K[k] = {}
        k += 1
    set_customer = range(1, (size + 1))
    for i in set_customer:
        K[0][i] = first_param

    for i in set_customer:
        temp = 1
        k = 1
        while temp <= 2 ** (i - 1):
            while k < 2 ** (size - i) * (2 * temp - 1):
                K[k][i] = first_param
                k += 1
            while k < 2 ** (size - i) * (2 * temp):
                K[k][i] = second_param
                k += 1
            temp += 1
    return K

#limit - limit on number of 1 in scenario
def set_scenarios_limit(n, limit):
    K = {}
    iter = 0
    for i in range(1, limit+1):
        all_combinations = combinations(range(1,n+1), limit-i)
        for combination in all_combinations:
            iter += 1
            K[iter] = []
            for cust in range(n + 1):
                if cust in combination:
                    K[iter].append(1)
                else:
                    K[iter].append(0)
    return K

#set scenario, return disctionary with scenario as a list:
def set_scenarios_list(size, first_param, second_param):
    K = {}
    k = 0
    while k < 2 ** size:
        K[k] = {}
        k += 1
    set_customer = range(1, (size + 1))
    for i in set_customer:
        K[0][i] = first_param

    for i in set_customer:
        temp = 1
        k = 1
        while temp <= 2 ** (i - 1):
            while k < 2 ** (size - i) * (2 * temp - 1):
                K[k][i] = first_param
                k += 1
            while k < 2 ** (size - i) * (2 * temp):
                K[k][i] = second_param
                k += 1
            temp += 1
    dict_of_lists = {}
    for k in K:
        dict_of_lists[k] = []
    return dict_of_lists

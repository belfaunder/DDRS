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
    solverType = 'Gurobi'
    farness = np.zeros((instance.NR_CUST,2))


    for cust in instance.customers:
        farness[cust.id-1, 0] = int(cust.id)
        farness[cust.id-1, 1] = (sum(instance.distanceMatrix[cust.id, j.id] for j in instance.customers if
                                          j is not cust) + \
                                      sum(instance.distanceMatrix[cust.id, j.id] for j in instance.pups) + \
                                      instance.distanceMatrix[cust.id, instance.depot.id]) / (
                                                 instance.NR_CUST + instance.NR_PUP)

    farness = farness[farness[:,1].argsort()]
    policy, rsValue = ring_star_deterministic_no_TW(instance, instance.NR_CUST)

    if 2 ** bitCount(policy) < constants.SAMPLE_SIZE:
        best_value = one_policy_cost_estimation(instance=instance, policy=policy, solverType=solverType)
    else:
        best_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
                                                                            policy=policy, solverType=solverType)
    # Flag is true if we want to check more policies
    flag = True
    while flag:
        cutomer_offer_incentive = farness[0][0]
        farness = np.delete(farness, 0, 0)
        if policy & (1 << int(cutomer_offer_incentive-1)):
            new_policy = policy & ~(1 << int(cutomer_offer_incentive-1))
            if 2 ** bitCount(policy) < constants.REMOTE_HEURISTIC_SAMPLE:
                test_value = one_policy_cost_estimation(instance=instance, policy=new_policy, solverType=solverType)
            else:
                test_value = sampleAverageApproximation_PoissonBinomial_1sample_2segm(instance=instance,
                                                    policy=new_policy, solverType=solverType,
                                                    sample_size =  constants.REMOTE_HEURISTIC_SAMPLE)
            #print("current", best_value,"policy", bin(policy))
            #print("test", test_value,"policy", bin(new_policy))
            if test_value <= best_value:
                best_value = test_value
                policy = new_policy
            else:
               flag = False
            if farness.size==0:
                flag = False
    return policy
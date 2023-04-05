import os
import sys
from pathlib import Path
import copy
import math
from itertools import combinations
from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import updateBoundsFromDictionary
from src.main.discount_strategy.algorithms.exact.bab.BAB_super_class import BAB_super_class
from src.main.discount_strategy.algorithms.exact.bab.BoundsCalculation import updateBoundsWithNewTSPs
import numpy as np
from time import process_time
from src.main.discount_strategy.util import constants
from src.main.discount_strategy.util import probability
from src.main.discount_strategy.util.bit_operations import bitCount

from src.main.discount_strategy.io import OCVRPParser
path_to_data = constants.PATH_TO_DATA
prefix=constants.PREFIX

from pyscipopt import Model, quicksum


def addcut(edges, set_visit, model, x_vars, n):
    new_cut_added = False
    for omega in range(2 ** n):
        unvisited = set_visit[omega].copy()
        cycle = range(len(set_visit[omega]) + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors =[]
                for edge in edges:
                    if edge[0]==omega:
                        if edge[1] == current:
                            if edge[2] in unvisited:
                                neighbors.append(edge[2])
                        if edge[2] == current:
                            if edge[1] in unvisited:
                                neighbors.append(edge[1])
            if len(cycle) > len(thiscycle):
                cycle = thiscycle

        if len(cycle) < len(set_visit[omega]):
            model.freeTransform()
            model.addCons(quicksum(x_vars[omega, i, j] for i in cycle for j in  cycle if j > i) <= len(cycle) - 1)
            new_cut_added = True
        #print("cut: len(%s) <= %s" % (cycle, len(cycle) - 1))
    return new_cut_added


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


def solve_gurobi_DDRS(instance):
    m = Model("ddrs")
    m.hideOutput()
    m.setParam('limits/time', constants.TIME_LIMIT)
    x_vars = {}
    for omega in range(2 ** instance.NR_CUST):
        for i in range(0, instance.NR_CUST + instance.NR_PUP + 2):
            for j in range(0, instance.NR_CUST + instance.NR_PUP + 2):#instance.distanceMatrix[(i,j)]
                if  j>i:

                    x_vars[omega, i, j] = m.addVar(vtype="B", name='x_{0}_{1}_{2}'.format(omega, i,j))

    for i in range(1, instance.NR_CUST + instance.NR_PUP + 1):
        instance.distanceMatrix[i, instance.NR_CUST + instance.NR_PUP + 1] = instance.distanceMatrix[0,i]
    instance.distanceMatrix[0, instance.NR_CUST + instance.NR_PUP + 1] = 0
    y_vars = {}
    for i in range(1, instance.NR_CUST + 1):
        y_vars[i] = m.addVar(vtype="B", name='y_{0}'.format(i))
    objective_var = m.addVar(lb = 0, ub = 1000,vtype="C", name='objective_var')


    # Add degree-2 constraint
    for omega in range(2 ** instance.NR_CUST):
        set_visit = [0, instance.NR_CUST + instance.NR_PUP + 1]
        for customer in instance.customers:
            if omega & (1 << (customer.id - 1)):
                if customer.closest_pup_id not in set_visit:
                    set_visit.append(customer.closest_pup_id)
            else:
                set_visit.append(customer.id)
        #print(set_visit)

        for i in range(instance.NR_CUST + instance.NR_PUP + 2):
            if i in set_visit:
                m.addCons( quicksum(x_vars[omega, j, i] for j in set_visit if j<i) +  quicksum(
                    x_vars[omega, i, j] for j in set_visit if j>i) == 2,
                        name="constraint_inflow2_{0}".format(i))
            else:
                m.addCons(quicksum(x_vars[omega, j, i] for j in range(instance.NR_CUST + instance.NR_PUP + 1) if j < i) + quicksum(
                    x_vars[omega, i, j] for j in range(instance.NR_CUST + instance.NR_PUP + 1) if j > i) == 0,
                            name="constraint_inflow0_{0}".format(i))

    objective = 0

    for i in range(1, instance.NR_CUST + 1):
        objective += y_vars[i]*instance.p_delta[i]*instance.shipping_fee[i]

    for omega in range(2 ** instance.NR_CUST):
        scenarioProb = 1
        for i in range(1, instance.NR_CUST + 1):
            if omega & (1 << (i - 1)):
                scenarioProb *= instance.p_delta[i]*y_vars[i]
            else:
                scenarioProb *= 1 - instance.p_delta[i] * y_vars[i]

        objective += scenarioProb * quicksum( x_vars[omega, i, j] * instance.distanceMatrix[i,j]
                                         for i in range(0, instance.NR_CUST + instance.NR_PUP + 2)
                                         for j in range(0, instance.NR_CUST + instance.NR_PUP + 2) if j>i)

    #subtour elimination contraint (no callback)
    # for omega in range(2 ** instance.NR_CUST):
    #     set_visit = [0, instance.NR_CUST + instance.NR_PUP + 1]
    #     for customer in instance.customers:
    #         if omega & (1 << (customer.id - 1)):
    #             if customer.closest_pup_id not in set_visit:
    #                 set_visit.append(customer.closest_pup_id)
    #         else:
    #             set_visit.append(customer.id)
    #
    #     #print("set_visit", set_visit, bin(omega))
    #     for cycle in powerset(set_visit):
    #         if len(cycle) >1 and  len(cycle) < len(set_visit):
    #             m.addCons(quicksum(x_vars[omega, i, j] for i in cycle for j in cycle if j > i) <= len(cycle) - 1)

    m.addCons(objective_var>= objective,name="constraint_inflow0}")
    m.setObjective( objective_var, "minimize")
    #m.writeProblem("test.lp")
    m.optimize()

    EPS = 1.e-6
    isMIP = False
    set_visit = {}
    for omega in range(2 ** instance.NR_CUST):
        set_visit[omega] = [0, instance.NR_CUST + instance.NR_PUP + 1]
        for customer in instance.customers:
            if omega & (1 << (customer.id - 1)):
                if customer.closest_pup_id not in set_visit[omega]:
                    set_visit[omega].append(customer.closest_pup_id)
            else:
                set_visit[omega].append(customer.id)
    #for omega in range(2 ** instance.NR_CUST):
    if True:
        while True:
            m.optimize()

            edges = []
            for omega in range(2 ** instance.NR_CUST):
                for i in range(0, instance.NR_CUST + instance.NR_PUP + 2):
                    for j in range(0, instance.NR_CUST + instance.NR_PUP + 2):
                        if j > i:
                            if m.getVal(x_vars[omega, i, j]) > EPS:
                                edges.append([omega, i, j])


            if addcut(edges, set_visit, m, x_vars, instance.NR_CUST)== False:
                if isMIP:  # integer variables, components connected: solution found
                    break
                m.freeTransform()
                for (omega, i, j) in x_vars:  # all components connected, switch to integer model
                    m.chgVarType(x_vars[omega, i, j], "B")
                    isMIP = True

    policy = 0
    for i in range(1, instance.NR_CUST + 1):
        if m.getVal(y_vars[i]) > 0.001:
            policy+= 2**(i-1)
    print(prefix +"Time limit SCIP:", constants.TIME_LIMIT)
    print(prefix +"Objective SCIP:", m.getObjVal())
    print(prefix +"Policy SCIP:", policy)
    return m.getObjVal(), policy

if __name__ == '__main__':

    if os.name != 'nt':
       file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                    "i_VRPDO_2segm_manyPUP_scip", str(sys.argv[-1]) + ".txt")
    else:
       file_instance = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_scip",
                                    "VRPDO_size_6_phome_0.4_ppup_0.0_incrate_0.06_nrpup1_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    print(OCVRPInstance)
    start_time = process_time()
    solve_gurobi_DDRS(OCVRPInstance)
    print(prefix + 'Time_running_scip,s: ', process_time() - start_time)


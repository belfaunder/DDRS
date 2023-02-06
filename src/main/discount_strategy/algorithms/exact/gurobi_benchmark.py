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

import gurobipy as grb
import numpy as np

def solve_gurobi_DDRS(instance):
    m = grb.Model()
    #m.setParam('OutputFlag', False)
    # m.Params.Threads = 1
    x_vars = {}
    for omega in range(2 ** instance.NR_CUST):
        for i in range(0, instance.NR_CUST + instance.NR_PUP + 1):
            for j in range(0, instance.NR_CUST + instance.NR_PUP + 1):#instance.distanceMatrix[(i,j)]
                x_vars[omega, i, j] = m.addVars( obj=0, vtype=grb.GRB.BINARY, name='e_omega_{}'.format(omega))

    y_vars = {}
    for i in range(1, instance.NR_CUST + 1):
        y_vars[i] = m.addVars(obj=0, vtype=grb.GRB.BINARY, name='y_{}'.format(i))

    for omega in range(2 ** instance.NR_CUST):
        for i in range(0, instance.NR_CUST + instance.NR_PUP + 1):
            # forbid loops
            m.addConstr(x_vars[omega, i, i] == 0, name="forbid_loop_{0}".format(i))
            #edge in the opposite direction
            for j in range(i+1, instance.NR_CUST + instance.NR_PUP + 1)
                x_vars[omega, j, i] = x_vars[omega, i, j]

    # Add degree-2 constraint
    constr_inflow2 = grb.tupledict()
    for i in set_visit:
        constr_inflow2[i] = m.addConstr(grb.quicksum(x_vars[i, j] for j in set_visit) == 2,
                                        name="constraint_inflow2_{0}".format(i))
    pass

if __name__ == '__main__':
    if os.name != 'nt':
        file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data",
                                     "i_VRPDO_discount_proportional_2segm_manyPUP", str(sys.argv[-1]) + ".txt")
    else:
        file_instance = os.path.join(path_to_data, "data", "i_VRPDO_discount_proportional_2segm_manyPUP",
                                     "VRPDO_size_10_phome_0.4_ppup_0.0_incrate_0.06_0.txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    print(OCVRPInstance)
    solve_gurobi_DDRS(OCVRPInstance)

from pathlib import Path
import sys
import os
import gurobipy as grb
import numpy as np
from itertools import combinations
from copy import copy, deepcopy

path_to_concorde = os.path.join((Path(os.path.abspath(__file__)).parents[6]), "pyconcorde")
sys.path.insert(1, path_to_concorde)

#if os.name != 'nt':
#    from concorde.tsp import TSPSolver_ as TSPSolverConcorde
#    from concorde.tests.data_utils import get_dataset_path


from src.main.discount_strategy.util.bit_operations import bitCount

def subtourelim(model, where):
    if where == grb.GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._x_vars)
        selected = grb.tuplelist((i, j) for i, j in model._x_vars.keys()
                                 if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, model._set_visit)
        if len(tour) < len(model._set_visit):
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                         # model.addConstr(grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                         <= len(tour) - 1)

# Given a tuplelist of edges, find the shortest subtour
def subtour(edges, set_visit):
    unvisited = set_visit.copy()
    cycle = range(len(set_visit) + 1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

class TSPSolver:
    def __init__(self, instance, solverType):
        self.instance = instance
        self.baseModel = self.tsp_base_model(instance.distanceMatrix, instance.NR_CUST, instance.NR_PUP)
        self.solverType = solverType
        if solverType == 'Concorde':
            fname = get_dataset_path("berlin" + str(instance.NR_CUST + 2))
            self.solver = TSPSolverConcorde.from_tspfile(fname)
    # create a tsp model that visit all customers at home
    def tsp_base_model(self, distanceMatrix, nr_cust, nr_pup):
        m = grb.Model()
        m.setParam('OutputFlag', False)
        m.Params.Threads = 4

        set_visit = list(range(nr_cust + 1 + nr_pup))

        #length of the route that includes only one depot and one pickup point
        m._dist_pup = {}
        for pup_id in range(nr_cust+1, nr_cust+nr_pup+1):
             m._dist_pup[pup_id]= distanceMatrix[(0, pup_id)] * 2

        # Create variables
        x_vars = m.addVars(distanceMatrix.keys(), obj=distanceMatrix, vtype=grb.GRB.BINARY, name='e')
        for i, j in x_vars.keys():
            x_vars[j, i] = x_vars[i, j]  # edge in opposite direction

        # forbid loops
        for i in set_visit:
            m.addConstr(x_vars[i, i] == 0, name="forbid_loop_{0}".format(i))

        # Add degree-2 constraint
        constr_inflow2 = grb.tupledict()
        for i in set_visit:
            constr_inflow2[i] = m.addConstr(grb.quicksum(x_vars[i, j] for j in set_visit) == 2,
                        name="constraint_inflow2_{0}".format(i))

        m._set_visit = set_visit
        m._constr_inflow2 = constr_inflow2
        m._constr_inflow0= grb.tupledict()
        m._x_vars = x_vars
        m.Params.lazyConstraints = 1
        m.optimize(subtourelim)
        return m

    def tspCost(self, scenarioID):
        if self.solverType == 'Gurobi':
            tspCost = self.tspCostGurobi(scenarioID)
        elif self.solverType == 'Concorde':
            tspCost = self.tspCostConcorde(scenarioID)
        return tspCost

    def tspCostGurobi(self, scenarioID):
        n = self.instance.NR_CUST
        # create set of customers that should be visited in scenario
        set_visit = [0]

        for customer in self.instance.customers:
            if scenarioID & (1 << (customer.id - 1)):
                if customer.closest_pup_id not in set_visit:
                    set_visit.append(customer.closest_pup_id)
            else:
                set_visit.append(customer.id)

        if len(set_visit) > 2:
            x_vars = self.baseModel._x_vars
            # Add degree-2 constraint
            for i in range(1, n + self.instance.NR_PUP+1):
                if i in set_visit:
                    try:
                        self.baseModel.remove(self.baseModel._constr_inflow0[i])
                        del self.baseModel._constr_inflow0[i]
                        self.baseModel._constr_inflow2[i] = self.baseModel.addConstr(
                            grb.quicksum(x_vars[i, j] for j in range(n + self.instance.NR_PUP + 1)) == 2,
                            name="constraint_inflow2_{0}".format(i))
                    except:
                        pass
                else:
                    try:
                        self.baseModel.remove(self.baseModel._constr_inflow2[i])
                        del self.baseModel._constr_inflow2[i]
                        self.baseModel._constr_inflow0[i] = self.baseModel.addConstr(
                            grb.quicksum(x_vars[i, j] for j in range(n + self.instance.NR_PUP + 1)) == 0,
                            name="constraint_inflow0_{0}".format(i))
                    except:
                        pass

            self.baseModel._set_visit = set_visit
            #self.baseModel.update()
            # Optimize model

            self.baseModel.optimize(subtourelim)
            # self.baseModel.printAttr('X')

            # m.write("upd.lp")
            #vals = self.baseModel.getAttr('x', x_vars)
            # selected = grb.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
            # print(selected)
            # for tuple in selected:
            #     print(tuple, self.instance.distanceMatrix[tuple])

            #tour = subtour(selected, set_visit)
            #assert len(tour) == n - len(skip)+1

            # print('')
            # print('Optimal tour: %s' % str(tour))
            #
            # print(scenarioID, 'Optimal cost: %g' %  self.baseModel.objVal)
            # print('')
            tsp_cost = self.baseModel.objVal
        else:
            # TODO - this would be more difficult or simply delete it
            tsp_cost = self.baseModel._dist_pup[set_visit[1]]
        #print("")
        #print("NEW_TSP", bin(scenarioID)[2:], tsp_cost)
        return tsp_cost


    #Cost of scenario, where all vertices and all pickup points are visited
    def tspCostUbScenario(self):
        n = self.instance.NR_CUST
        # create set of customers that should be visited in scenario
        set_visit = [0] + [customer.id for customer in self.instance.customers]
        for customer in self.instance.customers:
            if customer.closest_pup_id not in set_visit:
                set_visit.append(customer.closest_pup_id)

        x_vars = self.baseModel._x_vars
        # Add degree-2 constraint
        for i in range(1, n + self.instance.NR_PUP + 1):
            if i in set_visit:
                try:
                    self.baseModel.remove(self.baseModel._constr_inflow0[i])
                    del self.baseModel._constr_inflow0[i]
                    self.baseModel._constr_inflow2[i] = self.baseModel.addConstr(
                        grb.quicksum(x_vars[i, j] for j in range(n + self.instance.NR_PUP + 1)) == 2,
                        name="constraint_inflow2_{0}".format(i))
                except:
                    pass
            else:
                try:
                    self.baseModel.remove(self.baseModel._constr_inflow2[i])
                    del self.baseModel._constr_inflow2[i]
                    self.baseModel._constr_inflow0[i] = self.baseModel.addConstr(
                        grb.quicksum(x_vars[i, j] for j in range(n + self.instance.NR_PUP + 1)) == 0,
                        name="constraint_inflow0_{0}".format(i))
                except:
                    pass

        self.baseModel._set_visit = set_visit
        # self.baseModel.update()
        # Optimize model

        self.baseModel.optimize(subtourelim)
        tsp_cost = self.baseModel.objVal

        lb = 10**5
        for pup in self.instance.pups:
            lb = min(self.instance.distanceMatrix[(0, pup.id)] * 2, lb)

        return tsp_cost, lb

    def tspCostWihoutReuse(self, scenario):
        distanceMatrix = self.instance.distanceMatrix
        nr_cust = self.instance.NR_CUST

        m = grb.Model()
        m.setParam('OutputFlag', False)

        # create set of customers that should be visited in scenario
        set_visit = [0]
        iter = 1
        for i in scenario:
            if i == 0:
                set_visit.append(iter)
        set_visit.append(nr_cust + 1)
        m._dist_pup = distanceMatrix[(0, nr_cust + 1)] * 2
        if len(set_visit) > 2:
            # Create variables
            x_vars = m.addVars(distanceMatrix.keys(), obj=distanceMatrix, vtype=grb.GRB.BINARY, name='e')
            for i, j in x_vars.keys():
                x_vars[j, i] = x_vars[i, j]  # edge in opposite direction

            # forbid loops
            for i in list(range(nr_cust + 2)):
                m.addConstr(x_vars[i, i] == 0, name="forbid_loop_{0}".format(i))

            # Add degree-2 constraint
            for i in list(range(nr_cust + 2)):
                if i in set_visit:
                    m.addConstr(grb.quicksum(x_vars[i, j] for j in range(nr_cust + 2)) == 2,
                                name="constraint_inflow2_{0}".format(i))
                else:
                    m.addConstr(grb.quicksum(x_vars[i, j] for j in range(nr_cust + 2)) == 0,
                                name="constraint_inflow0_{0}".format(i))

            m._set_visit = set_visit
            m._x_vars = x_vars
            m.Params.lazyConstraints = 1
            #m.update()
            # Optimize model

            m.optimize(subtourelim)

            # self.baseModel.printAttr('X')

            # m.write("upd.lp")
            # vals = self.baseModel.getAttr('x', x_vars)
            # selected = grb.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
            # print(selected)

            # tour = subtour(selected, set_visit)
            # assert len(tour) == n - len(skip)+1

            # print('')
            # print('Optimal tour: %s' % str(tour))

            # print(scenario, 'Optimal cost: %g' %  m.objVal)
            # print('')
            tsp_cost = m.objVal
        else:
            tsp_cost = m._dist_pup

        return tsp_cost

    def tspCostConcorde(self, scenarioID):

        num_discounts = bitCount(scenarioID)

        if num_discounts > self.instance.NR_CUST - 3:
            tsp_cost = self.tspCostGurobi(scenarioID)
        else:
            self.solver._data.x[0], self.solver._data.x[1] = self.instance.pup.xCoord, self.instance.depot.xCoord
            self.solver._data.y[0], self.solver._data.y[1] = self.instance.pup.yCoord, self.instance.depot.yCoord
            i = 2
            for customer in self.instance.customers:
                if not scenarioID & (1 << (customer.id - 1)):
                    self.solver._data.x[i] = customer.xCoord
                    self.solver._data.y[i] = customer.yCoord
                    i += 1
            for iter in range(i, self.instance.NR_CUST + 2):
                self.solver._data.x[iter] = self.instance.pup.xCoord
                self.solver._data.y[iter] = self.instance.pup.yCoord

            self.solver._ncount = self.instance.NR_CUST + 2 - num_discounts

            solution = self.solver.solve()
            tsp_cost = solution.optimal_value
        return tsp_cost
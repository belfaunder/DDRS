from pathlib import Path
import sys
import os
import gurobipy as grb
from itertools import combinations
from gurobipy import GRB
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)
import constants


# solve ring start for all customers form disctionar "nodes"
# input: size of discount that customers get(all equal)
# max number of customers that can get a discount
# return min cost of ring-star, opt_model, x_vars
# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        n = model._n
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._x_vars)
       # wvals = model.cbGetSolution(model._w_vars)
        selected = grb.tuplelist((i, j) for i, j in model._x_vars.keys()
                                if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, model._set_visit)
        #print(tour)
        inpuptemp = []
        for i in range(1, n + 1):
            if i not in tour:
                inpuptemp.append(i)
        #print("inpuptemp", inpuptemp)
        if len(tour) < len(model._set_visit):
            #print(len(tour))
            # add subtour elimination constr. for every pair of cities in tour

            if (tour[0]==0 and tour[len(tour)-1]==n+1) :
                model.cbLazy(model._x_vars[n+1,0] + grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                             <= len(tour) - 1 + grb.quicksum(model._w_vars[i] for i in inpuptemp)/(n-len(tour) +2))
            elif (tour[0]==n+1 and tour[len(tour)-1]==0) :
                model.cbLazy(
                    model._x_vars[0, n+1] + grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                    <= len(tour) - 1+ grb.quicksum(model._w_vars[i] for i in inpuptemp)/(n-len(tour) +2))
            else:
                model.cbLazy(grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                            <= len(tour)-1 + grb.quicksum(model._w_vars[i] for i in inpuptemp)/(n-len(tour) +2))
# Given a tuplelist of edges, find the shortest subtour

def subtour(edges, set_visit):
    unvisited = set_visit.copy()
    cycle = range(len(set_visit)+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle) and len(thiscycle) > 1 and thiscycle != [0, 6]:
            cycle = thiscycle
    return cycle

def tsp_base_model(sets, input_data):

    return m

def ring_star_deterministic_no_TW(instance, max_number_offered_discounts, **kwargs):
    distanceMatrix = instance.distanceMatrix
    n = instance.NR_CUST
    if 'discount' in kwargs:
        discount = kwargs['discount']
    else:
        discount = instance.shipping_fee

    # initialize the full graph
    set_N = range(1,n+1)
    set_N1 = range(1,n+2)
    set_0N1 = range(n+2)

    max_pup = min(max_number_offered_discounts, constants.PUP_CAPACITY)
    #print("max_pup", max_pup)
    m = grb.Model()
    m._set_visit = [i for i in range(n+2)]
    m._n = n
    m.setParam('OutputFlag', False)

    # Create variables
    x_vars = m.addVars(distanceMatrix.keys(), obj=distanceMatrix, vtype=GRB.BINARY, name='e')
    for i, j in x_vars.keys():
        if (i==0 and j==n+1) or (i==n+1 and j==0):
            continue
        else:
            x_vars[j, i] = x_vars[i, j]  # edge in opposite direction

    w_vars = grb.tupledict()
    for i in set_N:
        w_vars[i] = m.addVar(obj=discount[i], vtype=GRB.BINARY, name='w[%d]'%(i))
    #m.addConstr(w_vars[0] == 0)
    #m.addConstr(w_vars[n+1] == 0)
    m.addConstr(grb.quicksum(w_vars[i] for i in set_N) <= max_pup)

    # forbid loops
    for i in set_0N1:
        m.addConstr(x_vars[i, i] == 0, name="forbid_loop_{0}".format(i))

    # Add degree-2 constraint
    for i in set_0N1:
        if i==0:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + x_vars[n+1,0]== 2,
                        name="constraint_inflow2_{0}".format(i))
        elif i==n+1:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + x_vars[0,n+1]== 2,
                        name="constraint_inflow2_{0}".format(i))
        else:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + 2*w_vars[i] == 2,
                            name="constraint_inflow2_{0}".format(i))
    # Optimize model
    m._x_vars = x_vars
    m._w_vars = w_vars
    m.Params.lazyConstraints = 1

    m.optimize(subtourelim)
    #m.write("upd.lp")
    vals = m.getAttr('x', w_vars)
    selected_disc = grb.tuplelist((i) for i in vals.keys() if vals[i] > 0.5)
    policy = [0]*(n)
    policyID = 0
    for i in selected_disc:
        policy[n-i] = 1
        policyID = policyID ^(1 << (i-1))

    #print("policy", policy)
    #vals = m.getAttr('x', x_vars)
    #selected = grb.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    #print(selected)
    #print([i for i in range(n+2) if i not in selected_disc])
    #tour = subtour(selected,[i for i in range(n+2) if i not in selected_disc])
    #print('Optimal tour: %s' % str(tour))

    #m.printAttr('X')
    #print(bin(policyID)[2:])
    return policyID, m.objVal




def ring_star_deterministic_no_TW1(discount, max_number_offered_discounts, sets, input_data):

    # initialize the full graph
    n = input_data["n"]
    set_N = sets["N"]
    set_N1 = sets["N1"]
    set_0N1 = sets["0N1"]
    dist = input_data['distances']

    max_pup = min(max_number_offered_discounts, constants.PUP_CAPACITY)
    #print("max_pup", max_pup)
    m = grb.Model()
    m._set_visit = [i for i in range(n+2)]
    m._n = n
    m.setParam('OutputFlag', False)

    # Create variables
    x_vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in x_vars.keys():
        if (i==0 and j==n+1) or (i==n+1 and j==0):
            continue
        else:
            x_vars[j, i] = x_vars[i, j]  # edge in opposite direction

    w_vars = grb.tupledict()
    for i in set_0N1:
        w_vars[i] = m.addVar(obj=discount, vtype=GRB.BINARY, name='w[%d]'%(i))
    m.addConstr(w_vars[0] == 0)
    m.addConstr(w_vars[n+1] == 0)
    m.addConstr(grb.quicksum(w_vars[i] for i in set_N) <= max_pup)

    # forbid loops
    for i in set_0N1:
        m.addConstr(x_vars[i, i] == 0, name="forbid_loop_{0}".format(i))

    # Add degree-2 constraint
    for i in set_0N1:
        if i==0:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + x_vars[n+1,0]== 2,
                        name="constraint_inflow2_{0}".format(i))
        elif i==n+1:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + x_vars[0,n+1]== 2,
                        name="constraint_inflow2_{0}".format(i))
        else:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + 2*w_vars[i] == 2,
                            name="constraint_inflow2_{0}".format(i))
    # Optimize model
    m._x_vars = x_vars
    m._w_vars = w_vars
    m.Params.lazyConstraints = 1

    m.optimize(subtourelim)
    #m.write("upd.lp")
    vals = m.getAttr('x', w_vars)
    selected_disc = grb.tuplelist((i) for i in vals.keys() if vals[i] > 0.5)
    policy = [0]*(n+1)
    for i in selected_disc:
        policy[i] = 1
    #print("policy", policy)
    #vals = m.getAttr('x', x_vars)
    #selected = grb.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    #print(selected)
    #print([i for i in range(n+2) if i not in selected_disc])
    #tour = subtour(selected,[i for i in range(n+2) if i not in selected_disc])
    #print('Optimal tour: %s' % str(tour))

    #m.printAttr('X')

    return policy, m.objVal


def ring_star_deterministic_no_TW_old(discount, max_number_offered_discounts, sets, input_data):
    # initialize VRP only for customers that should be visited
    n = input_data['n']
    set_N = sets['N']
    set_P = range(input_data['n'] + 1, input_data['n'] + 1 + constants.FLEET_SIZE)
    set_0N1 = set(set_N).union([0], set_P)
    set_N1 = set(set_N).union(set_P)

    c = input_data['distances']

    max_pup = min(max_number_offered_discounts, constants.PUP_CAPACITY)

    opt_model = grb.Model(name="MIP Model")
    opt_model.setParam('OutputFlag', False)
    opt_model.setParam('Threads', constants.LIMIT_CORES)
    # equals to 1 if (i,j) is in optimal route and 0 otherwise
    x_vars = {(i, j): opt_model.addVar(vtype=grb.GRB.BINARY,
                                       name="x_{0}_{1}".format(i, j))
              for i in set_0N1 for j in set_0N1}

    # equals to 1 if fitm provide a discount to customer i and 0 otherwise
    w_vars = {i: opt_model.addVar(vtype=grb.GRB.BINARY,
                                  name="w_{0}".format(i))
              for i in set_N}

    u_vars = {(i): opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                    lb=0,
                                    ub=constants.VEHICLE_CAPACITY,
                                    name="t_{0}".format(i)) for i in set_N1}

    # part of demand of PUP that this vehicel will take(inunits, not in %)
    z_vars = {p: opt_model.addVar(vtype=grb.GRB.INTEGER,
                                  lb=0,
                                  ub=constants.VEHICLE_CAPACITY,
                                  name="z_{0}".format(p)) for p in set_P}
    opt_model.update()

    # 0 == constraints on inflow once to customer node in  in scenario k customer is visited--------------
    constraints0 = {i: opt_model.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + w_vars[i] == 1,
                                           name="constraint_inflow_{0}".format(i)) for i in set_N}

    # 1 == constraints on inflow in PUP--------------
    constraints1 = {p: opt_model.addConstr(grb.quicksum(x_vars[p, j] for j in set_0N1) == 1,
                                           name="constraint_inflow_{0}".format(p)) for p in set_P}

    # 2-1 == constraints on fulfillment of PUP--------------
    constraints2 = {p: opt_model.addConstr(z_vars[p] <= constants.VEHICLE_CAPACITY * grb.quicksum(x_vars[p, j] for j in set_0N1),
                                           name="constraint_inflow_{0}".format(p)) for p in set_P}

    # 2-2 == constraint on fulfillment of PUP
    opt_model.addConstr(grb.quicksum(z_vars[p] for p in set_P) >= grb.quicksum(w_vars[i] for i in set_N),
                        name="constraint_pup")

    # 2 == forbid loops+
    constraints6 = {(i): opt_model.addConstr(x_vars[i, i] == 0,
                                             name="constraint_forbid_loops_{0}".format(i)) for i in set_N1}

    constraints7 = {(p, p1): opt_model.addConstr(x_vars[p, p1] == 0,
                                                 name="constraint_forbid_loops_{0}".format(p, p1)) for p in set_P for p1
                    in set_P}

    # 8-0 == constraint on outflow from the depot------------
    opt_model.addConstr(grb.quicksum(x_vars[0, j] for j in set_0N1) <= constants.FLEET_SIZE, name="constraint_depot_0")

    # 9 == constraint on connectivity+

    for i in set_0N1:
        sumin = 0
        sumout = 0
        for j in set_0N1:
            sumin += x_vars[j, i]
            sumout += x_vars[i, j]
        opt_model.addConstr(sumin - sumout == 0, name="constraint_connectivity_{0}".format(i))

    # -------------------------capacity and subtour elimination constraints-------------------------#

    # 7 == constraints on capacity and sutour elimination
    for i in set_N:
        for j in set_N1:
            if (i != j):
                opt_model.addConstr(
                    u_vars[i] - u_vars[j] + 1, grb.GRB.LESS_EQUAL, constants.BIGM * (1 - x_vars[i, j]),
                    "constraint_capacity_{0}_{1}".format(i, j))
    for p in set_P:
        for j in set_N:
            opt_model.addConstr(
                u_vars[p] - u_vars[j] + z_vars[p], grb.GRB.LESS_EQUAL, constants.BIGM * (1 - x_vars[p, j]),
                "constraint_capacity_{0}_{1}".format(p, j))

    # 8-0 == capacity constraint in PUP------------
    opt_model.addConstr(grb.quicksum(w_vars[i] for i in set_N) <= max_number_offered_discounts,
                        name="constraint_PUP_capacity_0")

    opt_model.update()

    objective = 0
    for i in set_N:
        objective += discount * w_vars[i]
    for i in set_0N1:
        for j in set_0N1:
            if j != i:
                objective += x_vars[i, j] * c[i, j]

    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)

    # opt_model.write("out_opt_model.lp")
    opt_model.optimize()

    opt_model.printAttr('X')

    # strategy of given discounts
    opt_strategy = [0]
    for i in set_N:
        if w_vars[i].x > 0.5:
            opt_strategy.append(1)
        else:
            opt_strategy.append(0)
    x = {}

    for i in set(sets['0N1']).union(set_P):
        for j in set(sets['0N1']).union(set_P):
            try:
                if x_vars[i, j].x > 0.5:
                    x[i, j] = 1
                else:
                    x[i, j] = 0
            except:
                x[i, j] = 0

   # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    # constrained_layout=True)
   # for ax in axs:
   #     ax.set_aspect(1.0)
   # print_vertex(input_data, opt_strategy, fig, axs[0], "deterministic", sets['N'])
   # print_route2(x, fig, axs[0], n, input_data['nodes'])

    return opt_model.objVal, x, opt_strategy
'''
n = 7
lowest_discount = {}
discount = 10
sets, input_data = initialise_VRP(n, "C101.csv", 1)
plt.figure(0)

ax, fig = plt.gca(), plt.gcf()
constrained_layout=True
ax.set_aspect(1.0)
print_vertex(input_data, [0] * (n + 1), fig, ax, "deterministic", sets['N'])
plt.show()
policy, cost = ring_star_deterministic_no_TW(discount, n, sets, input_data)
print("new", policy)
print(cost)
'''
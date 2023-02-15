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
        wvals = model.cbGetSolution(model._w_vars)
        visitvals = model.cbGetSolution(model._visit_vars)
        selected = grb.tuplelist((i, j) for i, j in model._x_vars.keys()
                                if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, model._set_visit)
        #print(tour)
        should_be_visited = n - sum(wvals[i] for i in model._set_N) + sum(visitvals[i] for i in range(n+1, n+1+model._NR_PUP))
        inpuptemp = [] # + grb.quicksum(model._w_vars[i] for i in inpuptemp)/(n-len(tour) +1+model._NR_PUP)
        for i in range(1, n + 1):
            if i not in tour:
                inpuptemp.append(i)
        #print("inpuptemp", inpuptemp)
        if len(tour) < should_be_visited:
            #print(len(tour))
            # add subtour elimination constr. for every pair of cities in tour

            if (tour[0]==0 and tour[len(tour)-1]>=n+1) :
                model.cbLazy(model._x_vars[tour[len(tour)-1],0] + grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                             <= len(tour) - 1 )
            elif (tour[0]>=n+1 and tour[len(tour)-1]==0):
                model.cbLazy(
                    model._x_vars[0, tour[0]] + grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                    <= len(tour) - 1 )
            else:
                model.cbLazy(grb.quicksum(model._x_vars[i, j] for i, j in combinations(tour, 2))
                            <= len(tour)-1  )

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


def ring_star_deterministic_no_TW(instance, max_number_offered_discounts, **kwargs):
    distanceMatrix = instance.distanceMatrix
    n = instance.NR_CUST
    if 'discount' in kwargs:
        discount = kwargs['discount']
    else:
        discount = instance.shipping_fee

    # initialize the full graph
    set_N = range(1,n+1)
    set_N1 = range(1,n+instance.NR_PUP+1)
    set_0N1 = range(n+instance.NR_PUP+1)

    max_pup = min(max_number_offered_discounts, constants.PUP_CAPACITY)
    m = grb.Model()
    m._set_visit = [i for i in set_0N1]
    m._n = n
    m._NR_PUP = instance.NR_PUP
    m._set_N = set_N
    m.setParam('OutputFlag', False)

    # Create variables
    x_vars = m.addVars(distanceMatrix.keys(), obj=distanceMatrix, vtype=GRB.BINARY, name='e')
    for i, j in x_vars.keys():
        if (i==0 and j>=n+1) or (i>=n+1 and j==0): #?
            continue
        else:
            x_vars[j, i] = x_vars[i, j]  # edge in opposite direction

    w_vars = grb.tupledict()
    for i in set_N:
        w_vars[i] = m.addVar(obj=discount[i], vtype=GRB.BINARY, name='w[%d]'%(i))

    visit_vars = grb.tupledict()

    for pup in instance.pups:
        visit_vars[pup.id] = m.addVar(obj=0, vtype=GRB.BINARY, name='visit[%d]'%(pup.id))
        m.addConstr(n*visit_vars[pup.id]>= grb.quicksum(w_vars[cust_id] for cust_id in pup.closest_cust_id ),
                    name="visitbigM1_{0}".format(pup.id))
        m.addConstr( visit_vars[pup.id] <= grb.quicksum(w_vars[cust_id] for cust_id in pup.closest_cust_id),
                     name="visitbigM2_{0}".format(pup.id))
    #m.addConstr(w_vars[0] == 0)
    #m.addConstr(w_vars[n+1] == 0)
    m.addConstr(grb.quicksum(w_vars[i] for i in set_N) <= max_pup)

    # forbid loops
    for i in set_0N1:
        m.addConstr(x_vars[i, i] == 0, name="forbid_loop_{0}".format(i))

    # Add degree-2 constraint
    for i in set_0N1:
        if i==0:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + grb.quicksum(x_vars[p,0] for p in range(n+1, n+instance.NR_PUP + 1))== 2,
                        name="constraint_inflow2_{0}".format(i))
        elif i >= n + 1:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + x_vars[0,i] == 2 * visit_vars[i],
                        name="constraint_inflow2_{0}".format(i))
        else:
            m.addConstr(grb.quicksum(x_vars[i, j] for j in set_0N1) + 2*w_vars[i] == 2,
                            name="constraint_inflow2_{0}".format(i))
    # Optimize model
    m._x_vars = x_vars
    m._w_vars = w_vars
    m._visit_vars = visit_vars
    m.Params.lazyConstraints = 1

    m.optimize(subtourelim)
    m.write("upd.lp")
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



def ring_star_deterministic_no_TW_old(instance, max_number_offered_discounts, **kwargs):
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
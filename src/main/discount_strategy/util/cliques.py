from pathlib import Path
import sys
import os

from itertools import combinations
#import numpy as np
import networkx as nx

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)
import constants


def k_cliques(graph, max_k):
    # 2-cliques
    cliques = [{i, j} for i, j in graph.edges() if i != j]
    k = 2

    while cliques and k < max_k:
        # merge k-cliques into (k+1)-cliques
        cliques_1 = set()
        for u, v in combinations(cliques, 2):
            w = u ^ v
            if len(w) == 2 and graph.has_edge(*w):
                cliques_1.add(tuple(u | w))

        # remove duplicates
        cliques = list(map(set, cliques_1))
        k += 1
    cliques = list(map(list, cliques))
    return k, cliques

def find_cliques(input_data):

    nodes = input_data["n"]
    discount = input_data["discount"]

    graph = nx.Graph()
    graph.add_nodes_from(range(nodes+1))
    edges = input_data['distances'].copy()
    clique_left = True

    cluster = {}
    for clique_size in range(2, nodes+2):
        if clique_left:
            for (i,j) in edges:
                if edges[(i,j)] <= discount *clique_size/(clique_size+1):
                    graph.add_edge(i, j)
            #plt.figure(figsize=(8, 8))
            #nx.draw_networkx(graph)
            #plt.show()
            k, cliques = k_cliques(graph, clique_size)
            #print('%d-cliques: #%d, %s ...' % (k, len(cliques), cliques[:8]))
            if (k < clique_size) or len(cliques) == 0:
                clique_left = False
            else:
                cluster[clique_size] = cliques
    return cluster


#list of all possible ub insertion cost that are cheapper than min_default = dist[0,node] or dist[n+1,node]
def sort_worst_ub(node, dist,n):
    min_default = min(dist[0, node], dist[n + 1, node])
    min_insertion = {}

    min_insertion[0] = min_default
    for i in range(1, n+1):
        if i!= node:
            if dist[i, node] < min_default:
                min_insertion[i] = dist[i,node]
    sorted_insertion = {k: v for k, v in sorted(min_insertion.items(), key=lambda item: item[1])}
    return sorted_insertion
#list of all possible lb insertion cost that are cheapper than min_default = dist[0,node] or dist[n+1,node]
def sort_worst_lb(node, dist,n):
    min_default = max(0,min(2*dist[0, node], 2*dist[n + 1, node], dist[0, node] + dist[n + 1, node] - dist[0,n+1]))
    min_insertion = {}
    min_insertion[0,0] = min_default
    for i in range(0, n+2):
        if i != node:
            for j in range(i,n+2):
                if j!= node:
                    if dist[i, node]+dist[node,j] - dist[i,j] < min_default:
                        min_insertion[i,j] = max(0, dist[i, node]+dist[node,j] - dist[i,j])
    sorted_insertion = {k: v for k, v in sorted(min_insertion.items(), key=lambda item: item[1])}
    return sorted_insertion

#calculate the worst UB on the insetion cost given the information about customers with discount
def ub_insertion_cost(ub_insertion, node, set_no_discount, p_dev):
    insertion_cost = 0
    p_left = 1
    for i in ub_insertion:
        if i!=0:
            if i in set_no_discount:
                insertion_cost += ub_insertion[i] *  p_left
                p_left = 0
            else:
                insertion_cost += ub_insertion[i]*p_dev*p_left
                p_left *= (1-p_dev)
        else:
            insertion_cost += ub_insertion[i] * p_left
    return insertion_cost*2

def lb_insertion_cost(lb_insertion, node, set_discount, p_dev, known):
    insertion_cost = known['val']
    p_left = 1-known['prob']
    previous_insertion = []
    for (i,j) in lb_insertion:
        if p_left>0:
            if [i, j] != [0, 0]:
                prob_current = 1
                if i == j:
                    if i in set_discount:
                        prob_current = p_dev
                else:
                    if i in set_discount and j in set_discount:
                        prob_current *= (p_dev**2)
                    elif (i in set_discount and j not in set_discount) or (j in set_discount and i not in set_discount):
                        prob_current *= p_dev
                for (m,k) in previous_insertion:
                    if m == i or m == j:
                        if k in set_discount:
                            prob_current *= (1-p_dev)
                        else:
                            prob_current *= 0
                    if k == i or k == j:
                        if m in set_discount:
                            prob_current *= (1-p_dev)
                        else:
                            prob_current *= 0
                insertion_cost += min(prob_current,p_left)*lb_insertion[i,j]
                p_left -= prob_current
                previous_insertion.append([i, j])
            else:
                insertion_cost += lb_insertion[i,j] * p_left
        else: break

    return insertion_cost

'''
cliques = find_cliques(input_data)
    delete = []
    if len(cliques) > 0:
        for y, policy in Y.items():
            cl_size = 2
            while cl_size <= max(cliques, key = int):
                for cluster in cliques[cl_size]:
                    if n+1 in cluster:
                        if [policy[i] for i in cluster if i != n + 1].count(0) < cl_size-1:
                            delete.append(y)
                            cl_size = n + 3
                            break
                        else:
                            pass
                    else:
                        if [policy[i] for i in cluster].count(policy[cluster[0]]) < cl_size:
                            delete.append(y)
                            cl_size = n+3
                            break
                        else:
                            pass
                cl_size +=1


    for i in delete:
        del Y[i]
'''
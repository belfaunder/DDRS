from pathlib import Path
import sys
import os
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]),"util")
sys.path.insert(1, path_to_util)

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


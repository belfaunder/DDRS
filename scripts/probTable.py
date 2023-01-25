import pandas as pd
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from scipy.special import comb
import itertools


path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"src","main","discount_strategy", "algorithms", "exact")
sys.path.insert(2, path_to_exact)
from TSPSolver import TSPSolver

path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"src","main","discount_strategy", "io")
sys.path.insert(1, path_to_io)
import OCVRPParser

def bitCount(int_type):
    count = 0
    while (int_type):
        int_type &= int_type - 1
        count += 1
    return (count)

def heatMap(df):
    #Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    #Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(df, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(df.columns)), df.columns);
    #Apply yticks
    plt.yticks(range(len(df.columns)), df.columns)
    #show plot
    plt.show()


n = 6
p_dev = 0.2

numDisc = n
p_devs = [0.1, 0.3,0.5,0.7,0.9]
colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y']
plt.figure(0)
color_iter = 0
ax, fig = plt.gca(), plt.gcf()
for p in p_devs:

    probs = []
    for numDev in range(numDisc+1):
        probs.append(((1 - p) ** (numDisc - numDev) * p ** (numDev)) * comb(numDisc, numDev))
    ax.plot(probs, color = colors[color_iter], label='P_choose_disc: ' + str(round((1-p)*100))+"%")
    color_iter +=1
    ax.set_title("Cumulative probability of deviated scenarios \nfor one policy. N: "+str(numDisc))
    plt.xlabel('Num_deviated cusomers \n(=Num of visited customers if all customers received discount)')
    ax.legend(bbox_to_anchor=(1, 0.5))
plt.show()

costs = []
costs_disc =[]
setMayVary = list(range(1,n+1))

file_instance = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "instances_set18_DiscVar",
                                 "instance6_R1_11_3.3_50.txt")
OCVRPInstance = OCVRPParser.parse(file_instance)
solver = TSPSolver(instance=OCVRPInstance, solverType = 'Gurobi')

for gamma in range(n+1):
    route_cost_gamma = 0
    num_gamma = 0
    for combination in itertools.combinations(setMayVary, gamma):
        scenario = 2^n-1
        for offset in combination:
            mask = ~(1 << (offset - 1))
            scenario = scenario & mask
        route_cost_gamma += solver.tspCost(scenario)
        num_gamma += 1
    costs.append(route_cost_gamma/num_gamma)


plt.figure(0)
ax, fig = plt.gca(), plt.gcf()
plt.scatter(list(range(n+1)),costs)
ax.set_title("Average scenario costs")
plt.xlabel('Num of visited customers')
plt.show()

data = np.zeros((2**n+1,2**n+1))
data[2**n, 0] = 2**n
data[0,2**n]=2**n
for policy in range(2**n):
    data[policy, 0] = policy
    for scenario in range(2 ** n):
        data[0,scenario]=scenario
        prob_is_zero = False
        for offset in range(n):
            mask = 1<<offset
            if mask&scenario and not mask&policy:
                prob_is_zero = True
        if not prob_is_zero:

            #print("scenario, policy", scenario, policy,  (1 - p_dev) ** bitCount(policy & scenario) * p_dev ** (bitCount(policy ^ scenario)))
            data[policy+1, scenario+1] = (1 - p_dev) ** bitCount(policy & scenario) * p_dev ** (bitCount(policy ^ scenario))



df = pd.DataFrame(data=data[1:,1:],    # values
              index=data[1:,0],    # 1st column as index
              columns=data[0,1:])


print("")

#sns.heatmap(df, annot=True, fmt=".1f")
sns.heatmap(df, annot=False, cmap="YlGnBu")
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.title("probability table for "+str(n)+" customers." +" Prob_choose_disc " +str(round((1-p_dev)*100,1)))
plt.ylim(b, t)

#plt.show()

print(file_instance)


costs = []
costs_disc =[]
for scenario in range(2**n):
    route_cost = solver.tspCost(scenario)
    disc_cost = 3.3*bitCount(scenario)
    costs.append(route_cost)
    costs_disc.append(disc_cost+route_cost)


plt.figure(0)
ax, fig = plt.gca(), plt.gcf()
ax.plot(costs)
ax.plot(costs_disc)
ax.set_title("scenario costs")
#plt.show()







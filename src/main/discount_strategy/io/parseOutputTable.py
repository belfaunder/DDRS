#!/usr/bin/python3
import seaborn as sns
from pathlib import Path
import sys
import os
import csv
import math
import matplotlib
import pandas as pd
from openpyxl import load_workbook
# import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rc
import datetime
import numpy as np

path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "io")
sys.path.insert(2, path_to_io)
import OCVRPParser

path_to_exact = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(3, path_to_exact)
from ring_star_without_TW import ring_star_deterministic_no_TW

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "util")
sys.path.insert(1, path_to_util)
import constants
from bit_operations import bitCount

path_to_data = constants.PATH_TO_DATA
path_to_images = constants.PATH_TO_IMAGES
import pickle

colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', 'indigo', 'deeppink']


def def_plt_settings():
    sns.set()
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, sharex=True)


def calculate_exp_disc_cost(policy, file_instance, folder_data):
    file_instance = os.path.join(folder_data, file_instance + ".txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    discount_cost = 0
    for i in OCVRPInstance.customers:
        if policy & (1 << (i.id - 1)):
            discount_cost += (1 - OCVRPInstance.p_home[i.id]) * OCVRPInstance.shipping_fee[i.id]
    return discount_cost


def calculate_lower_bounds(file_instance, folder_data):
    file_instance = os.path.join(folder_data, file_instance + ".txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    # __, lower_bound = ring_star_deterministic_no_TW(OCVRPInstance,  OCVRPInstance.NR_CUST)
    __, lower_bound = ring_star_deterministic_no_TW(OCVRPInstance, OCVRPInstance.NR_CUST,
                                                    discount=[0] * (OCVRPInstance.NR_CUST + 1))

    return lower_bound


def calculate_upper_bounds(file_instance, folder_data):
    file_instance = os.path.join(folder_data, file_instance + ".txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    __, upper_bound = ring_star_deterministic_no_TW(OCVRPInstance, 0)

    return upper_bound


def calculate_exp_pup_utilization(policy, file_instance, folder_data):
    file_instance = os.path.join(folder_data, file_instance + ".txt")
    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()
    in_pup = 0
    for i in OCVRPInstance.customers:
        if policy & (1 << (i.id - 1)):
            in_pup += (1 - OCVRPInstance.p_home[i.id])
        else:
            in_pup += OCVRPInstance.p_pup[i.id]
    return int(round(100 * in_pup / 15, 0))


def writer(file_path, data, rowTitle):
    outputFile = file_path
    # outputFile = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", file_path)

    with open(outputFile, mode='a', newline='', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(rowTitle)
        for row in data:
            writer.writerow(row)


def heuristic_comparison_probability_table(df):
    df = df[df.discount_code == 1].copy()
    df = df[['prob_choose_disc', 'RS-BAB,AV', 'NODISC-BAB,LB']].copy()
    # df['RS-BAB,AV'] = df['RS-BAB,AV']*100
    # df['RS-BAB,AV'] = df['RS-BAB,AV'] * 100
    df['RS_BAB,AV_max'] = df['RS-BAB,AV']
    df['RS_BAB,AV_min'] = df['RS-BAB,AV']
    df['NODISC-BAB,LB_max'] = df['NODISC-BAB,LB']
    df['NODISC-BAB,LB_min'] = df['NODISC-BAB,LB']
    df.columns = ["Probability", "RS-BAB, MEAN", "NODISC-BAB,MEAN", "RS-BAB, MAX", "RS-BAB, MIN", "NODISC-BAB,MAX",
                  "NODISC-BAB,MIN"]
    df = df.groupby(['Probability']).agg({"RS-BAB, MEAN": 'mean', "NODISC-BAB,MEAN": 'mean',
                                          "NODISC-BAB,MIN": 'min', "NODISC-BAB,MAX": 'max', "RS-BAB, MIN": 'min',
                                          "RS-BAB, MAX": 'max'})
    print('')
    print(df.dtypes)
    print(df.to_latex(float_format='{:.1%}'.format))


def heuristic_comparison_probability(df, column_compare):
    tips = sns.load_dataset("tips")
    ax = sns.boxplot(x='prob_choose_disc', y=column_compare, hue='nrCust', data=df, linewidth=2.5, palette="Blues")

    if 'NODISC' in column_compare:
        plt.ylabel('NODISC - BAB')
        name = 'Effect_of_Probability_det_heuristic_NODISC_HEURISTIC.png'
        # plt.title(str(
        #    'Effect of Probability to choose discounted option on\nDifference in NODISC strategy  and Heuristic solutions, n:20,30,40,50'))

    else:
        plt.ylabel('RS - BAB')
        name = 'Effect_of_Probability_det_heuristic_DET_HEURISTIC.png'
        # plt.title(str(
        #    'Effect of Probability to choose discounted option on\nDifference in D-OCTSP S_OSTSP solutions'))

    plt.xlabel('Probability to choose discounted option ' + r"($\rho$)")
    vals = ax.get_yticks()
    print(vals)
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    ax.legend()
    # ax.legend(ncol=2, handles=handle_list)
    plt.grid(axis='y')
    path = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "images")
    plt.savefig(os.path.join(path, name))
    plt.show()


def heuristic_comparison_problemSize(df, column_compare):
    iter = 0
    handle_list = []
    mean_handle = mlines.Line2D([], [], color='black', label='Mean Value', marker='^')
    instance_handle = mlines.Line2D([], [], color='black', label='One instance', marker='_')
    handle_list.append(mean_handle)
    handle_list.append(instance_handle)
    scale = 1
    shift = 1

    for prob in df.prob_choose_disc.unique():
        mean, min, max = [], [], []
        ax = plt.gca()
        # prob and n are constant for those instances
        df1 = df[df.prob_choose_disc == prob]
        # for disc in df.discount_code.unique():
        for nrCust in df1.nrCust.unique():
            df_temp = df1[df1.nrCust == nrCust]
            # for prob in df1.prob_choose_disc.unique():
            mean.append(df_temp[column_compare].mean() * 100)
            ax.scatter([nrCust + iter * scale - shift] * len(df_temp[column_compare]), df_temp[column_compare] * 100,
                       color=colors[iter], marker='_', s=20.0)
            min.append(df_temp[column_compare].mean() * 100 - df_temp[column_compare].min() * 100)
            max.append(df_temp[column_compare].max() * 100 - df_temp[column_compare].mean() * 100)

        data_handler = ax.errorbar([i + iter * scale - shift for i in df1.nrCust.unique()], np.array(mean),
                                   yerr=[min, max],
                                   fmt='r^', mfc=colors[iter], mec=colors[iter], ecolor=colors[iter], capthick=2,
                                   label='Prob. ' + r"$\rho: $ " + str(prob), elinewidth=0.33)
        handle_list.append(data_handler)
        iter += 1

    if 'NODISC' in column_compare:
        plt.ylabel('AVERAGE |NODISC. - H.|, %')
    else:
        plt.ylabel('AVERAGE |DET. - H.|,%')

    plt.xlabel('Problem size, n')
    # plt.title('')
    ax.legend(ncol=2, handles=handle_list)
    plt.savefig('Heuristics_vs_RS' + column_compare.split(',')[0] + '.png')
    plt.show()


def heuristic_bab_comparison(df, dfHeuristic):
    nrCust = 20
    df1 = dfHeuristic[dfHeuristic.nrCust == nrCust].copy()

    for i, row in df1.iterrows():
        # f1.loc[df1["instance"] == row["instance"], 'EXACT_AVERAGE'] = (df.loc[df["instance"] == row["instance"], 'objectiveUB_BAB']+df.loc[df["instance"] == row["instance"], 'objectiveLB_BAB'])/2
        df1.loc[df1["instance"] == row["instance"], 'HEURISTIC_AVERAGE'] = (df1.loc[df1["instance"] == row[
            "instance"], 'objectiveUB_BAB'] + df1.loc[df1["instance"] == row["instance"], 'objectiveLB_BAB']) / 2
        # df1.loc[df1["instance"] == row["instance"], 'EXACT_AVERAGE'] = (df1.loc[df1["instance"] == row[
        #    "instance"], 'objectiveUB_BAB'] + df1.loc[df1["instance"] == row["instance"], 'objectiveLB_BAB']) / 2
    for i, row in df.iterrows():
        df.loc[df["instance"] == row["instance"], 'EXACT_AVERAGE'] = (df.loc[df["instance"] == row[
            "instance"], 'objectiveUB_BAB'] + df.loc[df["instance"] == row["instance"], 'objectiveLB_BAB']) / 2

    ax = plt.gca()
    df1.plot(kind='scatter', x='prob_choose_disc', y='HEURISTIC_AVERAGE', ax=ax, c=colors[0], marker='_')
    df.plot(kind='scatter', x='prob_choose_disc', y='EXACT_AVERAGE', ax=ax, c=colors[1], marker='_')
    mean = []
    mean_h = []
    for prob in [40, 60, 80]:
        print(prob)
        df_temp_h = df1[df1.prob_choose_disc == prob]
        df_temp_e = df[df.prob_choose_disc == prob]

        mean.append(df_temp_e['EXACT_AVERAGE'].mean())
        mean_h.append(df_temp_h['HEURISTIC_AVERAGE'].mean())
    plt.plot([i for i in df1.prob_choose_disc.unique()], mean, c=colors[1], label='Heuristic')
    plt.plot([i for i in df1.prob_choose_disc.unique()], mean_h, c=colors[0], label='Exact')
    plt.ylabel('HEURISTIC - EXACT, average cost')
    ax.legend()
    plt.show()

def parseProfile(file_path):

    # print("whatch out, I print the best known LB instead on Obj.Val")
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            #try:
            if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])
                    p_home = float(instance.split('_')[4])
                    discount = float((instance.split('_')[8]).split('.txt')[0])
                    try:
                        nodes = float(' '.join(lines[idx + 5].split()).split(' ')[1])
                        num_tsps = float(lines[idx + 6].split(':')[1])
                        obj_val = float(' '.join(lines[idx + 8].split()).split(' ')[3])
                        time_first_opt = float(lines[idx + 10].split(':')[1])
                        policy_ID = int(lines[idx + 11].split(':')[1])
                        num_disc = bitCount(policy_ID)

                        pruned_by_cliques_nonleaf = int(lines[idx + 13].split(':')[1])
                        pruned_by_cliques_leaf = int(lines[idx + 14].split(':')[1])
                        pruned_by_rs_nonleaf = int(lines[idx + 15].split(':')[1])
                        pruned_by_rs_leaf = int(lines[idx + 16].split(':')[1])
                        pruned_by_insertionCost_nonleaf = int(lines[idx + 17].split(':')[1])
                        pruned_by_insertionCost_leaf = int(lines[idx + 18].split(':')[1])
                        pruned_by_bounds_nonleaf = int(lines[idx + 19].split(':')[1])
                        pruned_by_bounds_leaf = int(lines[idx + 20].split(':')[1])

                        tsp_time = 0
                        time_exact_bounds = 0
                        time_lb_addition = 0
                        time_branch = 0
                        iter = idx + 25

                        time_running = float(lines[idx + 4].split(':')[1])
                        '''
                        while iter < idx + 80:
                            iter +=1
                            if '(runBranchAndBound)' in lines[iter]:
                                time_running = float(' '.join(lines[iter].split()).split(' ')[3])
                            if '(tspCost)' in lines[iter]:
                                tsp_time = float(' '.join(lines[iter].split()).split(' ')[3])
                            if '(updateBoundsFromDictionary)' in lines[iter]:
                                time_exact_bounds += float(' '.join(lines[iter].split()).split(' ')[3])
                            if '(updateByInsertionCost)' in lines[iter]:
                                time_exact_bounds -= float(' '.join(lines[iter].split()).split(' ')[3])
                                time_lb_addition += float(' '.join(lines[iter].split()).split(' ')[3])
    
                            if '(updateBoundsWithNewTSPs)' in lines[iter]:
                                time_exact_bounds += float(' '.join(lines[iter].split()).split(' ')[1])
                            if '(scenarioProb_2segm)' in lines[iter]:
                                time_exact_bounds += float(' '.join(lines[iter].split()).split(' ')[3])
                            if '(exploreNode)' in lines[iter]:
                                time_lb_addition += float(' '.join(lines[iter].split()).split(' ')[1])
                            if '(set_probability_covered)' in lines[iter]:
                                time_lb_addition += float(' '.join(lines[iter].split()).split(' ')[3])
                            if '(branch)' in lines[iter]:
                                time_branch += float(' '.join(lines[iter].split()).split(' ')[3])
                        '''
                        # data.append([eps, nrCust,p_home, p_pup, discount, time_running, time_first_opt, nodes, num_tsps, optimal,gap, obj_val, sd,
                        #               policy_ID, num_disc, instance])
                        data.append(
                            [nrCust, nrPup, p_home,  discount, time_running, time_first_opt, nodes, num_tsps,
                              obj_val,  policy_ID, num_disc, instance,
                             pruned_by_cliques_nonleaf, pruned_by_cliques_leaf, pruned_by_rs_nonleaf, pruned_by_rs_leaf,
                             pruned_by_insertionCost_nonleaf, pruned_by_insertionCost_leaf, pruned_by_bounds_nonleaf,
                             pruned_by_bounds_leaf, tsp_time*100/time_running, time_exact_bounds*100/time_running, time_lb_addition*100/time_running, time_branch*100/time_running ])
                    except:
                        data.append(
                            [nrCust, nrPup, p_home, discount])
                #except:
                    #data.append([nrCust, nrPup, p_home,  discount, "", "", "", "", "", "", "",
                    #             "", "", "", instance])
                    #print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['nrCust',"nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'obj_val_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                'pr_cliques_nonleaf', 'pr_cliques_leaf', 'pr_rs_nonleaf', 'pr_rs_leaf',
                'pr_insertionCost_nonleaf', 'pr_insertionCost_leaf', 'pr_bounds_nonleaf',
                'pr_bounds_leaf','tsp_time', 'time_exact_bounds', 'time_lb_addition', 'time_branch']
    df = pd.DataFrame(data,columns=rowTitle)
    return df
    #writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def parseBAB(file_path, folder, output_name):
    # print("whatch out, I print the best known LB instead on Obj.Val")
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                if 'Instance' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])

                    eps = round(float(lines[idx + 3].split(':')[1].split()[0])/100, 4)
                    p_home = instance.split('_')[4]
                    p_pup = instance.split('_')[6]
                    instance_id = instance.split('_')[-1]


                    discount = (instance.split('_')[8]).split('.txt')[0]
                    time_running = float(lines[idx + 4].split(':')[1])
                    nodes = float(lines[idx + 5].split(':')[1])
                    num_tsps = float(lines[idx + 6].split(':')[1])
                    optimal = int(lines[idx + 7].split(':')[1])
                    obj_val = float(lines[idx + 8].split(':')[1].split('Obj_val(lb)')[0])
                    try:
                        pruned_by_cliques_l= int(lines[idx + 14].split(':')[1])
                        pruned_by_cliques_nl= int(lines[idx + 13].split(':')[1])
                        pruned_by_rs_l =  int(lines[idx + 16].split(':')[1])
                        pruned_by_rs_nl = int(lines[idx + 15].split(':')[1])
                        pruned_by_insertionCost_nl= int(lines[idx + 17].split(':')[1])
                        pruned_by_insertionCost_l = int(lines[idx + 18].split(':')[1])
                        pruned_by_bounds_nl = int(lines[idx + 19].split(':')[1])
                        pruned_by_bounds_l = int(lines[idx + 20].split(':')[1])
                    except:
                        pruned_by_cliques = 0
                        pruned_by_insertionCost = 0
                        pruned_by_bounds = 0
                    #obj_val = float(lines[idx + 13].split('[')[1].split(',')[0])
                    # 2sd:
                    #sd = float(lines[idx + 13].split('[')[1].split(',')[0]) - float(
                    #    lines[idx + 13].split('[')[1].split(',')[1])
                    # obj_val = float(lines[idx + 12].split('best_known_LB')[1])
                    #gap = float(lines[idx + 9].split(':')[1])
                    time_first_opt = float(lines[idx + 10].split(':')[1])
                    policy_ID = int(lines[idx + 11].split(':')[1])

                    num_disc = bitCount(policy_ID)
                    # data.append([eps, nrCust,p_home, p_pup, discount, time_running, time_first_opt, nodes, num_tsps, optimal,gap, obj_val, sd,
                    #               policy_ID, num_disc, instance])
                    data.append(
                        [eps, nrCust, nrPup, p_home,  discount, time_running, time_first_opt, nodes, num_tsps, optimal,'',
                         obj_val,'', policy_ID, num_disc, instance,instance_id, p_pup, pruned_by_cliques_nl,
                         pruned_by_cliques_l,pruned_by_rs_nl, pruned_by_rs_l, pruned_by_insertionCost_nl, pruned_by_insertionCost_l, pruned_by_bounds_nl, pruned_by_bounds_l])
            except:
                data.append([eps, nrCust, nrPup, p_home,  discount, "", "", "", "", "", "", "",
                             "", "", "", instance, p_pup])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['eps', 'nrCust',"nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'optimal', 'gap', 'obj_val_bab', '2sd_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                'instance_id','p_pup', 'pruned_by_cliques_nl','pruned_by_cliques_l','pruned_by_rs_nl', 'pruned_by_rs_l',
                                             'pruned_by_insertionCost_nl', 'pruned_by_insertionCost_l', 'pruned_by_bounds_nl', 'pruned_by_bounds_l']
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def parseBAB_REMOTE(file_path, folder, output_name):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                if 'Instance' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])

                    p_home = instance.split('_')[4]
                    instance_id = instance.split('_')[-1]
                    discount = (instance.split('_')[8]).split('.txt')[0]
                    obj_val = float(lines[idx + 5].split('[')[1].split(',')[0])
                    policy_ID = int(lines[idx + 4].split(':')[1])

                    data.append(
                        [ instance, nrCust, nrPup, p_home,  discount,   obj_val, policy_ID, instance_id])
            except:
                data.append([ instance])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['instance', 'nrCust_rem',"nrPup_rem", 'p_home_rem', 'discount_rate_rem', 'obj_val_remote', 'policy_remote_ID', 'instance_id_rem' ]
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)

def parseBAB_temp(file_path, folder, output_name):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                if 'Instance' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])
                    eps = round(float(lines[idx + 3].split(':')[1].split()[0])/100, 4)
                    p_min = instance.split('_')[4]
                    p_av = instance.split('_')[6]
                    p_max = instance.split('_')[8]
                    l_min = instance.split('_')[10]
                    l_max = instance.split('_')[12]
                    instance_id = instance.split('_')[-1]
                    discount = (instance.split('_')[14]).split('.txt')[0]
                    time_running = float(lines[idx + 4].split(':')[1])
                    nodes = float(lines[idx + 5].split(':')[1])
                    num_tsps = float(lines[idx + 6].split(':')[1])
                    optimal = int(lines[idx + 7].split(':')[1])
                    obj_val = float(lines[idx + 8].split(':')[1].split('Obj_val(lb)')[0])
                    #obj_val = float(lines[idx + 13].split('[')[1].split(',')[0])
                    # 2sd:
                    #sd = float(lines[idx + 13].split('[')[1].split(',')[0]) - float(
                    #    lines[idx + 13].split('[')[1].split(',')[1])
                    # obj_val = float(lines[idx + 12].split('best_known_LB')[1])
                    #gap = float(lines[idx + 9].split(':')[1])
                    time_first_opt = float(lines[idx + 10].split(':')[1])
                    policy_ID = int(lines[idx + 11].split(':')[1])

                    num_disc = bitCount(policy_ID)
                    # data.append([eps, nrCust,p_home, p_pup, discount, time_running, time_first_opt, nodes, num_tsps, optimal,gap, obj_val, sd,
                    #               policy_ID, num_disc, instance])
                    data.append(
                        [ nrCust, nrPup, p_min, p_av, p_max, l_min, l_max,  discount, time_running, time_first_opt, nodes, num_tsps, optimal,
                         obj_val,policy_ID, num_disc, instance,instance_id  ])
            except:
                data.append([eps, nrCust, nrPup,   discount, "", "", "", "", "", "", "",
                             "", "", "", instance])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = [ 'nrCust',"nrPup", 'p_min',"p_av","p_max","l_min", "l_max", 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'optimal', 'obj_val_bab',    'policy_bab_ID', 'num_disc_bab', 'instance', 'instance_id' ]
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def parseBAB_RS_NODISC(file_path, folder, output_name):
    # print("whatch out, I print the best known LB instead on Obj.Val")
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            #try:
            if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])
                    p_home = instance.split('_')[4]
                    discount_rate = (instance.split('_')[8]).split('.txt')[0]

                    time_bab = float(lines[idx + 4].split(':')[1])
                    time_tb = float(lines[idx + 10].split(':')[1])
                    nodes = float(lines[idx + 5].split(':')[1])
                    num_tsps = float(lines[idx + 6].split(':')[1])
                    optimal = int(lines[idx + 7].split(':')[1])
                    obj_val_bab = float(lines[idx + 13].split('[')[1].split(',')[0])
                    # 2sd:
                    sd_bab = float(lines[idx + 13].split('[')[1].split(',')[0]) - float(
                        lines[idx + 13].split('[')[1].split(',')[1])
                    policy_ID_bab = int(lines[idx + 11].split(':')[1])
                    try:
                        obj_val_rs = float(lines[idx + 15].split('[')[1].split(',')[0])
                        # 2sd:
                        sd_rs = float(lines[idx + 15].split('[')[1].split(',')[0]) - float(
                            lines[idx + 15].split('[')[1].split(',')[1])
                        policy_ID_rs = int(lines[idx + 14].split(':')[1])
                        obj_val_nodisc = float(lines[idx + 16].split('[')[1].split(',')[0])
                        obj_val_uniform = float(lines[idx + 17].split('[')[1].split(',')[0])
                        sd_uniform = float(lines[idx + 17].split('[')[1].split(',')[0]) - float(
                            lines[idx + 17].split('[')[1].split(',')[1])

                        gap_rs = (obj_val_rs - obj_val_bab) / obj_val_bab * 100
                        gap_nodisc = (obj_val_nodisc - obj_val_bab) / obj_val_bab * 100
                        gap_uniform = (obj_val_uniform - obj_val_bab) / obj_val_bab * 100
                        data.append(
                            [instance, nrCust,nrPup,  p_home,  discount_rate, policy_ID_bab, obj_val_bab, sd_bab, time_bab,
                             time_tb, nodes,
                             num_tsps, optimal, policy_ID_rs, obj_val_rs, sd_rs, gap_rs, obj_val_nodisc,
                             gap_nodisc,
                             obj_val_uniform, sd_uniform, gap_uniform])
                    except:
                        data.append(
                            [instance, nrCust,nrPup, p_home,  discount_rate, policy_ID_bab, obj_val_bab, sd_bab,
                             time_bab,
                             time_tb, nodes,
                             num_tsps, optimal, "", "", "", "", "", "",  "", "", "", ""])

            #except:
            else:
                data.append([ nrCust, p_home,  discount_rate, "", "", "", "", "", "", "",
                             "", "", "", instance])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['instance', 'nrCust', 'nrPup', 'p_home',  'discount_rate', 'policy_bab_ID', 'obj_val_bab', '2sd_bab',
                'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'optimal', 'policy_ID_rs', 'obj_val_rs', '2sd_rs', 'gap_rs', 'obj_val_nodisc',
                'gap_nodisc', 'obj_val_uniform', '2sd_uniform', 'gap_uniform']
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def parseBABHeuristic(file_path, folder, output_name):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # for idx, line in enumerate(lines):
        # if 'Instance:' in line:
        #    print(lines[idx - 2].split(':')[1])
        #    data.append(["HEURISTIC_SAMPLE_SIZE:",lines[idx - 2].split(':')[1].replace('\n', '')])
        #    data.append(["HEURISTIC_TIME_LIMIT:",lines[idx - 1].split(':')[1].replace('\n', '')])
        ##    data.append(["SOLVER_TYPE:",lines[idx - 3].split(':')[1].replace('\n', '')])
        #    break

        for idx, line in enumerate(lines):
            try:
            #if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])

                    eps = round(float(lines[idx -1].split(':')[1]), 3)
                    p_home = instance.split('_')[4]
                    p_pup = instance.split('_')[6]
                    discount = (instance.split('_')[8]).split('.txt')[0]
                    sample = float(lines[idx -4].split(':')[1])
                    time_limit = float(lines[idx -3].split(':')[1])
                    time_bab = float(lines[idx + 4].split(':')[1])
                    nodes = float(lines[idx + 5].split(':')[1])
                    num_tsps = float(lines[idx + 6].split(':')[1])
                    obj_val_bab = float(lines[idx + 10].split('[')[1].split(',')[1])
                    # 2sd:
                    sd_bab = float(lines[idx + 10].split('[')[1].split(',')[0]) - float(
                        lines[idx + 10].split('[')[1].split(',')[1])
                    policy_ID_bab = int(lines[idx + 8].split(':')[1])
                    num_offered_disc_bab =  bitCount(policy_ID_bab)
                    try:
                        obj_val_rs = float(lines[idx + 12].split('[')[1].split(',')[2].split(']')[0])
                        # 2sd:
                        sd_rs = float(lines[idx + 12].split('[')[1].split(',')[0]) - float(
                            lines[idx + 12].split('[')[1].split(',')[1])

                        policy_ID_rs = int(lines[idx + 11].split(':')[1])
                        obj_val_nodisc = float(lines[idx + 13].split('[')[1].split(',')[0])

                        obj_val_uniform = float(lines[idx + 14].split('[')[1].split(',')[0])
                        sd_uniform = float(lines[idx + 14].split('[')[1].split(',')[0]) - float(
                            lines[idx + 14].split('[')[1].split(',')[1])

                        gap_rs = (obj_val_rs - obj_val_bab) / obj_val_bab * 100
                        gap_nodisc = (obj_val_nodisc - obj_val_bab) / obj_val_bab * 100
                        gap_uniform = (obj_val_uniform - obj_val_bab) / obj_val_bab * 100
                        data.append(
                            [instance, nrCust, nrPup, p_home, p_pup, discount, policy_ID_bab, num_offered_disc_bab, obj_val_bab, sd_bab, time_bab,
                             nodes,
                             num_tsps,
                             policy_ID_rs, obj_val_rs, sd_rs, gap_rs, obj_val_nodisc,  gap_nodisc,
                             obj_val_uniform, sd_uniform, gap_uniform, eps, sample, time_limit])
                    except:

                        data.append(
                            [instance, nrCust, nrPup, p_home, p_pup, discount, policy_ID_bab, obj_val_bab, sd_bab, time_bab, nodes,
                             num_tsps,
                             "", "", "", "", "", "", "",
                             "", "", "", eps, sample, time_limit])
            #else:
            except:
                #data.append([instance, nrCust, p_home, p_pup, discount])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['instance', 'nrCust', 'nrPup', 'p_home', 'p_pup', 'discount_rate', 'policy_bab_ID', 'num_disc_bab','obj_val_bab', '2sd_bab',
                'time_bab', 'nodes',
                'num_tsps', 'policy_ID_rs', 'obj_val_rs', '2sd_rs', 'gap_rs', 'obj_val_nodisc',
                'gap_nodisc', 'obj_val_uniform', '2sd_uniform', 'gap_uniform', 'eps','sample','time_limit']
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def parseEnumeration(file_path, folder, output_name):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                # if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])

                    time_calculate_all_TSPs = float(lines[idx + 4].split(':')[1])
                    obj_val = float(lines[idx + 5].split(':')[1])
                    time_running = float(lines[idx + 8].split()[1])
                    policy_ID = int(lines[idx + 6].split(':')[1])
                    data.append([nrCust, nrPup, time_running,time_calculate_all_TSPs, obj_val, policy_ID, instance])
                # if 'time_calculation_all_TSPs:' in line:
                #     instance =  ''
                #     nrCust = ''
                #     nrPup = 1
                #
                #     time_calculate_all_TSPs = float(lines[idx + 0].split(':')[1])
                #     obj_val = float(lines[idx + 1].split(':')[1])
                #     time_running = float(lines[idx + 4].split()[1])
                #     policy_ID = int(lines[idx + 2].split(':')[1])
                #     data.append([nrCust, nrPup, time_running,time_calculate_all_TSPs, obj_val, policy_ID, instance])


            except:
                data.append(["", "", "", "", instance])
                print("enumeration problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['nrCust_enum','nrPup_enum', 'time_running_enum','time_calc_allTSPs', 'obj_val_enum', 'policy_enum_ID', 'instance']
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def sensitivity_disc_size_comparison_nodisc(folder, folder_data):
    # parseBAB_RS_NODISC(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_disc.txt"), folder, "bab_rs_nodisc_i_VRPDO_disc")
    df = pd.read_csv(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_disc.csv"))
    df['instance_type'] = df['instance'].apply(lambda x: str(x).split('_')[0])
    # df.rename(columns={'discount_rate': 'discountRate'}, inplace=True)

    df['expDiscountCostBab'] = df.apply(
        lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data), axis=1)
    df['expDiscountCostRs'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_ID_rs'], x['instance'], folder_data),
                                       axis=1)
    df['expDiscountCostUniform'] = df.apply(lambda x: calculate_exp_disc_cost(32767, x['instance'], folder_data),
                                            axis=1)

    df['p_accept'] = round(1 - df['p_pup'] - df['p_home'], 2)
    df['in_pup_bab'] = df.apply(lambda x: calculate_exp_pup_utilization(x['policy_bab_ID'], x['instance'], folder_data),
                                axis=1)
    df['in_pup_rs'] = df.apply(lambda x: calculate_exp_pup_utilization(x['policy_ID_rs'], x['instance'], folder_data),
                               axis=1)
    df['in_pup_uniform'] = df.apply(lambda x: calculate_exp_pup_utilization(32767, x['instance'], folder_data), axis=1)

    # df['exp_discount_cost_bab'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance']), axis=1)
    # df['exp_routing_cost_bab'] = df['obj_val_bab'] - df['exp_discount_cost_bab']
    #
    # df['exp_discount_cost_bab'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance']), axis=1)
    # df['exp_routing_cost_bab'] = df['obj_val_bab'] - df['exp_discount_cost_bab']

    df['nodisc_bab, percent'] = round((df['obj_val_nodisc'] - df['obj_val_bab']) / df['obj_val_nodisc'] * 100, 1)
    df['nodisc_rs, percent'] = round((df['obj_val_nodisc'] - df['obj_val_rs']) / df['obj_val_nodisc'] * 100, 1)
    df['nodisc_uniform, percent'] = round((df['obj_val_nodisc'] - df['obj_val_uniform']) / df['obj_val_nodisc'] * 100,
                                          1)

    df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)
    df['num_offered_disc_bab_percent'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID']) / 15 * 100), axis=1)

    df['num_offered_disc_rs'] = df.apply(lambda x: round(bitCount(x['policy_ID_rs'])), axis=1)
    # df['num_offered_disc_uniform'] = 100

    for instance_type in df.instance_type.unique():
        # print("instance_type", instance_type)
        # df1 = df[df.instance_type == instance_type].copy()
        df1 = df.copy()
        discount_rates = []
        bab_cost = []
        rs_cost = []
        all_cost = []
        noi_cost = []
        discounts = df1.discount_rate.unique().tolist()
        discounts.sort()
        # discounts = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,  1.2]

        df_results = pd.DataFrame(index=discounts,
                                  columns=['SDSCost', 'imprNOI', 'imprNOImin', 'imprNOImax', 'imprNOI-sd', 'imprDDS',
                                           'imprDDSmin,', 'imprDDSmax', 'imprDDS-sd', 'imprALL', 'imprALLmin',
                                           'imprALLmax', 'imprALL-sd'])
        for d in discounts:
            df_slice = df[(df.discount_rate == d)].copy()

            df_results.at[d, 'SDSCost'] = round(df_slice['obj_val_bab'].mean(), 1)
            df_results.at[d, 'imprNOI'] = round(df_slice['gap_nodisc'].mean(), 1)
            df_results.at[d, 'imprNOImin'] = round(df_slice['gap_nodisc'].min(), 1)
            df_results.at[d, 'imprNOImax'] = round(df_slice['gap_nodisc'].max(), 1)

            df_results.at[d, 'imprNOI-sd'] = round(df_slice['gap_nodisc'].std(), 1)
            df_results.at[d, 'imprDDS'] = round(df_slice['gap_rs'].mean(), 1)
            df_results.at[d, 'imprDDSmin'] = round(df_slice['gap_rs'].min(), 1)
            df_results.at[d, 'imprDDSmax'] = round(df_slice['gap_rs'].max(), 1)

            df_results.at[d, 'imprDDS-sd'] = round(df_slice['gap_rs'].std(), 1)
            df_results.at[d, 'imprALL'] = round(df_slice['gap_uniform'].mean(), 1)
            df_results.at[d, 'imprALLmin'] = round(df_slice['gap_uniform'].min(), 1)
            df_results.at[d, 'imprALLmax'] = round(df_slice['gap_uniform'].max(), 1)
            df_results.at[d, 'imprALL-sd'] = round(df_slice['gap_uniform'].std(), 1)

        # df_results2 = df_results[ ['SDSCost', 'imprNOI', 'imprNOI-sd', 'imprDDS','imprDDS-sd','imprALL','imprALL-sd']].copy()
        df_results2 = df_results[
            ['SDSCost', 'imprNOI', 'imprNOImin', 'imprNOImax', 'imprDDS', 'imprDDSmin', 'imprDDSmax', 'imprALL',
             'imprALLmin', 'imprALLmax']].copy()

        print(df_results2.to_latex(float_format='{:0.1f}'.format, na_rep=''))
        print("")

        for d in discounts:
            discount_rates.append(d)
            bab_cost.append(round(df1.loc[(df1.discount_rate == d), 'obj_val_bab'].mean(), 1))
            bab_cost.append('')
            bab_cost.append('')
            bab_cost.append('')
            rs_cost.append(round(df1.loc[(df1.discount_rate == d), 'obj_val_rs'].mean(), 1))
            rs_cost.append(-int(round(df1.loc[(df1['discount_rate'] == d), 'gap_rs'].mean(), 0)))
            rs_cost.append(int(round(df1.loc[(df1['discount_rate'] == d), 'gap_rs'].std(), 0)))

            all_cost.append(round(df1.loc[(df1.discount_rate == d), 'obj_val_uniform'].mean(), 1))
            all_cost.append(-int(round(df1.loc[(df1['discount_rate'] == d), 'gap_uniform'].mean(), 0)))
            all_cost.append(int(round(df1.loc[(df1['discount_rate'] == d), 'gap_uniform'].std(), 0)))

            noi_cost.append(round(df1.loc[(df1.discount_rate == d), 'obj_val_nodisc'].mean(), 1))
            noi_cost.append(-int(round(df1.loc[(df1['discount_rate'] == d), 'gap_nodisc'].mean(), 0)))
            noi_cost.append(int(round(df1.loc[(df1['discount_rate'] == d), 'gap_nodisc'].std(), 0)))
            # bab_savings.append(int(round(df1.loc[(df1['discount_rate'] == d) , 'num_offered_disc_bab'].mean(), 0)))
            # bab_savings.append(int(round(df1.loc[(df1['discount_rate'] == d), 'exp_discount_cost_bab, percent'].mean(), 0)))
            # rs_saving.append(round(df1.loc[(df1.discount_rate == d),'nodisc_rs, percent'].mean(),1))
            # rs_saving.append(int(round(df1.loc[(df1['discount_rate'] == d), 'num_offered_disc_rs'].mean(), 0)))
            # rs_saving.append(int(round(df1.loc[(df1['discount_rate'] == d), 'exp_discount_cost_rs, percent'].mean(), 0)))

            # uniform_savings.append(round(df1.loc[(df1.discount_rate == d),'nodisc_uniform, percent'].mean(),1))
            # uniform_savings.append(int(round(df1.loc[(df1['discount_rate'] == d), 'num_offered_disc_uniform'].mean(), 0)))
            # uniform_savings.append(int(round(df1.loc[(df1['discount_rate'] == d), 'exp_discount_cost_uniform, percent'].mean(), 0)))

        print("$u$ &", '&& '.join(map(str, discount_rates)), "\\")

        print("S-DS&", ' & '.join(map(str, bab_cost)), "\\")
        print("NOI &", ' & '.join(map(str, noi_cost)), "\\")
        print("D-DS &", ' & '.join(map(str, rs_cost)), "\\")
        print("ALL &", ' & '.join(map(str, all_cost)), "\\")
    df_temp = df.copy()

    # sns.set()
    # sns.set(font_scale=1.2)
    # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    #
    # fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    #
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='num_offered_disc_bab', linewidth=1, marker='o',
    #              markersize=6, color='b', err_style="bars", ci=68, err_kws={'capsize': 3}, label = 'S-DS', legend = False)
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='num_offered_disc_rs', linewidth=1, marker='o',
    #              markersize=6, color='r',err_style="bars", ci=68, err_kws={'capsize': 3}, label = 'D-DS', legend = False)
    #
    # axes.legend()
    # axes.set(xlabel='Incentive value coefficient, ' + r'$u$')
    # axes.set(ylabel='Number of offered incentives')
    # plt.savefig(os.path.join(path_to_images, 'Incentive_size_number_discounts.eps'), transparent=False, bbox_inches='tight')
    # plt.show()

    sns.set()
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='gap_rs', linewidth=1, marker='o',
                 markersize=6, color='b', err_style="bars", ci=None, err_kws={'capsize': 3}, label='S-DS, number',
                 legend=False)
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='obj_val_rs', linewidth=1, marker='o',
    #             markersize=6, color='r', err_style="bars", ci=None, err_kws={'capsize': 3},  label = 'D-DS, number', legend = False)

    ax2 = axes.twinx()
    sns.lineplot(ax=ax2, data=df_temp, x='discount_rate', y='num_offered_disc_bab', linewidth=1, marker='o',
                 markersize=6, color='b', err_style="bars", ci=None, err_kws={'capsize': 3}, label='S-DS, cost',
                 legend=False)
    sns.lineplot(ax=ax2, data=df_temp, x='discount_rate', y='num_offered_disc_rs', linewidth=1, marker='o',
                 markersize=6, color='r', err_style="bars", ci=None, err_kws={'capsize': 3}, label='D-DS, cost',
                 legend=False)

    ax2.lines[0].set_linestyle("--")
    ax2.lines[1].set_linestyle("--")

    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    axes.set(xlabel='Incentive rate, ' + r'$u$')
    axes.set(ylabel=r'Cost of incentives')
    ax2.set(ylabel=r'Number of offered incentives')
    plt.savefig(os.path.join(path_to_images, 'Incentive_size_number_discounts.eps'), transparent=False,
                bbox_inches='tight')
    plt.show()

    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{mathptmx}']  # load times roman font
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"]})
    #
    # sns.set()
    # sns.set(font_scale=1.2)
    # sns.set_style("whitegrid", {'axes.grid' : False, 'lines.linewidth': 0.2})
    # sns.set_style('ticks',{"xtick.direction": "in","ytick.direction": "in"})
    #
    # fig, axes = plt.subplots(1, 1,  figsize=(6, 5))
    # sns.lineplot(ax=axes, data=df_temp, x='discountRate', y='expDiscountCostBab', linewidth = 1, marker='o', markersize=8,color='b', label = 'S-DS', err_style="bars", ci=68, err_kws={'capsize':3})
    # sns.lineplot(ax=axes, data=df_temp, x='discountRate', y='expDiscountCostRs',linewidth = 1, marker='^', markersize=8, color='r',  label = 'D-DS', err_style="bars", ci=68, err_kws={'capsize':3})
    # axes.set(xlabel='Incentive rate, ' + r'$u$')
    #
    # axes.set(ylabel=r'Expected incentive cost')
    # plt.savefig(os.path.join(path_to_images,'Incentive_size.eps'), transparent=False, bbox_inches='tight')
    # plt.show()
    sns.set()
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # df_temp2 = df_temp[df_temp['discount_rate'].isin([0.05, 0.2,0.4,0.6,0.8, 1.0, 1.2, 1.4])].copy()
    # df_temp3 = df_temp2[['expDiscountCostBab','obj_val_bab' ]].copy()
    sns.lineplot(ax=ax, data=df_temp, x='discount_rate', y='obj_val_bab', linewidth=1, marker='o',
                 markersize=6, color='black', err_style="bars", err_kws={'capsize': 3}, label='Fulfillment cost')
    sns.lineplot(ax=ax, data=df_temp, x='discount_rate', y='expDiscountCostBab', linewidth=1, marker='^',
                 markersize=8, color='green', err_style="bars", err_kws={'capsize': 3}, label='Incentive cost')
    # bar2 = sns.barplot(data = df_temp2,  x= 'discount_rate', y='obj_val_bab',  ci=None, color='darkblue')
    # bar1 = sns.barplot(data=df_temp2, x='discount_rate', y='expDiscountCostBab', ci=None, color='lightblue')
    ax.set(xlabel='Incentive rate, ' + r'$u$')
    ax.set(ylabel=r'Cost')

    ax2 = ax.twinx()
    sns.lineplot(ax=ax2, data=df_temp, x='discount_rate', y='num_offered_disc_bab_percent', linewidth=1, marker='s',
                 markersize=6, color='darkblue', err_style="bars", err_kws={'capsize': 3}, label='% offered incentives',
                 legend=False)
    ax2.lines[0].set_linestyle("--")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.set_ylim(-1, 80)
    ax2.set(ylabel='% offered incentives')
    ax.legend(lines + lines2, labels + labels2)
    plt.savefig(os.path.join(path_to_images, 'Fulfillment_incentive_cost_discount.eps'), transparent=False,
                bbox_inches='tight')
    plt.show()


    fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(ax=ax2, data=df_temp, x='discount_rate', y='gap_nodisc', marker='o', color='b', markersize=6,
                 label='NOI', err_style="bars", linewidth=1, ci=68, err_kws={'capsize': 3})
    sns.lineplot(ax=ax2, data=df_temp, x='discount_rate', y='gap_rs', marker='^', color='r', markersize=6,
                 label='D-DS', err_style="bars", linewidth=1, ci=68, err_kws={'capsize': 3})
    sns.lineplot(ax=ax2, data=df_temp, x='discount_rate', y='gap_uniform', marker='s', color='g', markersize=6,
                 label='ALL', err_style="bars", linewidth=1, ci=68, err_kws={'capsize': 3})

    axes = ax2.twinx()
    sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='obj_val_bab', linewidth=1, marker='o',
                 markersize=6, color='black', err_style="bars", ci=None, err_kws={'capsize': 3}, label='S-DS Cost',
                 legend=False)
    axes.lines[0].set_linestyle("--")
    sns.set()
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})

    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95), )
    ax2.set_ylim(0, 30)
    axes.set(xlabel='Incentive rate, ' + r'$u$')
    axes.set(ylabel=r'Fulfillment cost')
    ax2.set(ylabel=r'Improvement (%)')
    plt.savefig(os.path.join(path_to_images, 'Improvement_number_disc_disc_rate.eps'), transparent=False,
                bbox_inches='tight')
    plt.show()

    sns.set()
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='gap_nodisc', marker='o', color='b', markersize=6,
                 label='NOI', err_style="bars", linewidth=1, ci=68, err_kws={'capsize': 3})
    sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='gap_rs', marker='^', color='r', markersize=6,
                 label='D-DS', err_style="bars", linewidth=1, ci=68, err_kws={'capsize': 3})
    sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='gap_uniform', marker='s', color='g', markersize=6,
                 label='ALL', err_style="bars", linewidth=1, ci=68, err_kws={'capsize': 3})

    axes.set(xlabel='Incentive value coefficient, ' + r'$u$')
    axes.set(ylabel='Improvement (%)')
    axes.set_ylim(0, 30)
    # axes.set_xlim(0, 1.5)
    # sns.despine()
    plt.savefig(os.path.join(path_to_images, 'Improvement_disc.eps'), transparent=False, bbox_inches='tight')
    plt.show()

    # sns.set()
    # sns.set(font_scale=1.2)
    # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    # fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='obj_val_nodisc', marker='o', palette="deep", markersize=6,
    #              color='black',
    #              label='NOI', err_style="bars", linewidth = 1,ci=68, err_kws={'capsize': 3})
    # axes.lines[0].set_linestyle("--")
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='obj_val_bab', marker='o', color='b', markersize=6,
    #              label='S-DS', err_style="bars",linewidth = 1,ci=68, err_kws={'capsize': 3})
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='obj_val_rs', marker='^', color='r', markersize=6,
    #              label='D-DS', err_style="bars",linewidth = 1, ci=68, err_kws={'capsize': 3})
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='obj_val_uniform', marker='s', color='g', markersize=6,
    #              label='ALL', err_style="bars", linewidth = 1,ci=68, err_kws={'capsize': 3})
    #
    # axes.set(xlabel='Incentive value coefficient, ' + r'$u$')
    # axes.set(ylabel='Fulfillment cost')
    # axes.set_ylim(40, 80)
    # # axes.set_xlim(0, 1.5)
    # # sns.despine()
    # plt.savefig(os.path.join(path_to_images, 'Total_expected_cost.eps'), transparent=False, bbox_inches='tight')
    # plt.show()

    # axes[0].set(xlabel='Incentive rate, ' + r'$u_{spd}$')
    # axes[1].set(xlabel='Incentive rate, ' + r'$u_{spd}$')
    # axes[2].set(xlabel='Incentive rate, ' + r'$u_{spd}$')
    # axes[0].set(ylabel='Savings with respect to BM (%)')
    # axes[1].set(ylabel='Incentive cost as percentage of total cost (%)')
    # axes[2].set(ylabel='Total expected delivery cost')
    # axes[0].set_ylim(-10,None )
    # axes[1].set_ylim(None,40 )
    # axes[2].set_ylim(None, 75)
    # df['exp_discount_cost_bab'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_bab_ID'] , x['instance'] ), axis = 1)
    # df['exp_discount_cost_rs'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_ID_rs'], x['instance']), axis=1)
    #
    # sns.set()
    # sns.set_style("whitegrid")
    # fig, axes = plt.subplots(1, 1, sharex=True, figsize=(5, 5))
    # # df_temp = df[(df.instance_type == 'solomonRC100') ].copy()
    # df_temp = df.copy()
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='exp_discount_cost_bab', marker='o', color='b',
    #              err_style="bars", ci=68)
    # sns.lineplot(ax=axes, data=df_temp, x='discount_rate', y='exp_discount_cost_rs', marker='o', color='r',
    #              err_style="bars", ci=68)
    # plt.show()


# def sensitivity_prob_comparison_nodisc_old(folder, folder_data):
#     parseBAB_RS_NODISC(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_prob.txt"), folder, "bab_rs_nodisc_i_VRPDO_prob")
#     df = pd.read_csv(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_prob.csv"))
#     df['instance_type'] = df['instance'].apply(lambda x: str(x).split('_')[0])
#     df['in_pup_bab'] = df.apply(lambda x: calculate_exp_pup_utilization(x['policy_bab_ID'], x['instance'], folder_data),
#                                 axis=1)
#
#     df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID']) ), axis=1)
#     df['num_offered_disc_rs'] = df.apply(lambda x: round(bitCount(x['policy_ID_rs']) / 15 * 100), axis=1)
#     df['num_offered_disc_uniform'] = 100
#
#     df['in_pup_rs'] = df.apply(lambda x: calculate_exp_pup_utilization(x['policy_ID_rs'], x['instance'], folder_data),
#                                axis=1)
#     df['in_pup_uniform'] = df.apply(lambda x: calculate_exp_pup_utilization(32767, x['instance'], folder_data), axis=1)
#     df['p_accept'] = round(1 - df['p_pup'] - df['p_home'], 2)
#
#     df['exp_discount_cost_bab'] = df.apply(
#         lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data), axis=1)
#     df['upper_bound'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data),
#                                  axis=1)
#     df['exp_discount_cost_bab_percent'] = (df['exp_discount_cost_bab'] / df['obj_val_bab']) * 100
#
#     df['exp_discount_cost_rs'] = df.apply(
#         lambda x: calculate_exp_disc_cost(x['policy_ID_rs'], x['instance'], folder_data), axis=1)
#     df['exp_discount_cost_rs_percent'] = (df['exp_discount_cost_rs'] / df['obj_val_rs']) * 100
#
#     df['exp_discount_cost_uniform'] = df.apply(lambda x: calculate_exp_disc_cost(32767, x['instance'], folder_data),
#                                                axis=1)
#     df['exp_discount_cost_uniform_percent'] = (df['exp_discount_cost_uniform'] / df['obj_val_uniform']) * 100
#
#     df['nodisc_bab, %'] = round((df['obj_val_nodisc'] - df['obj_val_bab']) / df['obj_val_bab'] * 100, 1)
#     df['nodisc_rs, %'] = round((df['obj_val_nodisc'] - df['obj_val_rs']) / df['obj_val_nodisc'] * 100, 1)
#     df['nodisc_uniform, %'] = round((df['obj_val_nodisc'] - df['obj_val_uniform']) / df['obj_val_nodisc'] * 100, 1)
#     df['rs_bab_comp,%'] = df.apply(lambda x: (x['obj_val_rs'] - x['obj_val_bab']) / x['obj_val_bab'] * 100, axis=1)
#     df['all_bab_comp,%'] = df.apply(lambda x: (x['obj_val_uniform'] - x['obj_val_bab']) / x['obj_val_bab'] * 100,
#                                     axis=1)
#
#     # loss of using the best policy among d-ds, all, BM
#     df['loss_best_comp,%'] = df.apply(
#         lambda x: min(x['nodisc_bab, %'], (x['obj_val_rs'] - x['obj_val_bab']) / x['obj_val_rs'] * 100,
#                       (x['obj_val_nodisc'] - x['obj_val_bab']) / x['obj_val_nodisc'] * 100), axis=1)
#     for instance_type in df.instance_type.unique():
#         print("instance_type", instance_type)
#         df1 = df[df.instance_type == instance_type].copy()
#         discount_rates = []
#
#         rs_saving = []
#         uniform_savings = []
#         # key(p_pup):p_delta
#         dict_probabilities = {0.0: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
#                               0.2: [0.0, 0.2, 0.4, 0.6, 0.8],
#                               0.4: [0.0, 0.2, 0.4, 0.6],
#                               0.6: [0.0, 0.2, 0.4],
#                               0.8: [0.0, 0.2],
#                               1.0: [0.0]}
#
#         # upper and lower bounds:
#         df['lower_bound'] = df.apply(lambda x: calculate_lower_bounds(x['instance'], folder_data), axis=1)
#         df['upper_bound'] = df.apply(lambda x: calculate_upper_bounds(x['instance'], folder_data), axis=1)
#         df['improvement'] = df.apply(lambda x: 100*(x['upper_bound']-x['obj_val_bab'])/x['upper_bound'], axis=1)
#
#         # df_temp = df[
#         #     (df.p_pup == 0.0) | (df.p_pup == 0.2) | (df.p_pup == 0.4) | (df.p_pup == 0.6) | (df.p_pup == 0.8)   ].copy()
#         # sns.set()
#         # sns.set(font_scale=1.2)
#         # #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         #
#         # x = [ 0.2, 0.4, 0.6, 0.8, 1]
#         # cmap = sns.color_palette("deep")
#         # m = np.array(['o', 'P', 's', '^', 'D',"+"])
#         # lns = np.array(['-', '--', '-.', ':', 'dashdot'])
#         # change_x = [-0.04, -0.02, 0, 0.02,0.04]
#         # labels = [r'$p = 0.8$', r'$p = 0.6$', r'$p = 0.4$',  r'$p = 0.2$',  r'$p = 0.0$',]
#         # p_pups =  [0.8, 0.6, 0.4, 0.2, 0.0]
#         # for iter in [0,1,2,3,4]:
#         #     df_temp2 = df_temp[df_temp['p_pup'] == p_pups[iter]].copy()
#         #     y = []
#         #     yerr = []
#         #     ymin = []
#         #     ymax = []
#         #     x_print = []
#         #     for delta in x:
#         #         df_temp3 = df_temp2[df_temp2.p_accept == delta].copy()
#         #         x_print.append(delta + change_x[iter])
#         #         y.append(df_temp3['obj_val_bab'].mean())
#         #         ymin.append(df_temp3['obj_val_bab'].min())
#         #         ymax.append(df_temp3['obj_val_bab'].max())
#         #
#         #         yerr.append([df_temp3['obj_val_bab'].mean() - df_temp3['obj_val_bab'].min(),
#         #                      df_temp3['obj_val_bab'].max() - df_temp3['obj_val_bab'].mean()])
#         #
#         #     yerr = np.transpose(yerr)
#         #     # plt.scatter(x_print, y_min,  marker = '-', s= 16,  label = class_name[class_id-1])
#         #     # plt.errorbar(x_print, y_min, yerr=None, marker = m[class_id-1],  ms=3, linestyle = '', mec = cmap[class_id-1],mfc= cmap[class_id-1], c = cmap[class_id-1],)
#         #
#         #     plt.errorbar(x_print, y, yerr=yerr, marker=m[iter], ms=7, linewidth=0.5,
#         #                  label=labels[iter],
#         #                  mec=cmap[iter], mfc=cmap[iter], c=cmap[iter], elinewidth=1, capsize=3)
#
#
#         #sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='obj_val_bab', hue='p_pup', style="p_pup", markers=True,
#         #             markersize=9, linewidth=1,
#         #             palette="deep", err_style="bars", ci=100, err_kws={'capsize': 3}, dashes=False)
#
#
#
#         # file_path = os.path.join(folder, "bab_rs_nodisc_i_VRPDO_prob_bounds.xlsx")
#         # writer = pd.ExcelWriter(file_path, engine='openpyxl')
#         # df.to_excel(writer, index=False)
#         # writer.save()
#         #axes.fill_between(x, df['upper_bound'].min(), df['upper_bound'].max(), alpha=0.2)
#         #axes.axhline(df['upper_bound'].mean(), ls='--')
#         # lower bound
#         #axes.axhline(df['lower_bound'].mean(), ls='--')
#
#         # axes.set(xlabel='' + r'$\Delta$')
#         # axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
#         # axes.set(ylabel='Expected fulfillment cost')
#         # plt.legend(title=False, loc='lower right',ncol=2, bbox_to_anchor=(0.99, 0.01))
#         # #plt.legend(title=False, loc='lower right', bbox_to_anchor=(0.99, 0.1),
#         # #           labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$', r'$p = 0.8$', r'$p = 1.0$'])
#         # plt.savefig(os.path.join(path_to_images, 'Total_cost_hue_ppup_bounds.eps'), transparent=False,
#         #             bbox_inches='tight')
#         # plt.show()
#
#
#         # sns.set()
#         # sns.set(font_scale=1.1)
#         # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='exp_discount_cost_bab', hue='p_accept', style="p_accept", linewidth=1,markers=True, markersize=7,
#         #             palette="deep",  err_style="bars", ci=68, err_kws={'capsize': 3})
#         # axes.set(xlabel='' + r'$p$')
#         # axes.set(ylabel='Expected incentive cost')
#         # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$'])
#         # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$',
#         #                                 r'$\Delta = 1.0$'])
#         # plt.savefig(os.path.join(path_to_images, 'Incentive_cost_hue_ppup.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#         #
#
#         # sns.set()
#         # df_temp = df[(df.p_pup == 0.0)].copy()
#         # sns.set(font_scale=1.3)
#         # #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         #
#         # cmap = sns.color_palette("deep")
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # x = [ 0.2, 0.4, 0.6, 0.8, 1]
#         # y=[]
#         # yerr =[]
#         # for p_accept in x:
#         #     df_temp2 = df_temp[df_temp.p_accept == p_accept].copy()
#         #     y.append(df_temp2['num_offered_disc_bab'].mean())
#         #     yerr.append([df_temp2['num_offered_disc_bab'].mean() - df_temp2['num_offered_disc_bab'].min(),
#         #                                        df_temp2['num_offered_disc_bab'].max() - df_temp2['num_offered_disc_bab'].mean()])
#         # yerr = np.transpose(yerr)
#         # plt.errorbar(x, y, yerr=yerr, marker='D', ms=7, capsize=3,mec=cmap[4], mfc=cmap[4], c=cmap[4] )
#         #
#         # # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='num_offered_disc_bab',
#         # #              linewidth=1, markersize=7, markers=True, marker='o',
#         # #              palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # axes.set(xlabel='' + r'$\Delta$')
#         # axes.set(ylabel='Number of offered incentives')
#         # # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$', r'$\Delta = 1.0$'])
#         # plt.savefig(os.path.join(path_to_images, 'Number_incentives_delta.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#
#
#
#
#         # fontsize = 15
#         # sns.set(font_scale=1.4)
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # sns.color_palette("Spectral", as_cmap=True)
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='rs_bab_comp,%',markers=True,
#         #              markersize=8,linewidth=1,palette="deep", err_style="bars", ci=96, err_kws={'capsize': 3})
#         # #sns.relplot(data=df_temp, x='p_pup', y='p_accept',hue='gap_nodisc_mean', s=100, palette="Spectral")
#         # plt.show()
#
#         # fontsize = 15
#         # sns.set(font_scale=1.4)
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='nodisc_bab, %', hue='p_home',  style="p_home",markers=True, markersize=8,
#         #              linewidth=1,palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # plt.show()
#         # sns.set(font_scale=1.4)
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.directionp_accep": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='rs_bab_comp,%', hue='p_home', style="p_home", markers=True,
#         #              markersize=8,
#         #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # plt.show()
#         # sns.set(font_scale=1.4)
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='all_bab_comp,%', hue='p_home', style="p_home", markers=True,
#         #              markersize=8,
#         #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # plt.show()
#         #
#
#         # fontsize = 15
#         # sns.set(font_scale=1.4)
#         # df_temp = df[(df.p_pup == 0.0)| (df.p_pup==0.4) | (df.p_pup ==0.8)].copy()
#         # df_temp['1p_accept'] = 1-df_temp['p_accept']
#         #
#         # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='nodisc_bab, %', hue='p_home', style='p_home',
#         #              markers=True, markersize=8, linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # axes.set(xlabel=r'$\Delta$')
#         # axes.set(ylabel='Improvement over NOI (%)')
#         # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.4$', r'$p = 0.8$'])
#         # plt.savefig(os.path.join(path_to_images, 'Bab_nodisc_comparison.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#         #
#         # sns.set(font_scale=1.4)
#         # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='rs_bab_comp,%',hue='p_home', style='p_home',
#         #              markers=True, markersize=8,
#         #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.4$', r'$p = 0.8$'])
#         # axes.set(xlabel=r'$\Delta$')
#         # axes.set(ylabel='Improvement over D-DS (%)')
#         # plt.savefig(os.path.join(path_to_images, 'Bab_rs_comparison.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#         #
#         # sns.set(font_scale=1.4)
#         # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='all_bab_comp,%',  markers=True, markersize=8,hue='p_home', style='p_home',
#         #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 4})
#         # axes.set(xlabel=r'$\Delta$')
#         # axes.set(ylabel='Improvement over ALL (%)')
#         # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.4$', r'$p = 0.8$'])
#         # plt.savefig(os.path.join(path_to_images, 'Bab_all_comparison.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#
#
#         fig, axes = plt.subplots(1, 4, sharey=True, figsize=(19, 5), gridspec_kw={'width_ratios': [10, 8, 6, 4]})
#         sns.set(font_scale=1.4)
#
#         rc = {'font.sans-serif': 'Computer Modern Sans Serif'}
#         sns.set_context(rc=rc)
#         sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         plt.rcParams.update(**rc)
#         cmap = sns.color_palette("deep")
#         m = np.array(['o', '^', 'D', '^', 'D', "+"])
#
#         iter = -1
#         for p_pup_param in [0.0, 0.2, 0.4, 0.6]:
#             iter += 1
#             df_temp = df[df.p_pup == p_pup_param].copy()
#
#             x = [0.2, 0.4, 0.6, 0.8, 1.0]
#             if iter == 1:
#                 x = [0.2, 0.4, 0.6, 0.8]
#                 #axes[iter].set_xlim(None, 0.82)
#             if iter == 2:
#                 x = [0.2, 0.4, 0.6]
#                 #axes[iter].set_xlim(None, 0.62)
#             if iter == 3:
#                 x = [ 0.2, 0.4]
#                 #axes[iter].set_xlim(0.18, 0.45)
#                 #axes[iter].set_xticks([0.0, 0.2, 0.4])
#             y = []
#             yerr = []
#
#             y_d = []
#             yerr_d = []
#
#             y_noi = []
#             yerr_noi = []
#
#             x_all = [temp-0.02 for temp in x]
#             x_noi = [temp+0.02 for temp in x]
#
#             for p_accept in x:
#                 df_temp2 = df_temp[df_temp.p_accept == p_accept].copy()
#                 y.append(df_temp2['all_bab_comp,%'].mean())
#                 yerr.append([df_temp2['all_bab_comp,%'].mean() - df_temp2['all_bab_comp,%'].min(),
#                                                    df_temp2['all_bab_comp,%'].max() - df_temp2['all_bab_comp,%'].mean()])
#
#                 y_d.append(df_temp2['rs_bab_comp,%'].mean())
#                 yerr_d.append([df_temp2['rs_bab_comp,%'].mean() - df_temp2['rs_bab_comp,%'].min(),
#                              df_temp2['rs_bab_comp,%'].max() - df_temp2['rs_bab_comp,%'].mean()])
#
#                 y_noi.append(df_temp2['nodisc_bab, %'].mean())
#                 yerr_noi.append([df_temp2['nodisc_bab, %'].mean() - df_temp2['nodisc_bab, %'].min(),
#                                df_temp2['nodisc_bab, %'].max() - df_temp2['nodisc_bab, %'].mean()])
#
#             yerr = np.transpose(yerr)
#             yerr_d = np.transpose(yerr_d)
#             yerr_noi = np.transpose(yerr_noi)
#
#             axes[iter].errorbar(x_all, y, yerr=yerr, marker=m[2], ms=7, linewidth=0.3,
#                          label="ALL", mec=cmap[2], mfc=cmap[2], c=cmap[2], elinewidth=1.5, capsize=4)
#
#             axes[iter].errorbar(x_noi, y_noi, yerr=yerr_noi, marker=m[0], ms=7, linewidth=0.3,
#                          label="NOI", mec=cmap[0], mfc=cmap[0], c=cmap[0], elinewidth=1.5, capsize=4)
#
#             axes[iter].errorbar(x, y_d, yerr=yerr_d, marker=m[1], ms=7, linewidth=0.3,
#                          label="DI", mec=cmap[1], mfc=cmap[1], c=cmap[1], elinewidth=1.5, capsize=4)
#
#             axes[iter].set(xlabel='Problem size, n')
#
#             # sns.lineplot(ax=axes[iter], data=df_temp, x='p_accept', y='all_bab_comp,%', marker="s", markersize=7,
#             #              linewidth=1,
#             #              palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3}, label='ALL', legend=0)
#             # sns.lineplot(ax=axes[iter], data=df_temp, x='p_accept', y='rs_bab_comp,%', marker="o", markersize=8,
#             #              linewidth=1,
#             #              palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3}, label='DI', legend=0)
#             # sns.lineplot(ax=axes[iter], data=df_temp, x='p_accept', y='nodisc_bab, %',  marker="^", markersize=9,
#             linewidth=1,
#             #             palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3}, label = 'NOI', legend=0)
#             # plt.legend(title=False,loc = 'upper right', bbox_to_anchor=(0.995, 0.995))
#
#             axes[iter].set_title(r'$p =$' + str(p_pup_param))
#             # if iter == 1:
#             #     axes[iter].set_xlim(None, 0.82)
#             # if iter == 2:
#             #     axes[iter].set_xlim(None, 0.62)
#             # if iter == 3:
#             #     axes[iter].set_xlim(None, 0.45)
#             axes[iter].set_xlabel(r'$\Delta$', fontsize=18)
#
#         handles, labels = axes[iter].get_legend_handles_labels()
#
#         axes[0].set_ylabel('Savings (%)', fontsize = 19)
#         axes[0].legend(handles, labels,loc = 'upper left', bbox_to_anchor=(0.01, 0.99),  title=False)
#
#         plt.savefig(os.path.join(path_to_images, 'Improvement_threparcee_plots.eps'), transparent=False,
#                     bbox_inches='tight')
#         plt.show()
#         #
#         # sns.set(font_scale=1.4)
#         # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='rs_bab_comp,%',
#         #              markers=True, markersize=8,
#         #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
#         # #axes.set(xlabel=r'$p')
#         # axes.set(ylabel='Improvement over D-DS (%)')
#         # plt.savefig(os.path.join(path_to_images, 'Bab_rs_comparison.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#         #
#         # sns.set(font_scale=1.4)
#         # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#         # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
#         # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
#         # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
#         # fig, axes = plt.subplots(1, 1, sharex=True)
#         # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='all_bab_comp,%', markers=True, markersize=8,
#         #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 4})
#         # #axes.set(xlabel=r'$p$')
#         # axes.set(ylabel='Improvement over ALL (%)')
#         # plt.savefig(os.path.join(path_to_images, 'Bab_all_comparison.eps'), transparent=False, bbox_inches='tight')
#         # plt.show()
#
#         # print("bab")
#         # for p_pup in dict_probability_print:
#         #     bab_savings = []
#         #     for p_delta in dict_probability_print[p_pup]:
#         #         #bab_savings.append( str(int(round(df1.loc[(df1['p_pup'] == p_pup) & (df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_bab'].mean(),0))))
#         #         bab_savings.append(str(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                     df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'nodisc_bab, %'].mean(), 1)))
#         #         #in_pup_bab
#         #         bab_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                         df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'num_offered_disc_bab'].mean(),0)))
#         #         bab_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'exp_discount_cost_bab_percent'].mean(), 0)))
#         #         #append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_bab'].mean(),0)))
#         #     print("&$p=",p_pup,"$&", ' & '.join(map(str, bab_savings)), "\\")
#         # print("rs")
#         # for p_pup in dict_probability_print:
#         #     rs_savings = []
#         #     for p_delta in dict_probability_print[p_pup]:
#         #         # rs_savings.append(str(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #         #         df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_rs'].mean(),
#         #         #                                 0))))
#         #         rs_savings.append(str(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'nodisc_rs, %'].mean(), 1)))
#         #
#         #         rs_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'num_offered_disc_rs'].mean())))
#         #         rs_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'exp_discount_cost_rs_percent'].mean(), 0)))
#         #         rs_savings.append("ydali")
#         #         #rs_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #         #        df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_rs'].mean())))
#         #
#         #     print("&$p=",p_pup,"$&",' & '.join(map(str, rs_savings)), "\\")
#         # print("uniform")
#         # for p_pup in dict_probability_print:
#         #     uniform_savings = []
#         #     for p_delta in dict_probability_print[p_pup]:
#         #         #uniform_savings.append(str(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #         #        df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_uniform'].mean(),
#         #         #                                     0))))
#         #         uniform_savings.append(str(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                     df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'nodisc_uniform, %'].mean(), 1)))
#         #
#         #         uniform_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                     df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'num_offered_disc_uniform'].mean())))
#         #         uniform_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'exp_discount_cost_uniform_percent'].mean(), 0)))
#         #         uniform_savings.append("ydali")
#         #         #uniform_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
#         #         #        df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_uniform'].mean())))
#         #
#         #     print("&$p=",p_pup,"$&", ' & '.join(map(str, uniform_savings)), "\\")


def print_convergence_gap():
    mainDirStorage = os.path.join(path_to_data, "output")
    convergence = os.path.join(mainDirStorage, 'convergence_19.txt')
    # matplotlib.rcParams['text.usetex'] = True
    # rc('text', usetex=True)
    # plt.rcParams['font.size'] = '16'
    with open(convergence, 'rb') as file:
        time = pickle.load(file)
        lbPrint = pickle.load(file)
        ubPrint = pickle.load(file)
    fig, axes = plt.subplots(1, 1)
    sns.set()
    sns.set(font_scale=1.2)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})

    gap = []
    # for i, lb in enumerate(lbPrint):
    #    gap.append((ubPrint[i] - lbPrint[i]) / ubPrint[i])
    # plt.plot(time, gap, '-', label='Gap')
    #bab_obj = 614.3618363291553/10
    bab_obj = 638.0608078079995/10
    plt.axhline(y=bab_obj, color='r', linestyle='--', alpha=0.3, label='optimal objective value')
    ubPrint_new = [i / 10 for i in ubPrint]
    lbPrint_new = [i / 10 for i in lbPrint]

    lbPrint_new.reverse()
    lbPrint_removed = [lbPrint_new[0]]
    for lb in lbPrint_new:
        lbPrint_removed.append(min(lb, lbPrint_removed[-1]))
    lbPrint_removed.reverse()

    plt.plot(time, ubPrint_new, '-', label='best known upper bound ')
    plt.plot(time, lbPrint_removed[:-1], ':', label='best known lower bound ')
    #axes.set(yscale="log")

    # plt.plot(time, ubPrint, '-',  label='best upper bound ' + r'$(\overline{z}^{*})$')
    # plt.plot(time, lbPrint, '-', label='best lower bound '+r'$(\underline{z}^{*})$')
    #axes.set_ylim(40, 80)
    plt.xlabel("Running time (sec)")
    plt.ylabel("Expected fulfillment cost")
    plt.legend()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.savefig(os.path.join(path_to_images, 'convergence.eps'), transparent=False, bbox_inches='tight')
    plt.show()


def sensitivity_p_home(df, folder):
    path = os.path.join(folder, 'images')

    df['GAP_(OPT-NODISC), %'] = df['nodisc_gap'] * (-100)
    df['GAP_(OPT-DET), %'] = df['rs_gap'] * (-100)
    df['GAP_(OPT-UNIFORM), %'] = df['uniform_gap'] * (-100)
    df['GAP_(OPT-BEST), %'] = df[['GAP_(OPT-NODISC), %', 'GAP_(OPT-DET), %', 'GAP_(OPT-UNIFORM), %']].max(axis=1)
    df['relative_discount'] = df['discount'] / 360

    columns_compare = ['NODISC', 'UNIFORM', 'DET']

    df['p_disc_sensitive'] = round(1 - df['p_home'] - df['p_pup'], 2)

    for instance_type in df.instance_type.unique():

        df_temp = df[df.instance_type == instance_type].copy()
        for discount in df_temp.discount.unique():
            df_temp1 = df_temp[(df_temp.discount == discount) & (df_temp.p_pup == 0.2)].copy()
            sns.set()
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(17, 5))
            axes[0].set_title('No discount')
            axes[1].set_title('All discount')

            # fig.suptitle("Instance type: " + instance_type + ", number of customers: 14, discount size: "+str(discount), y=1)
            iter_ax = 0
            for column_compare in columns_compare:

                axes[iter_ax].set_title(column_compare)

                if column_compare == 'DET':
                    g = sns.lineplot(ax=axes[iter_ax], data=df_temp1, x='p_disc_sensitive',
                                     y='GAP_(OPT-' + column_compare + '), %', marker="o",
                                     palette="deep")

                    new_labels = [r'$P_{pup} = 0.0$', r'$P^{pickup} = 0.2$']
                    axes[iter_ax].legend(new_labels)
                    # for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

                else:
                    sns.lineplot(ax=axes[iter_ax], data=df_temp1, x='p_disc_sensitive',
                                 y='GAP_(OPT-' + column_compare + '), %',
                                 hue='p_pup', marker="o", palette="deep", legend=False)

                axes[iter_ax].set(xlabel=r'$P^{discount sensitive}$')
                axes[iter_ax].set(ylabel='Cost savings in comparison to proposed DS, %')
                iter_ax += 1
            plt.savefig(os.path.join(path, 'Sensitivity_p_home-' + str(
                instance_type) + '-NR_CUST_14' + '-discount_size_' + str(discount) + '.eps'), bbox_inches='tight',
                        pad_inches=0.25, format='eps')

            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            fig.suptitle(instance_type + ", number of customers: 14, discount: " + str(discount), y=1)
            sns.lineplot(ax=ax, data=df_temp1, x='p_home', y='GAP_(OPT-BEST), %', hue='p_pup', marker="o",
                         legend="full", palette="deep")
            plt.savefig(os.path.join(path, 'Sensitivity_p_home-' + str(
                instance_type) + '-NR_CUST_14' + '-OPT-BEST-discount_size_' + str(discount) + '.eps'),
                        bbox_inches='tight',
                        pad_inches=0.25, format='eps')
            plt.show()


def nr_cust_variation(df):
    #df = df[df.nrCust < 19].copy()
    df_results = pd.DataFrame(index=list(range(10, 21)),
                              columns=['t_bab_av', 't_bab_sd','t_bab_min','t_bab_max', 'tto_best','tto_best_min', 'tto_best_max', 'n_bab_av', 'n_bab_sd','n_bab_min','n_bab_max',
                                       'sp2',
                                       'g_opt_3600', 'closed_3600', 'sp1', 't_enum_av', 't_enum_sd','t_enum_min','t_enum_max',
                                       'n_enum', 'n_dom', 'n_dom_min','n_dom_max'])
    for nrCust in range(10, 21):
        df_slice = df[(df.nrCust == nrCust) ].copy()
        #df_slice = df[(df.nrCust == nrCust)& (df.p_pup == 0.2) & (df.discount_rate ==0.06)].copy()

        if nrCust < 21:
            df_results.at[nrCust, 't_enum_av'] = df_slice['time_running_enum'].mean()
            df_results.at[nrCust, 't_enum_sd'] = df_slice['time_running_enum'].std()
            df_results.at[nrCust, 't_enum_min'] = df_slice['time_running_enum'].min()
            df_results.at[nrCust, 't_enum_max'] = df_slice['time_running_enum'].max()
            df_results.at[nrCust, 'n_enum'] = 2 ** round(float(nrCust))

            df_results.at[nrCust, 'n_dom'] = df_slice['time_bab_domcheck'].mean()
            df_results.at[nrCust, 'n_dom_min'] = df_slice['time_bab_domcheck'].min()
            df_results.at[nrCust, 'n_dom_max'] = df_slice['time_bab_domcheck'].max()

            df_results.at[nrCust, 't_bab_av'] = df_slice['time_bab'].mean()
            df_results.at[nrCust, 't_bab_sd'] = df_slice['time_bab'].std()

            df_results.at[nrCust, 't_bab_min'] = df_slice['time_bab'].min()
            df_results.at[nrCust, 't_bab_max'] = df_slice['time_bab'].max()
            df_results.at[nrCust, 'n_bab_av'] = int(df_slice['nodes'].mean() + 0.5)
            df_results.at[nrCust, 'n_bab_min'] = int(df_slice['nodes'].min() + 0.5)
            df_results.at[nrCust, 'n_bab_max'] = int(df_slice['nodes'].max() + 0.5)

            df_results.at[nrCust, 'n_bab_sd'] = int(df_slice['nodes'].std() + 0.5)

        # df_results.at[nrCust, 'g_opt_3600'] = df_slice['opt_gap_h'].mean()
        # df_results.at[nrCust, 'tto_best'] = df_slice['time_tb'].mean()
        # df_results.at[nrCust, 'tto_best_min'] = df_slice['time_tb'].min()
        # df_results.at[nrCust, 'tto_best_max'] = df_slice['time_tb'].max()
        #df_results.at[nrCust, 'closed_3600'] = sum(df_slice['solved_h'])
        #print(sum(df_slice['solved_h']), nrCust)

    #df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()  'n_bab_av', 'n_bab_min', 'n_bab_max'

    df_results = df_results[['t_enum_av', 't_enum_min', 't_enum_max', 'sp1', 't_bab_av','t_bab_min', 't_bab_max','sp2','n_dom', 'n_dom_min','n_dom_max']].copy()
    print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))
    #print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))

    print("")

    # fig, ax = plt.subplots()
    # sns.set_style("whitegrid")
    # ax = sns.scatterplot(data=df_bab, x='nrCust', y='time_running_bab')
    # ax = sns.lineplot(data=df_bab, x='nrCust', y='time_running_bab', hue = 'discount',ci=None)
    # ax.set(ylabel='Solution time, s')
    # plt.show()
    # fig, ax = plt.subplots()
    # sns.set_style("whitegrid")
    # ax = sns.scatterplot(data=df_bab, x='nrCust', y='time_running_bab')
    # df_temp = df_bab[(df_bab.discount == 8) & (df_bab.p_home == 0.2)].copy()
    # ax = sns.lineplot(data=df_temp, x='nrCust', y='time_running_bab', hue='p_pup', ci=None)
    # ax = sns.lineplot(data=df_enum, x='nrCust_enum', y='time_running_enum', ci=None,
    #                  label="mean Enumeration solution time")
    # ax.set(ylabel='Solution time, s')
    # plt.show()


def nr_cust_variation_heuristic(df, df_small):
    index = [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 35, 40, 45, 50]
    df = df[(df.p_home == 0.2)&(df.discount_rate==0.06)].copy()


    df = df[(df['sample'] == 200)].copy()

    sns.set()
    sns.set(font_scale=1.35)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(15, 7))


    cmap = sns.color_palette("deep")
    m = np.array(['o', '^', 'D', '^', 'D', "+"])
    y = []
    yerr = []

    y_d =[]
    yerr_d = []

    y_all =[]
    yerr_all = []

    index_noi = [x-0.2 for x in index]
    index_rs = [x + 0.2 for x in index]


    for iter in index:
        if iter >19:
            df_temp = df[df['nrCust'] ==iter].copy()
        else:
            df_temp = df_small[df_small['nrCust'] == iter].copy()

        df_temp['gap_rs'] = df_temp.apply(lambda x: max(0.08, x['gap_rs']), axis=1)
        df_temp['gap_nodisc'] = df_temp.apply(lambda x: max(0.08, x['gap_nodisc']), axis=1)

        y.append(df_temp['gap_nodisc'].mean())
        yerr.append([df_temp['gap_nodisc'].mean() - df_temp['gap_nodisc'].min(),
                         df_temp['gap_nodisc'].max() - df_temp['gap_nodisc'].mean()])

        y_all.append(df_temp['gap_uniform'].mean())
        yerr_all.append([df_temp['gap_uniform'].mean() - df_temp['gap_uniform'].min(),
                         df_temp['gap_uniform'].max() - df_temp['gap_uniform'].mean()])

        y_d.append(df_temp['gap_rs'].mean())
        yerr_d.append([df_temp['gap_rs'].mean() - df_temp['gap_rs'].min(),
                     df_temp['gap_rs'].max() - df_temp['gap_rs'].mean()])



    yerr = np.transpose(yerr)
    yerr_d = np.transpose(yerr_d)
    yerr_all = np.transpose(yerr_all)

    axes.set_ylim(0, 20.5)

    plt.errorbar(index_noi, y, yerr=yerr, marker=m[0], ms=7, linewidth=0.5,
                    label="NOI",  mec=cmap[0], mfc=cmap[0], c=cmap[0], elinewidth=1, capsize=4)

    plt.errorbar(index, y_all, yerr=yerr_all, marker=m[2], ms=7, linewidth=0.5,
                 label="ALL", mec=cmap[2], mfc=cmap[2], c=cmap[2], elinewidth=1, capsize=4)

    plt.errorbar(index_rs, y_d, yerr=yerr_d, marker=m[1], ms=7, linewidth=0.5,
                 label="DI", mec=cmap[1], mfc=cmap[1], c=cmap[1], elinewidth=1, capsize=4)



    # sns.lineplot(ax=axes, data=df, x='nrCust', y='gap_nodisc', marker='s', color='darkblue', markersize=9,
    #              linewidth=1,
    #              err_style="bars", ci=68, err_kws={'capsize': 3}, label='NOI')
    # sns.lineplot(ax=axes, data=df, x='nrCust', y='gap_uniform', marker='o', color='g', markersize=9, linewidth=1,
    #              err_style="bars", ci=68, err_kws={'capsize': 3}, label='ALL')
    # sns.lineplot(ax=axes, data=df, x='nrCust', y='gap_rs', marker='^', color='r', markersize=9, linewidth=1,
    #              err_style="bars", ci=68, err_kws={'capsize': 3}, label='DI')

    #axes.set_ylim(-1,None)
    #axes.axhline(0, ls='--')
    axes.set(xlabel='Problem size, n')
    axes.set(ylabel='Savings (%)')
    # axes.set(ylabel='Solution time, s (log-scale)')
    plt.legend(title=False, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    #           labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$', r'$p = 0.8$', r'$p = 1.0$'])
    plt.savefig(os.path.join(path_to_images, 'heuristic_improvement.eps'), transparent=False, bbox_inches='tight')

    plt.show()
    sns.set()
    sns.set(font_scale=1.1)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, sharex=True)
    ax2 = axes.twinx()
    sns.lineplot(ax=axes, data=df, x='nrCust', y='time_bab_exact', marker='s', color='darkblue', markersize=9,
                 linewidth=1,
                 err_style="bars", ci=68, err_kws={'capsize': 3}, label='Exact B&B', legend=False)
    sns.lineplot(ax=axes, data=df, x='nrCust', y='time_bab', marker='^', color='r', markersize=9, linewidth=1,
                 err_style="bars", ci=68, err_kws={'capsize': 3}, label='Heuristic B&B', legend=False)
    sns.lineplot(ax=ax2, data=df, x='nrCust', y='opt_gap_h', marker='o', color='g', markersize=9, linewidth=1,
                 err_style="bars", ci=0, err_kws={'capsize': 3}, label='Optimality gap', legend=False)

    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.set_ylim(0, 6)
    ax2.set(ylabel='Optimality gap, %')
    axes.set(xlabel='Instance size')
    axes.set(ylabel='Solution time, s')
    # axes.set(ylabel='Solution time, s (log-scale)')

    plt.savefig(os.path.join(path_to_images, 'heuristic_running_time.eps'), transparent=False, bbox_inches='tight')
    # plt.show()

    df_result_table = pd.DataFrame(index=index,
                                   columns=['nr', 'Time_exact_av', 'Time_exact_sd', 'space_1', 'time_h_av', 'time_h_sd',
                                            'Gap_h_exact_av', 'Gap_h_exact_sd', "space2",
                                            'impr_nodisc_av', 'impr_nodisc_min', 'impr_nodisc_max', 'impr_nodisc_sd',
                                            'impr_rs_av', 'impr_rs_sd', 'impr_rs_min',
                                            'impr_rs_max', 'impr_all_av', 'impr_all_sd', 'impr_all_min', 'impr_all_max',
                                            'space3', 'space4'])

    for nrCust in index:
        df_slice = df[(df.nrCust == nrCust) ].copy()
        df_result_table.at[nrCust, 'Time_exact_av'] = df_slice['time_bab_exact'].mean()
        df_result_table.at[nrCust, 'Time_exact_sd'] = df_slice['time_bab_exact'].std()

        df_result_table.at[nrCust, 'time_h_av'] = min(7200.0, df_slice['time_bab'].mean())
        df_result_table.at[nrCust, 'time_h_sd'] = df_slice['time_bab'].std()

        df_result_table.at[nrCust, 'Gap_h_exact_av'] = df_slice['opt_gap_h'].mean()
        df_result_table.at[nrCust, 'Gap_h_exact_sd'] = df_slice['opt_gap_h'].std()

        df_result_table.at[nrCust, 'impr_nodisc_av'] = df_slice['gap_nodisc'].mean()
        df_result_table.at[nrCust, 'impr_rs_av'] = df_slice['gap_rs'].mean()
        df_result_table.at[nrCust, 'impr_all_av'] = df_slice['gap_uniform'].mean()

        df_result_table.at[nrCust, 'impr_nodisc_sd'] = df_slice['gap_nodisc'].std()
        df_result_table.at[nrCust, 'impr_rs_sd'] = df_slice['gap_rs'].std()
        df_result_table.at[nrCust, 'impr_all_sd'] = df_slice['gap_uniform'].std()

        df_result_table.at[nrCust, 'impr_nodisc_min'] = df_slice['gap_nodisc'].min()
        df_result_table.at[nrCust, 'impr_rs_min'] = df_slice['gap_rs'].min()
        df_result_table.at[nrCust, 'impr_all_min'] = df_slice['gap_uniform'].min()

        df_result_table.at[nrCust, 'impr_nodisc_max'] = df_slice['gap_nodisc'].max()
        df_result_table.at[nrCust, 'impr_rs_max'] = df_slice['gap_rs'].max()
        df_result_table.at[nrCust, 'impr_all_max'] = df_slice['gap_uniform'].max()

    df_result_table_short = df_result_table[[
        'impr_nodisc_av', 'impr_nodisc_min', 'impr_nodisc_max', 'space3',
        'impr_rs_av', 'impr_rs_min', 'impr_rs_max', 'space4', 'impr_all_av', 'impr_all_min', 'impr_all_max']].copy()
    print(df_result_table_short.to_latex(float_format='{:0.1f}'.format, na_rep=''))
    print("")

    # df['num_disc'] = df['policy_bab_ID'].apply(lambda x: bitCount(x))


def parseSampling(file_path, df_bab, folder):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    policy_ID = int(lines[idx + 3].split(':')[1].split('+')[0])
                    obj_val = float(lines[idx + 4].split('[')[1].split(',')[0])
                    ub = float(lines[idx + 4].split('[')[1].split(',')[2].split(']')[0])
                    lb = float(lines[idx + 4].split('[')[1].split(',')[1])
                    data.append([obj_val, lb, ub, policy_ID, instance])
            except:
                data.append(["", "", "", "", instance])
                print("sampling problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['obj_val_sampling', 'lb_sampling', 'ub_sampling', 'policy_ID_sampling', 'instance']
    writer(os.path.join(folder, "sampling_bab.csv"), data, rowTitle)
    df_sampling = pd.read_csv(
        os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", folder, "sampling_bab.csv"))
    df = df_bab.merge(df_sampling, on='instance')
    df['obj_val_bab'] = df['obj_val_sampling']
    del df['obj_val_sampling']
    return df


def experiment_sensitivity_disc_size(folder):
    # parseBAB(os.path.join(folder, "babExact_clustered.txt"), folder)
    # parseRS(os.path.join(folder,"RingStar_clustered.txt"), folder)
    # df_bab = pd.read_csv(os.path.join(folder, "bab_processed.csv"))
    # df_enum = pd.read_csv(os.path.join(folder, "Enumeration_processed.csv"))
    # df_rs = pd.read_csv(os.path.join(folder,"RS_processed.csv"))
    # create_comparison_table(df_bab, df_enum, df_rs,  folder)
    df_comparison = pd.read_excel(os.path.join(folder, "Comparison_full_table.xlsx"))
    sensitivity_disc_size(df_comparison, folder)


def experiment_sensitivity_p_home(folder):
    # sensitivity_p_home:
    # parseBAB(os.path.join(folder, "babExact_phome_sensitivity.txt"), folder)
    # parseRS(os.path.join(folder, "RingStar_phome_sensitivity.txt"), folder)
    df_bab = pd.read_csv(os.path.join(folder, "bab_processed.csv"))
    df_enum = pd.read_csv(os.path.join(folder, "Enumeration_processed.csv"))
    df_rs = pd.read_csv(os.path.join(folder, "RS_processed.csv"))
    # create_comparison_table(df_bab, df_enum, df_rs,  folder)
    df_comparison = pd.read_excel(os.path.join(folder, "Comparison_full_table.xlsx"))
    sensitivity_p_home(df_comparison, folder)


def experiment_variation_nrcust(folder):
    # Speed test on nr_cust_variation instances
    # parseBAB_RS_NODISC(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_notfinished.txt"), folder, "i_VRPDO_time")
    #parseBAB_RS_NODISC(os.path.join(folder, "bab_VRPDO_discount_proportional_02_02.txt"), folder, "i_VRPDO_discount_proportional_02_02")
    #parseBAB(os.path.join(folder, "02_23_bab_exact.txt"), folder, "02_23_bab_exact")
    #parseEnumeration(os.path.join(folder, "02_13_enumeration.txt"), folder, "02_13_enumeration")

    if False: #print table with bab_exact and enumeration running times
        df = pd.read_csv(os.path.join(folder, "02_23_bab_exact.csv"))
        df_enum =  pd.read_csv(os.path.join(folder, "02_13_enumeration.csv"))
        # df = df[df.nrCust < 19].copy()
        df_results = pd.DataFrame(index=list(range(10, 21)),
                                  columns=['t_enum_av', 't_enum_min', 't_enum_max', 'sp1','t_bab_av', 't_bab_min', 't_bab_max'])
        for nrCust in range(10, 21):
            df_slice_bab = df[(df.nrCust == nrCust)].copy()
            df_slice_enum = df_enum[(df_enum.nrCust_enum == nrCust)].copy()
            if nrCust < 21:
                df_results.at[nrCust, 't_enum_av'] = df_slice_enum['time_running_enum'].mean()
                df_results.at[nrCust, 't_enum_min'] = df_slice_enum['time_running_enum'].min()
                df_results.at[nrCust, 't_enum_max'] = df_slice_enum['time_running_enum'].max()

                df_results.at[nrCust, 't_bab_av'] = df_slice_bab['time_bab'].mean()
                df_results.at[nrCust, 't_bab_min'] = df_slice_bab['time_bab'].min()
                df_results.at[nrCust, 't_bab_max'] = df_slice_bab['time_bab'].max()

        df_results = df_results[
            ['t_enum_av', 't_enum_min', 't_enum_max', 'sp1', 't_bab_av', 't_bab_min', 't_bab_max']].copy()
        print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))
        # print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep='')
        print("")
    # check dominance usage  02_26_bab_exact_domcheck.txt
    #print table with percentage of nodes pruned by dominance rules 1, 2, and bounding
    if False:
        # parseBAB(os.path.join(folder, "05_07_babfull_exact_num_pruned.txt"), folder, "05_07_babfull_exact_num_pruned")
        df_bab = parseProfile(os.path.join(folder, "05_07_babfull_exact_num_pruned.txt"))
        #df_bab = pd.read_csv(os.path.join(folder, "05_07_babfull_exact_num_pruned.csv"))
                #['nrCust',"nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                #'num_tsps', 'obj_val_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                #'pr_cliques_nonleaf', 'pr_cliques_leaf', 'pr_rs_nonleaf', 'pr_rs_leaf',
                #'pr_insertionCost_nonleaf', 'pr_insertionCost_leaf', 'pr_bounds_nonleaf',
                #'pr_bounds_leaf','tsp_time', 'time_exact_bounds', 'time_lb_addition', 'time_branch']
        df_results = pd.DataFrame(index=list(range(10, 20)),
                                  columns=['nr_nodes','npruned_d1', 'npruned_d2', 'npruned_bounds'])
        # df = df.fillna(0)
        for nrCust in range(10, 19):
            df_slice = df_bab[(df_bab.nrCust == nrCust)].copy()
            if nrCust < 20:
                print(nrCust)
                df_results.at[nrCust, 'nr_nodes'] = int(df_slice['nodes'].mean() +0.5)

                df_results.at[nrCust, 'npruned_d1'] = int(df_slice['pr_insertionCost_leaf'].mean() + df_slice['pr_insertionCost_nonleaf'].mean()+\
                                                          df_slice['pr_rs_leaf'].mean() + df_slice['pr_rs_nonleaf'].mean()+0.5)
                df_results.at[nrCust, 'npruned_d2'] = int(df_slice['pr_cliques_nonleaf'].mean() + df_slice['pr_cliques_leaf'].mean()+0.5)
                df_results.at[nrCust, 'npruned_bounds'] = int(df_slice['pr_bounds_leaf'].mean() + df_slice['pr_bounds_nonleaf'].mean()+0.5)


        # df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()  'n_bab_av', 'n_bab_min', 'n_bab_max'

        print(df_results.to_latex(float_format='{:0.0f}'.format, na_rep=''))
        # print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))

        print("")
    # print table with runing time and number of explored nodes if we use or not dominance rules
    if False:
        rowTitle = ['nrCust', "nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                    'num_tsps', 'obj_val_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                    'pr_cliques_nonleaf', 'pr_cliques_leaf', 'pr_rs_nonleaf', 'pr_rs_leaf',
                    'pr_insertionCost_nonleaf', 'pr_insertionCost_leaf', 'pr_bounds_nonleaf',
                    'pr_bounds_leaf', 'tsp_time', 'time_exact_bounds', 'time_lb_addition', 'time_branch']
        folder_domcheck = os.path.join(folder,"07_2023_dominance_checks")
        df_full = parseProfile(os.path.join(folder_domcheck, "05_07_babfull_exact_num_pruned.txt"))
        df_noins = parseProfile(os.path.join(folder_domcheck, "05_07_babnoins_exact_num_pruned.txt"))
        df_nocliques = parseProfile(os.path.join(folder_domcheck, "05_07_babnocliques_exact_num_pruned.txt"))
        df_noinscliques= parseProfile(os.path.join(folder_domcheck, "05_07_babnocliquesnoins_exact_num_pruned.txt"))
        df_results = pd.DataFrame(index=list(range(10, 20)),
                                  columns=[ 't_noinscliques', 'n_noinscliques', 'sp1',
                                            't_nocliques', 'n_nocliques', 'sp3',
                                           't_noins', 'n_noins', 'sp2',
                                          't_full', 'n_full'])
        # df = df.fillna(0)
        for nrCust in range(10, 16):
            df_full_slice = df_full[(df_full.nrCust == nrCust)].copy()
            df_noins_slice = df_noins[(df_noins.nrCust == nrCust)].copy()
            df_nocliques_slice = df_nocliques[(df_nocliques.nrCust == nrCust)].copy()
            df_noinscliques_slice = df_noinscliques[(df_noinscliques.nrCust == nrCust)].copy()

            if nrCust < 20:
                print(nrCust)
                df_results.at[nrCust, 't_full'] = df_full_slice['time_bab'].mean()
                df_results.at[nrCust, 'n_full'] = int(df_full_slice['nodes'].mean() + 0.5)

                df_results.at[nrCust, 't_noins'] = df_noins_slice['time_bab'].mean()
                df_results.at[nrCust, 'n_noins'] = int(df_noins_slice['nodes'].mean() + 0.5)
                #df_results.at[nrCust, 'n_noins'] = "{:<7}".format(str(int(df_noins_slice['nodes_domins'].mean() + 0.5))) + \
                #                                 "(" + str(int(df_noins_slice['pruned_domins'].mean() + 0.5)) + ")"

                df_results.at[nrCust, 't_nocliques'] = df_nocliques_slice['time_bab'].mean()
                df_results.at[nrCust, 'n_nocliques'] = int(df_nocliques_slice['nodes'].mean() + 0.5)

                df_results.at[nrCust, 't_noinscliques'] = df_noinscliques_slice['time_bab'].mean()
                df_results.at[nrCust, 'n_noinscliques'] = int(df_noinscliques_slice['nodes'].mean() + 0.5)

        # df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()  'n_bab_av', 'n_bab_min', 'n_bab_max'
        print(df_results.to_latex(float_format='{:0.0f}'.format, na_rep=''))
        # print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))

        print("")
    if True:
        rowTitle = ['nrCust', "nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                    'num_tsps', 'obj_val_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                    'pr_cliques_nonleaf', 'pr_cliques_leaf', 'pr_rs_nonleaf', 'pr_rs_leaf',
                    'pr_insertionCost_nonleaf', 'pr_insertionCost_leaf', 'pr_bounds_nonleaf',
                    'pr_bounds_leaf', 'tsp_time', 'time_exact_bounds', 'time_lb_addition', 'time_branch']
        folder_domcheck = os.path.join(folder,"07_2023_dominance_checks")
        df_full = parseProfile(os.path.join(folder_domcheck, "05_07_babfull_exact_num_pruned.txt"))
        df_noins = parseProfile(os.path.join(folder_domcheck, "05_07_babnoins_exact_num_pruned.txt"))
        df_nocliques = parseProfile(os.path.join(folder_domcheck, "05_07_babnocliques_exact_num_pruned.txt"))
        df_noinscliques = parseProfile(os.path.join(folder_domcheck, "05_07_babnocliquesnoins_exact_num_pruned.txt"))
        df_results = pd.DataFrame(index=list(range(10, 20)),
                                  columns=[ 't_full', 'np_full_bound', 'np_full_dom1', 'np_full_dom2', 'sp1',
                                            't_noins', 'np_noins_bound', 'np_noins_dom',  'sp2',
                                           't_nocliques', 'np_nocliques_bound',  'np_nocliques_dom', 'sp3',
                                      't_noinscliques', 'np_noinscliques_bound'                                  ])
        # df = df.fillna(0)
        for nrCust in range(10, 20):
            df_full_slice = df_full[(df_full.nrCust == nrCust)].copy()
            df_noins_slice = df_noins[(df_noins.nrCust == nrCust)].copy()
            df_nocliques_slice = df_nocliques[(df_nocliques.nrCust == nrCust)].copy()
            df_noinscliques_slice = df_noinscliques[(df_noinscliques.nrCust == nrCust)].copy()

            if nrCust < 20:
                print(nrCust)
                df_results.at[nrCust, 't_noinscliques'] = df_noinscliques_slice['time_bab'].mean()
                df_results.at[nrCust, 'np_noinscliques_bound'] = int(df_noinscliques_slice['pr_bounds_leaf'].mean() +
                                                                df_noinscliques_slice['pr_bounds_nonleaf'].mean()+0.5)

                df_results.at[nrCust, 't_noins'] = df_noins_slice['time_bab'].mean()
                df_results.at[nrCust, 'np_noins_bound'] = int(df_noins_slice['pr_bounds_leaf'].mean() + df_noins_slice['pr_bounds_nonleaf'].mean()+0.5)
                df_results.at[nrCust, 'np_noins_dom'] = int(df_noins_slice['pr_cliques_leaf'].mean() + df_noins_slice['pr_cliques_nonleaf'].mean()+0.5)


                df_results.at[nrCust, 't_nocliques'] = df_nocliques_slice['time_bab'].mean()
                df_results.at[nrCust, 'np_nocliques_dom'] = int(df_nocliques_slice['pr_insertionCost_leaf'].mean() +
                                                              df_nocliques_slice['pr_insertionCost_nonleaf'].mean() +
                                                              df_nocliques_slice['pr_rs_leaf'].mean() +
                                                              df_nocliques_slice['pr_rs_nonleaf'].mean() + 0.5)
                df_results.at[nrCust, 'np_nocliques_bound'] =  int(df_nocliques_slice['pr_bounds_leaf'].mean() +
                                                               df_nocliques_slice['pr_bounds_nonleaf'].mean()+0.5)

                df_results.at[nrCust, 't_full'] = df_full_slice['time_bab'].mean()
                df_results.at[nrCust, 'np_full_dom2'] = int(df_full_slice['pr_cliques_leaf'].mean() + df_full_slice['pr_cliques_nonleaf'].mean()+0.5)

                df_results.at[nrCust, 'np_full_dom1'] = int(df_full_slice['pr_insertionCost_leaf'].mean() +
                                                                df_full_slice['pr_insertionCost_nonleaf'].mean() +
                                                                df_full_slice['pr_rs_leaf'].mean() +
                                                                df_full_slice['pr_rs_nonleaf'].mean() + 0.5)
                df_results.at[nrCust, 'np_full_bound'] = int(df_full_slice['pr_bounds_leaf'].mean() +
                                                                  df_full_slice['pr_bounds_nonleaf'].mean() + 0.5)

        # df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()  'n_bab_av', 'n_bab_min', 'n_bab_max'
        print(df_results.to_latex(float_format='{:0.0f}'.format, na_rep=''))
        # print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))

        print("")
    if False:
        #parseBAB(os.path.join(folder, "02_26_bab_exact_domcheck.txt"), folder, "02_26_bab_exact_domcheck")
        df_bab = pd.read_csv(os.path.join(folder, "02_26_bab_exact_domcheck.csv"))
        df_bab = df_bab[['instance', 'nrCust', 'nodes', 'time_bab']].copy()

        df_dom_lb = parseProfile(os.path.join(folder, "02_26_bab_dominance_check_lb.txt"))
        df_dom_cliques = parseProfile(os.path.join(folder, "02_26_bab_dominance_check_cliques.txt"))
        df_dom_insertion = parseProfile(os.path.join(folder, "02_26_bab_dominance_check_ins.txt"))
        df_dom_basic= parseProfile(os.path.join(folder, "02_26_bab_dominance_check_basic.txt"))


        df_dom_basic = df_dom_basic[['instance', 'nodes',  'time_bab']].copy()
        df_dom_basic.rename(columns={'nodes': 'nodes_dombasic', 'time_bab': 'time_dombasic'}, inplace=True)

        df_dom_cliques['pruned_domcliques'] = df_dom_cliques['pr_cliques_nonleaf'] + df_dom_cliques['pr_cliques_leaf']
        df_dom_cliques = df_dom_cliques[['instance', 'nodes', 'pruned_domcliques', 'time_bab']].copy()
        df_dom_cliques.rename(columns={'nodes': 'nodes_domclique', 'time_bab': 'time_domclique'}, inplace=True)

        df_dom_insertion['pruned_domins'] = df_dom_insertion['pr_insertionCost_nonleaf'] + df_dom_insertion['pr_insertionCost_leaf']
        df_dom_insertion = df_dom_insertion[['instance', 'nodes', 'pruned_domins', 'time_bab']].copy()
        df_dom_insertion.rename(columns={'nodes': 'nodes_domins', 'time_bab': 'time_domins'}, inplace=True)

        df_dom_lb['pruned_domlb'] = df_dom_lb['pr_bounds_nonleaf'] + df_dom_lb['pr_bounds_leaf']
        df_dom_lb = df_dom_lb[['instance', 'nodes', 'pruned_domlb', 'time_bab']].copy()
        df_dom_lb.rename(columns={'nodes': 'nodes_domlb', 'time_bab': 'time_domlb'}, inplace=True)

        df = df_bab.merge(df_dom_lb, how='left', on='instance')
        df = df.merge(df_dom_cliques, how='left', on='instance')
        df = df.merge(df_dom_insertion, how='left', on='instance')
        df = df.merge(df_dom_basic, how='left', on='instance')
        df_results = pd.DataFrame(index=list(range(10, 21)),
                              columns=['t_bab', 'n_bab','sp1',
                                        't_basic','n_basic', 'sp4',
                                       't_ins','n_ins','sp2',
                                       't_cliques','n_cliques','sp3',
                                       't_lb','n_lb'])
        #df = df.fillna(0)
        for nrCust in range(10, 20):
            df_slice = df[(df.nrCust == nrCust) ].copy()
            if nrCust < 20:
                print(nrCust)
                df_results.at[nrCust, 't_bab'] = df_slice['time_bab'].mean()
                df_results.at[nrCust, 'n_bab'] = int(df_slice['nodes'].mean() + 0.5)

                df_results.at[nrCust, 't_ins'] = df_slice['time_domins'].mean()
                df_results.at[nrCust, 'n_ins'] = "{:<7}".format(str(int(df_slice['nodes_domins'].mean() + 0.5)))+\
                                                     "("+str(int(df_slice['pruned_domins'].mean() + 0.5)) + ")"

                df_results.at[nrCust, 't_cliques'] = df_slice['time_domclique'].mean()
                df_results.at[nrCust, 'n_cliques'] = "{:<7}".format(str(int(df_slice['nodes_domclique'].mean() + 0.5)))+\
                                                     "("+str(int(df_slice['pruned_domcliques'].mean() + 0.5)) + ")"

                df_results.at[nrCust, 't_lb'] = df_slice['time_domlb'].mean()
                df_results.at[nrCust, 'n_lb'] = int(df_slice['nodes_domlb'].mean() + 0.5)
                if nrCust < 19:
                    df_results.at[nrCust, 't_basic'] = df_slice['time_dombasic'].mean()
                    df_results.at[nrCust, 'n_basic'] = int(df_slice['nodes_dombasic'].mean() + 0.5)

        #df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()  'n_bab_av', 'n_bab_min', 'n_bab_max'

        print(df_results.to_latex(float_format='{:0.0f}'.format, na_rep=''))
        #print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))

        print("")
    if False: #print table with timelimit
        df_bab = pd.read_csv(os.path.join(folder, "02_23_bab_time_limit.csv"))
        df_time_limit = pd.read_csv(os.path.join(folder, "02_16_bab_time_limit.csv"))
        df_time_limit = df_time_limit[['instance', 'policy_bab_ID', 'obj_val_bab', '2sd_bab','optimal']].copy()
        df_time_limit.rename(columns={'policy_bab_ID': 'policy_h_ID', 'obj_val_bab': 'obj_val_h', '2sd_bab': '2sd_h',
                                 'optimal': 'optimal_h'}, inplace=True)

        df_bab = df_bab.merge(df_time_limit, how='left', on='instance')

        df_bab['opt_gap_h'] = df_bab.apply(
            lambda x: 0 if (x['policy_h_ID'] == x['policy_bab_ID'] or x.time_bab<3600) else (x['obj_val_h'] - x['obj_val_bab']) / x[
                'obj_val_bab'] * 100, axis=1)
        df_bab['solved_h'] = df_bab.apply(
           lambda x: 0 if x['time_bab'] >=3600 else 1, axis=1)

        df_results = pd.DataFrame(index=list(range(10, 21)),
                                  columns=['tto_best', 'tto_best_min',  'tto_best_max',
                                           'sp1',  'g_opt_3600', 'closed_3600'])

        df_bab = pd.read_excel(os.path.join(folder, "Comparison_full_table.xlsx"))
        for nrCust in range(10, 21):
            df_slice = df_bab[(df_bab.nrCust == nrCust)].copy()
            if nrCust < 21:
                df_results.at[nrCust, 'g_opt_3600'] = df_slice['opt_gap_h'].mean()
                df_results.at[nrCust, 'tto_best'] = df_slice['time_tb'].mean()
                df_results.at[nrCust, 'tto_best_min'] = df_slice['time_tb'].min()
                df_results.at[nrCust, 'tto_best_max'] = df_slice['time_tb'].max()
                df_results.at[nrCust, 'closed_3600'] = sum(df_slice['solved_h'])
            # print(sum(df_slice['solved_h']), nrCust)

        df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()
        print(df_results.to_latex(float_format='{:0.2f}'.format, na_rep=''))
        #print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))
        # file_path = os.path.join( folder,  "Comparison_full_table.xlsx")
        # writer = pd.ExcelWriter(file_path, engine='openpyxl')
        # df_bab.to_excel(writer, index=False)
        # writer.save()
        #


def experiment_variation_nrcust_heuristic(folder):
    #parseBAB(os.path.join(folder, "bab_3600_VRPDO_disc_proportional_02_02_006.txt"), folder,
    #         "bab_3600_VRPDO_disc_proportional_02_02_006")
    #parseBABHeuristic(os.path.join(folder, "h_VRPDO_disc_proportional_02_02.txt"), folder, "h_VRPDO_disc_proportional_02_02")

    #df_bab = pd.read_csv(os.path.join(folder, "i_VRPDO_time.csv"))
    #df_heuristic = pd.read_csv(os.path.join(folder, "h_i_VRPDO.csv"))

    df_bab = pd.read_csv(os.path.join(folder, "bab_3600_VRPDO_disc_proportional_02_02_006.csv"))


    #df_heuristic = pd.read_csv(os.path.join(folder, "h_rs_nodisc_VRPDO_disc_proportional.csv"))

    #df_bab = pd.read_csv(os.path.join(folder, "bab3600_VRPDO_disc_proportional_02_02.csv"))
    df_bab_small = pd.read_csv(os.path.join(folder, "bab_VRPDO_discount_proportional_small.csv"))

    df_heuristic = pd.read_csv(os.path.join(folder, "h_VRPDO_disc_proportional_02_02.csv"))

    df_small = pd.read_csv(os.path.join(folder, "bab_VRPDO_disc_proportional_small_02_02_006.csv"))
    df_small = df_small[(df_small.p_home == 0.2) & (df_small.discount_rate == 0.06)].copy()

    df_h_alternative = df_heuristic[['instance', 'obj_val_nodisc', 'obj_val_rs', 'obj_val_uniform','sample']].copy()
    df_h_alternative = df_h_alternative[df_h_alternative['sample'] == 200].copy()

    df_small = df_small.merge(df_h_alternative, how='left', on='instance')
    df_small['gap_rs'] = df_small.apply( lambda x: ((x['obj_val_rs']-x['obj_val_bab']) / x['obj_val_rs']*100), axis=1)
    df_small['gap_nodisc'] = df_small.apply(lambda x: (x['obj_val_nodisc'] - x['obj_val_bab'])*100 / x['obj_val_nodisc'], axis=1)
    df_small['gap_uniform'] = df_small.apply(lambda x: (x['obj_val_uniform'] - x['obj_val_bab'])*100 / x['obj_val_uniform'], axis=1)


    df_bab = df_bab[['instance',  'time_bab', 'obj_val_bab', '2sd_bab', 'policy_bab_ID']].copy()
    df_bab.rename(columns={'time_bab': 'time_bab_exact', 'obj_val_bab': 'obj_val_bab_exact', '2sd_bab': '2sd_bab_exact',
                           'policy_bab_ID': 'policy_bab_exact_ID'}, inplace=True)

    df = df_heuristic.merge(df_bab, how='outer', on='instance')
    # gap of heuristic in comparison with bab exact
    df['opt_gap_h'] = df.apply(
        lambda x: 0 if x['policy_bab_exact_ID'] == x['policy_bab_ID'] else (x['obj_val_bab'] - x['obj_val_bab_exact']) /
                                                                           x['obj_val_bab'] * 100, axis=1)
    nr_cust_variation_heuristic(df, df_small)

    # df.rename(columns={'nrCust': 'Number of customers', 'gap_nodisc': '(nodisc-bab)/nodisc, %','gap_rs':'(rs-bab)/rs, %','gap_uniform':'(uniform-bab)/uniform, %'}, inplace=True)
    # for i, row in df.iterrows():
    #     df.at[i, 'instance_type'] = (df.at[i, 'instance']).split('_')[0]
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # #sns.lineplot(ax=ax, x='nrCust', y='gap_rs',hue='discount', data=df, marker=True, palette="deep", label='(rs-bab)/rs')
    # #plt.title("Numer of explored nodes by heuristic\n Sample size:50; Eps:0.5%; Time limit: 600s")
    # sns.lineplot(ax=ax[0], x='Number of customers', y='(nodisc-bab)/nodisc, %', data=df, hue='instance_type', marker=True, palette="deep")
    # ax[0].set_title('No Discount')
    # sns.lineplot(ax=ax[1], x='Number of customers', y='(rs-bab)/rs, %', data=df, hue='instance_type', marker=True,
    #              palette="deep")
    # ax[1].set_title('Ring-Star')
    # sns.lineplot(ax=ax[2], x='Number of customers', y='(uniform-bab)/uniform, %', data=df, hue='instance_type', marker=True,
    #              palette="deep")
    # ax[2].set_title('Uniform')
    #
    # #plt.ylabel('Gap' + ', %')
    # plt.show()
    #
    # dict_n = {}
    # for n in range(20,51,5):
    #     dict_n[n] = str(n) + "  & & &&  & "
    #     df_temp = df[df[ 'Number of customers'] == n]
    #     dict_n[n] = dict_n[n] + str(round(df_temp['time_bab'].mean(),1))+ " &"+ str(int(df_temp['nodes'].mean()))
    #
    #     print(dict_n[n][:len(dict_n[n])]+'\\'+'\\')


def create_comparison_table(df_bab, df_enum, folder):
    df = df_enum.merge(df_bab, how='outer', on='instance')
    #
    # df = df[['nrCust', 'time_running_x', 'obj_val_x', 'policyENUM_ID', 'policyENUM', 'instance', 'nrCust_y',
    #         'prob_choose_disc', 'discount', 'time_running_bab', 'time_first_opt', 'Nodes','num_tsps', 'optimal',
    #         'gap', 'obj_val_y', 'policyBAB_ID', 'policyBAB', 'instance']].copy()
    df['opt_gap'] = (df['obj_val_bab'] - df['obj_val_enum']) / df['obj_val_bab']

    df['opt_gap_indicator'] = df['opt_gap'].apply(lambda x: 1 if x > 10e-5 else 0)

    df = df[['nrCust', 'p_home', 'p_pup', 'discount', 'time_running_enum', 'obj_val_enum', 'policy_enum_ID',
             'time_bab', 'time_tb', 'obj_val_bab', 'optimal', 'gap', 'opt_gap', 'opt_gap_indicator', 'nodes',
             'num_tsps',
             'policy_bab_ID', 'num_disc_bab', 'instance']].copy()
    file_path = os.path.join(folder, "Comparison_full_table.xlsx")
    writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a+')
    df.to_excel(writer, index=False)
    writer.save()


def concorde_gurobi_comparison():
    # parse_raw_txt
    folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "Concorde_Gurobi_comparison_Santini")
    # parseEnumeration( os.path.join(folder, "enumeration_Santini_Gurobi_2segments.txt"),folder, "enumeration_Gurobi_2segments")
    # parseEnumeration( os.path.join(folder, "enumeration_Santini_Gurobi_3segments.txt"), folder, "enumeration_Gurobi_3segments")
    # parseEnumeration(os.path.join(folder, "enumeration_Santini_Concorde_3segments.txt"), folder,
    #                 "enumeration_Concorde_3segments")
    # parseEnumeration(os.path.join(folder, "enumeration_Santini_Concorde_2segments.txt"), folder,
    #                                  "enumeration_Concorde_2segments")

    df_Concorde_2 = pd.read_csv(os.path.join(folder, "enumeration_Concorde_2segments.csv"))
    df_Concorde = pd.read_csv(os.path.join(folder, "bab_Concorde.csv"))

    # df_Concorde = parseSampling(os.path.join(folder, "sampling.txt"), df_Concorde, folder)
    # df_Concorde.to_csv(os.path.join(folder, "bab_Concorde.csv"))

    df_rs = pd.read_csv(os.path.join(folder, "RS_empty.csv"))
    create_comparison_table(df_Concorde, df_Concorde_2, df_rs, folder)

    # read csv to dataframe
    df_Gurobi_2 = pd.read_csv(os.path.join(folder, "enumeration_Gurobi_2segments.csv"))
    df_Gurobi_3 = pd.read_csv(os.path.join(folder, "enumeration_Gurobi_3segments.csv"))
    df_Concorde_3 = pd.read_csv(os.path.join(folder, "enumeration_Concorde_3segments.csv"))
    df_Concorde_2 = pd.read_csv(os.path.join(folder, "enumeration_Concorde_2segments.csv"))

    # rename columns
    df_Gurobi_2.rename(columns={'time_running_enum': 'time_enum_Gurobi_2', 'obj_val_enum': 'obj_val_enum_Gurobi_2'},
                       inplace=True)
    df_Gurobi_3.drop('nrCust_enum', axis=1, inplace=True)
    df_Gurobi_3.rename(columns={'time_running_enum': 'time_enum_Gurobi_3', 'obj_val_enum': 'obj_val_enum_Gurobi_'},
                       inplace=True)
    df_Concorde_3.rename(
        columns={'time_running_enum': 'time_enum_Concorde_3', 'obj_val_enum': 'obj_val_enum_Concorde_3'}, inplace=True)
    df_Concorde_3.drop('nrCust_enum', axis=1, inplace=True)
    df_Concorde_2.rename(
        columns={'time_running_enum': 'time_enum_Concorde_2', 'obj_val_enum': 'obj_val_enum_Concorde_2'}, inplace=True)
    df_Concorde_2.drop('nrCust_enum', axis=1, inplace=True)

    df_Concorde_3 = df_Concorde_3.merge(df_Concorde_2, on='instance', how='outer')
    df_Concorde_3 = df_Concorde_3.merge(df_Gurobi_3, on='instance', how='outer')
    df_Concorde_3 = df_Concorde_3.merge(df_Gurobi_2, on='instance', how='outer')

    # writer = pd.ExcelWriter(os.path.join(folder, "Comparison_Concorde_Gurobi.xlsx"), engine='openpyxl', mode='a+')
    # df_Concorde_3.to_excel(writer, index=False)
    # writer.save()

    df = pd.read_excel(os.path.join(folder, "Comparison_Concorde_Gurobi.xlsx"))

    df_result_table = pd.DataFrame(
        index=['Time Gurobi 2 segments,s', 'Time Gurobi 3 segments,s', 'Time Concorde 3 segments,s',
               'Time Concorde 2 segments,s'],
        # columns=df.nrCust_enum.unique())
        columns=list(range(8, 19)))

    # for nrCust in df.nrCust_enum.unique():
    for nrCust in range(8, 19):
        df_slice = df[(df.nrCust_enum == nrCust)].copy()
        df_result_table.at['Time Gurobi 2 segments,s', nrCust] = df_slice['time_enum_Gurobi_2'].mean()
        df_result_table.at['Time Gurobi 3 segments,s', nrCust] = df_slice['time_enum_Gurobi_3'].mean()
        df_result_table.at['Time Concorde 3 segments,s', nrCust] = df_slice['time_enum_Concorde_3'].mean()
        df_result_table.at['Time Concorde 2 segments,s', nrCust] = df_slice['time_enum_Concorde_2'].mean()

    print(df_result_table.to_latex(float_format='{:0.1f}'.format, na_rep='-'))
    print("")

    # Parse BAB results for concorde and Gurobi
    # parseBAB_Santini(os.path.join(folder, "bab_Santini_Concorde.txt"), folder, "bab_Concorde")
    # parseBAB_Santini(os.path.join(folder, "bab_Santini_Gurobi.txt"), folder, "bab_Gurobi")

    # read csv to dataframe
    df_Gurobi = pd.read_csv(os.path.join(folder, "bab_Gurobi.csv"))

    # rename columns
    df_Gurobi.rename(columns={'time_running_bab': 'time_running_bab_Gurobi', 'obj_val_bab': 'obj_val_bab_Gurobi',
                              'policy_bab_ID': 'policy_bab_ID_Gurobi'},
                     inplace=True)
    df_Gurobi.drop(
        ['p_home', 'p_pup', 'discount', 'nodes', 'num_tsps', 'optimal', 'gap', 'time_tp', 'policy_bab',
         'num_disc_bab'], axis=1, inplace=True)

    df_Concorde.rename(columns={'time_running_bab': 'time_running_bab_Concorde', 'obj_val_bab': 'obj_val_bab_Concorde',
                                'policy_bab_ID': 'policy_bab_ID_Concorde'},
                       inplace=True)
    df_Concorde.drop(
        ['nrCust', 'p_home', 'p_pup', 'discount', 'nodes', 'num_tsps', 'optimal', 'gap', 'time_tb', 'policy_bab',
         'num_disc_bab'], axis=1, inplace=True)

    df_Gurobi = df_Gurobi.merge(df_Concorde, on='instance', how='outer')
    # writer = pd.ExcelWriter(os.path.join(folder, "Comparison_BAB_Concorde_Gurobi.xlsx"), engine='openpyxl', mode='a+')
    # df_Gurobi.to_excel(writer, index=False)
    # writer.save()

    df = pd.read_excel(os.path.join(folder, "Comparison_BAB_Concorde_Gurobi.xlsx"))
    df_result_table = pd.DataFrame(
        index=['Time BAB Gurobi,s', 'Time BAB Concorde,s'],
        columns=df.nrCust.unique())
    # columns=list(range(8, 18)))

    for nrCust in df.nrCust.unique():
        # for nrCust in range(8, 18):
        df_slice = df[(df.nrCust == nrCust)].copy()
        df_result_table.at['Time BAB Gurobi,s', nrCust] = df_slice['time_running_bab_Gurobi'].mean()
        df_result_table.at['Time BAB Concorde,s', nrCust] = df_slice['time_running_bab_Concorde'].mean()

    print(df_result_table.to_latex(float_format='{:0.1f}'.format))
    print("")


def experiment_varying_discount(folder):
    # Parse BAB results for concorde and Gurobi
    #parseBABHeuristic(os.path.join(folder, "h_VRPDO_temp_check_discounts.txt"), folder, "h_VRPDO_temp_check_discounts")
    df_bab = pd.read_csv(os.path.join(folder, "h_VRPDO_temp_check_discounts.csv"))
    df_bab['num_disc_best'] = df_bab.apply(
        lambda x: bitCount(x['policy_bab_ID']) / x['nrCust'] * 100 if (
                    (x['gap_uniform'] >= 0) and (x['gap_nodisc'] >= 0) and (x['gap_rs'] >= 0))
        else (bitCount(x['policy_ID_rs']) / x['nrCust'] * 100 if (x['gap_rs'] < 0) else (
            100 if (x['gap_uniform'] < 0) else 0)), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.lineplot(ax=ax, data=df_bab, x='discount_rate', y='num_disc_best', hue='nrCust', marker="o", palette="deep")
    plt.show()


def experiment_heuristic_parameters_variation(folder):
    # Parse BAB results for concorde and Gurobi
    #parseBAB(os.path.join(folder, "h_rs.txt"), folder, "h_rs")
    df_rs = pd.read_csv(os.path.join(folder, "h_rs.csv"))

    #parseBAB(os.path.join(folder, "bab_time_limit.txt"), folder, "bab_time_limit")
    df_bab = pd.read_csv(os.path.join(folder, "bab_time_limit.csv"))

    #parseBABHeuristic(os.path.join(folder, "h_s20.txt"), folder, "h_s20")
    df_h20 = pd.read_csv(os.path.join(folder, "h_s20.csv"))

    #parseBABHeuristic(os.path.join(folder, "h_s100.txt"), folder, "h_s100")
    df_h100 = pd.read_csv(os.path.join(folder, "h_s100.csv"))

    #parseBABHeuristic(os.path.join(folder, "h_s200.txt"), folder, "h_s200")
    df_h200 = pd.read_csv(os.path.join(folder, "h_s200.csv"))

    nr_cust = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]
    #nr_cust = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30]
    df_results = pd.DataFrame(index=nr_cust,
                        columns=['exactt','exactn',"s1",'s500t','s500node', 's500gap','s2', 's200t',  's200node', 's200gap',
                        's3', 's100t', 's100node', 's100gap', 's4','s50t', 's50node','s50gap', 's5', 's20t', 's20node',
                        's20gap','s6','s0t', 's0node', 's0gap']) #s0t is ring star heuristic without sampling

    #df_h = df_h[(df_h['discount_rate'] == 0.06)].copy()
    df_bab = df_bab[['instance', 'policy_bab_ID', 'obj_val_bab', 'time_bab','nodes', 'nrCust']]
    df_rs = df_rs[['instance', 'policy_bab_ID', 'obj_val_bab', 'time_bab','nodes','nrCust']]
    df_rs['time_bab'] = df_rs.apply(lambda x: 3600 if x['time_bab'] >3600 else x['time_bab'], axis=1)
    df_bab['time_bab'] = df_bab.apply(lambda x: 3600 if x['time_bab'] >3600 else x['time_bab'], axis=1)
    #df_bab = df_bab.dropna(subset=['time_bab'])

    df_bab.rename(columns={'policy_bab_ID': 'policy_bab_ID_exact', 'obj_val_bab': 'obj_val_bab_exact',
                 'time_bab': 'time_bab_exact',  'nodes':'nodes_exact', 'nrCust':'nrCust_exact'}, inplace=True)
    #df_rs = df_rs.merge(df_bab, on='instance')

    for n in nr_cust:
        df_slice = df_bab[df_bab["nrCust_exact"] == n].copy()
        # df_temp_sample = df_h[df_h["nrCust"] == n].copy()
        # df_temp_rs = df_rs[df_rs["nrCust"] == n].copy()
        df_results.at[n, 'exactt'] = round(df_slice['time_bab_exact'].mean())
        df_results.at[n, 'exactn'] = round(df_slice['nodes_exact'].mean())

        # df_results.at[n, 'rst'] = round(df_temp_rs['time_bab'].mean(), 1)
        # df_results.at[n, 'rsn'] = round(df_temp_rs['nodes'].mean())
        # df_results.at[n, 'rsg'] = round(df_temp_rs['gap_av'].mean(), 2)

    for df in [df_h20, df_h100, df_h200, df_rs]:
        if df is df_rs:
            sample = "0"
        else:
            sample = str(round(df.iloc[5]['sample']))
        print("sample", sample)

        df = df[['instance','nrCust', 'time_bab', 'nodes', 'obj_val_bab', 'policy_bab_ID']]
        df['time_bab'] = df.apply(lambda x: 3600 if x['time_bab'] > 3600 else x['time_bab'], axis=1)
        df = df.merge(df_bab, on='instance')
        # df['gap_ub'] = (df['obj_val_bab'] + df['2sd_bab'] - (
        #         df_h['obj_val_bab_exact'] - df['2sd_bab_exact'])) / df_h[
        #                      'obj_val_bab'] * 100
        df['gap_av'] = (df['obj_val_bab'] - df['obj_val_bab_exact']) / df['obj_val_bab'] * 100
        df.loc[df['policy_bab_ID'] == df['policy_bab_ID_exact'], 'gap_av'] = 0

        for n in nr_cust:
            df_slice = df[df["nrCust"] == n].copy()
            df_results.at[n, 's'+sample+"t"] = round(df_slice['time_bab'].mean())
            df_results.at[n, 's'+sample+"node"] = round(df_slice['nodes'].mean())
            df_results.at[n, 's'+sample+"gap"] = round(df_slice['gap_av'].mean(), 2)

    #print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))
    #df_results = df_results[['exactt',  'exactn', "s1",  's200t', 's200node','s200gap', 's3',
    #                          's100t', 's100node','s100gap','s4',  's20t', 's20node','s20gap','s2', 's0t', 's0node','s0gap']].copy()
    df_results = df_results[['exactt', "s1", 's200t', 's200gap', 's3',
                             's100t', 's100gap', 's4', 's20t', 's20gap', 's2', 's0t',
                             's0gap']].copy()
    print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))


def experiment_bab_solution_time_classes(folder):
    #parseBAB(os.path.join(folder, "bab_VRPDO_disc_proportional_small_not_finished.txt"), folder, "bab_VRPDO_discount_proportional_small")
    df_bab = pd.read_csv(os.path.join(folder, "07_23_bab_classes_15.csv"))
    nr_cust = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for n in nr_cust:
        df_bab_temp = df_bab[(df_bab["nrCust"] == n)&(df_bab['discount_rate']==0.06)].copy()
        print(n, df_bab_temp['time_bab'].mean())

    df_bab['class'] = df_bab.apply( lambda x: 'low_determinism' if x['p_home']==0.7 else (
        'high_determinism' if x['p_home']==0.1 else (
            'high_disc' if x['discount_rate']==0.12 else (
                'low_disc' if x['discount_rate']==0.03 else 'normal'))), axis=1)

    df_bab['class_id'] = df_bab.apply(lambda x: 2 if x['p_home'] == 0.7 else (
        3 if x['p_home'] == 0.1 else (
            5 if x['discount_rate'] == 0.12 else (
                4 if x['discount_rate'] == 0.03 else  1))), axis=1)


    df_bab = df_bab.sort_values(by=['class_id'])
    # writer = pd.ExcelWriter(os.path.join(folder, "bab_VRPDO_discount_proportional_small.xlsx"), engine='openpyxl')
    # df_bab.to_excel(writer, index=False)
    # writer.save()

    sns.set()
    sns.set(font_scale=1.1)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})

    cmap = sns.color_palette("deep")
    print(cmap)
    palette = {key: value for key, value in zip(df_bab['class'].unique(), cmap)}

    print(palette)
    #dash_list = sns._core.unique_dashes(df_bab['class'].unique().size + 1)
    #style = {key: value for key, value in zip(df_bab['class'].unique(), dash_list[1:])}

    #style['normal'] =  ''  # empty string means solid
    m = np.array(['o', 'P', 'x', '^', 's'])
    lns = np.array(['-', '--', '-.', ':', 'dashdot'])
    #plt.figure(0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    #ax, fig = plt.gca(), plt.gcf()
    x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    class_name = ['C1 base case', 'C2 low incentive effectiveness' ,'C3 high incentive effectiveness', 'C4 low incentive value', 'C5 high incentive value']
    change_x = [-0.15, -0.3, 0, 0.15, 0.3]
    for class_id in [1,2,3,4,5]:
        df_temp = df_bab[df_bab['class_id']==class_id].copy()
        y = []
        yerr = []
        y_min =[]
        y_max =[]
        x_print = []
        for nr in x:
            df_temp2 = df_temp[df_temp.nrCust==nr].copy()
            x_print.append(nr+change_x[class_id-1])
            y.append(df_temp2['time_bab'].mean())
            yerr.append([df_temp2['time_bab'].mean()-df_temp2['time_bab'].min(), df_temp2['time_bab'].max()-df_temp2['time_bab'].mean()])
            y_min.append(df_temp2['time_bab'].min())
            y_max.append(df_temp2['time_bab'].max())
        yerr = np.transpose(yerr)
        #plt.scatter(x_print, y_min,  marker = '-', s= 16,  label = class_name[class_id-1])
        #plt.errorbar(x_print, y_min, yerr=None, marker = m[class_id-1],  ms=3, linestyle = '', mec = cmap[class_id-1],mfc= cmap[class_id-1], c = cmap[class_id-1],)

        plt.errorbar(x_print, y, yerr=yerr, marker = m[class_id-1],  ms=7, linestyle = "", label = class_name[class_id-1],
                     mec = cmap[class_id-1], mfc= cmap[class_id-1], c = cmap[class_id-1],  elinewidth=1, capsize=3)
        #plt.errorbar(x_print, y_max, yerr=None, marker='-', ms=4, linestyle='', mec=cmap[class_id - 1],
        #             mfc=cmap[class_id - 1], c=cmap[class_id - 1])
        #plt.errorbar(x_print, y, yerr=yerr, marker=m[class_id - 1], ms=3, linestyle=lns[class_id - 1],
        #             mec=cmap[class_id - 1], mfc=cmap[class_id - 1], c=cmap[class_id - 1],
        #             label=class_name[class_id - 1])

        #sns.lineplot(ax = ax, data=df_temp, x=x, y=y, ci=None, err_kws={'capsize': 0},markers = True  , markersize=6,   dashes=style)
    #ax = sns.scatterplot(data=df_bab, x='nrCust', y='time_bab', hue='class', palette="deep",
    #                markers=True,  style='class')
    #plt.legend(title=False, loc='upper left', bbox_to_anchor=(0.001, 0.99),
    #            labels=['low incentive effectiveness' , 'base case', 'low incentive value', 'high incentive value','high incentive effectiveness'])
    plt.legend(title=False)
    plt.ylabel("Solution time (sec)")
    ax.set(yscale="log")
    ax.set(xlabel='Problem size, n')
    ax.set_ylim(0, 100000)
    #plt.savefig(os.path.join(path_to_images, 'Solution_time_classes_2segm.eps'), transparent=False,
    #       bbox_inches='tight')
    plt.show()

def experiment_bab_solution_time_classes_pups(folder):
    #parseBAB(os.path.join(folder, "07_23_bab_classes_15.txt"), folder, "07_23_bab_classes_15")
    #df_bab = pd.read_csv(os.path.join(folder, "17_07_23_bab_classes.csv"))

    #plot 5 images: effect of n on running time for different npup
    if False:
        df_bab = df_bab[df_bab['nrCust']<19].copy()
        df_bab = df_bab[df_bab['instance_id'].isin([0,1, 2,4,5,6,8,9])].copy()

        df_bab['class'] = df_bab.apply(lambda x: 'low_determinism' if x['p_home'] == 0.7 else (
            'high_determinism' if x['p_home'] == 0.1 else (
                'high_disc' if x['discount_rate'] == 0.12 else (
                    'low_disc' if x['discount_rate'] == 0.03 else 'base'))), axis=1)

        df_bab['class_id'] = df_bab.apply(lambda x: 2 if x['p_home'] == 0.7 else (
            3 if x['p_home'] == 0.1 else (
                5 if x['discount_rate'] == 0.12 else (
                    4 if x['discount_rate'] == 0.03 else 1))), axis=1)

        df_bab.nrPup = df_bab['nrPup'].astype('int')
        df_bab['nrCust_print'] = df_bab['nrCust'] - 0.4 + (5-df_bab['nrPup']) * 0.1

        for class_id in [1, 2, 3, 4, 5]:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            df_temp = df_bab[df_bab['class_id'] == class_id].copy()
            #sns.lineplot(ax=ax, data=df_temp, x='nrCust', y='time_bab', hue = 'nrPup',  markers=True)
            sns.lineplot(ax=ax, data=df_temp, x='nrCust_print', y='time_bab', markers=True,
                         markersize=9, linewidth=1, hue='nrPup', style='nrPup',  # discount_rate   nrPup
                         palette="deep", err_style="bars", errorbar=(("pi")), err_kws={'capsize': 3},
                         dashes=False, legend=True)
            ax.set_title(df_temp['class'].iloc[0])
            ax.set(yscale="log")
            plt.show()

    #table with effect of parameters on solution time and fulfillment cost
    if True:
        #parseBAB(os.path.join(folder, "21_07_23_bab_classes.txt"), folder, "21_07_23_bab_classes")
        df_bab = pd.read_csv(os.path.join(folder, "17_07_23_bab_classes.csv"))
        df_bab = df_bab[df_bab['nrCust'] == 18].copy()


        df_results = pd.DataFrame(  columns=['class','delta', 'u', 'p', 'sp0', 'time', 'time_min',  'time_max','sp1','sp3', 'fulf_cost',
                                             'fulf_cost_min','fulf_cost_max','sp2', 'num_inc','num_inc_min','num_inc_max', 'nodes',  'num_tsps',
                                             'pruned_by_cliques_nl','pruned_by_cliques_l','pruned_by_rs_nl', 'pruned_by_rs_l',
                                             'pruned_by_insertionCost_nl', 'pruned_by_insertionCost_l', 'pruned_by_bounds_nl', 'pruned_by_bounds_l', 'pruned_n', 'pruned_n_min','pruned_n_max'])
        iter = -1
        dict_parameters = {0:[0.6, 0.06, 'C1'], 1:[0.3, 0.06,'C2'], 2:[0.9, 0.06,'C3'], 3:[0.6, 0.03,'C4'], 4:[0.6, 0.12,'C5']}
        df_bab_temp = df_bab[df_bab['instance_id'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()
        for key in dict_parameters:
            delta = dict_parameters[key][0]
            u = dict_parameters[key][1]
            if u==0.12:
                df_bab_temp = df_bab[df_bab['instance_id'].isin([0, 1, 2,3, 4, 5, 6,7,9])].copy()
            if u==0.03:
                df_bab_temp = df_bab[df_bab['instance_id'].isin([0, 1, 2,3, 4, 5, 7,8,9])].copy()
            if delta==0.3:
                df_bab_temp = df_bab[df_bab['instance_id'].isin([0, 1, 2, 4, 5,6, 8])].copy()
            df_bab_temp['pruned_nodes'] = df_bab_temp['pruned_by_cliques_nl'] + \
                                          df_bab_temp['pruned_by_cliques_l'] + \
                                          df_bab_temp['pruned_by_rs_nl'] + \
                                          df_bab_temp['pruned_by_rs_l'] + \
                                          df_bab_temp['pruned_by_insertionCost_nl'] + \
                                          df_bab_temp['pruned_by_insertionCost_l'] + \
                                          df_bab_temp['pruned_by_bounds_nl'] + df_bab_temp['pruned_by_bounds_l']
            for nr_pup in [1, 3, 5]:
                iter += 1
                print(key)
                df_slice = df_bab_temp[(df_bab_temp['p_home']==round(1-delta,1)) &(df_bab_temp['discount_rate']==u)&(df_bab_temp['nrPup']==nr_pup)].copy()
                if iter % 3==0:
                    df_results.at[iter, 'class'] = dict_parameters[key][2]
                    df_results.at[iter, 'delta'] = delta
                    df_results.at[iter, 'u'] = u*10
                df_results.at[iter, 'p'] = nr_pup
                df_results.at[iter, 'time'] = round(df_slice['time_bab'].mean())
                df_results.at[iter, 'time_min'] = round(df_slice['time_bab'].min())
                df_results.at[iter, 'time_max'] = round(df_slice['time_bab'].max())
                df_results.at[iter, 'fulf_cost'] = round(df_slice['obj_val_bab'].mean())
                df_results.at[iter, 'fulf_cost_min'] = round(df_slice['obj_val_bab'].min())
                df_results.at[iter, 'fulf_cost_max'] = round(df_slice['obj_val_bab'].max())
                df_results.at[iter, 'num_inc'] = df_slice['num_disc_bab'].mean()
                df_results.at[iter, 'num_inc_min'] = df_slice['num_disc_bab'].min()
                df_results.at[iter, 'num_inc_max'] = df_slice['num_disc_bab'].max()
                df_results.at[iter, 'nodes'] = round(df_slice['nodes'].mean())
                df_results.at[iter, 'num_tsps'] = round(df_slice['num_tsps'].mean())
                df_results.at[iter, 'pruned_by_cliques_nl'] = round(df_slice['pruned_by_cliques_nl'].mean())
                df_results.at[iter, 'pruned_by_cliques_l'] = round(df_slice['pruned_by_cliques_l'].mean())
                df_results.at[iter, 'pruned_by_rs_nl'] = round(df_slice['pruned_by_rs_nl'].mean())
                df_results.at[iter, 'pruned_by_rs_l'] = round(df_slice['pruned_by_rs_l'].mean())
                df_results.at[iter, 'pruned_by_insertionCost_nl'] = round(df_slice['pruned_by_insertionCost_nl'].mean())
                df_results.at[iter, 'pruned_by_insertionCost_l'] = round(df_slice['pruned_by_insertionCost_l'].mean())
                df_results.at[iter, 'pruned_by_bounds_nl'] = round(df_slice['pruned_by_bounds_nl'].mean())
                df_results.at[iter, 'pruned_by_bounds_l'] = round(df_slice['pruned_by_bounds_l'].mean())
                df_results.at[iter, 'pruned_n'] = round(df_slice['pruned_nodes'].mean())
                df_results.at[iter, 'pruned_n_min'] = round(df_slice['pruned_nodes'].min())
                df_results.at[iter, 'pruned_n_max'] = round(df_slice['pruned_nodes'].max())

    # df_results = df_results[['delta', 'u', 'p', 'sp0', 'time', 'time_min',  'time_max','sp1', 'fulf_cost',
    #                                          'sp2', 'num_inc', 'nodes',  'num_tsps',
    #                                          'pruned_by_cliques_nl','pruned_by_cliques_l','pruned_by_rs_nl', 'pruned_by_rs_l',
    #                                          'pruned_by_insertionCost_nl', 'pruned_by_insertionCost_l', 'pruned_by_bounds_nl', 'pruned_by_bounds_l']].copy()

    df_results = df_results[['class','delta', 'u', 'p', 'sp0', 'time', 'time_min',  'time_max','sp1', 'pruned_n', 'pruned_n_min', 'pruned_n_max','sp3',
                             'fulf_cost','fulf_cost_min','fulf_cost_max','sp2', 'num_inc','num_inc_min','num_inc_max']].copy()
    print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep='', index=False)) #
    # table with effect of parameters on solution time and fulfillment cost for n=30 customers
    if False:
        folder = os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison")

        # parseBAB(os.path.join(folder, "17_07_23_bab_classes.txt"), folder, "17_07_23_bab_classes")
        df_bab = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
        df_bab = df_bab[df_bab['nrCust'] == 30].copy()
        #df_bab = df_bab[df_bab['instance_id'].isin([0, 1, 2, 4, 5, 6, 8, 9])].copy()

        df_results = pd.DataFrame(columns=['delta', 'u', 'p',  'fulf_cost', 'num_inc'])
        iter = -1
        dict_parameters = {0: [0.6, 0.06], 1: [0.2, 0.06], 2: [0.8, 0.06], 3: [0.6, 0.03], 4: [0.6, 0.12]}
        for key in dict_parameters:
            delta = dict_parameters[key][0]
            u = dict_parameters[key][1]
            for nr_pup in [1, 3, 5]:
                iter += 1
                df_slice = df_bab[(df_bab['p_home'] == round(1 - delta, 1)) & (df_bab['discount_rate'] == u) & (
                            df_bab['nrPup'] == nr_pup)].copy()
                if iter % 3 == 0:
                    df_results.at[iter, 'delta'] = delta
                    df_results.at[iter, 'u'] = u * 10
                df_results.at[iter, 'p'] = nr_pup
                #df_results.at[iter, 'time'] = df_slice['time_bab'].mean()
                df_results.at[iter, 'fulf_cost'] = round(df_slice['obj_val_bab'].mean())
                df_results.at[iter, 'num_inc'] = df_slice['num_disc_bab'].mean()
        print(df_results.to_latex(float_format='{:0f}'.format, na_rep='', index=False))  #



def average_discount(instance_name):
    file_instance = os.path.join(path_to_data, "data", "i_VRPDO_saturation_manyPup",
                                 instance_name+".txt")
    instance = OCVRPParser.parse(file_instance)

    discount_value_av = []
    for cust in instance.customers:
        discount_value_av.append(instance.shipping_fee[cust.id])
    print(discount_value_av, instance_name, sum(discount_value_av)/len(discount_value_av))
    return sum(discount_value_av)/len(discount_value_av)

def experiment_saturation_choice_model():
    folder = os.path.join(path_to_data, "output", "VRPDO_saturation")
    #parseBAB(os.path.join(folder, "02_20_VRPDO_manypup_saturation.txt"), folder, "02_20_VRPDO_manypup_saturation")
    df_bab = pd.read_csv(os.path.join(folder, "02_20_VRPDO_manypup_saturation.csv"))
    df_bab['p_accept'] = round(1  - df_bab['p_home'], 3)
    df_bab = df_bab[['instance', 'p_accept','nrCust','p_home','discount_rate', 'obj_val_bab','policy_bab_ID']]

    df_bab['discount_value'] =df_bab.apply(
        lambda x: average_discount(x['instance']), axis=1)

    #df_bab['num_offered_disc_bab'] = df_bab.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)

    df_bab = pd.read_excel(os.path.join(folder, "02_20_VRPDO_manypup_saturation.xlsx"))
    df_bab['p_accept'] = round(1 - df_bab['p_home'], 3)
    df_bab = df_bab[['instance', 'p_accept', 'nrCust', 'p_home', 'discount_rate', 'obj_val_bab',
                     'policy_bab_ID','num_disc_bab']]

    df_bab['discount_value'] = df_bab.apply(
        lambda x: average_discount(x['instance']), axis=1)
    sns.set()
    sns.set(font_scale=1.2)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})


    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    ax2 = axes.twinx()

    sns.lineplot(ax=ax2, data=df_bab, x='discount_value', y='p_accept', linewidth=1, marker='o',
                markersize=10, color='black', err_style="bars", ci=None, err_kws={'capsize': 3}, label=r'$\Delta$',
                legend=False)
    sns.lineplot(ax=axes, data=df_bab, x='discount_value', y='obj_val_bab', linewidth=1, marker='s',
                markersize=9, color='g', err_style="bars", ci=None, err_kws={'capsize': 3},
                label='Expected fulfillment cost', legend=False)
    #sns.lineplot(ax=axes, data=df_bab, x='discount_value', y='num_offered_disc_bab', linewidth=1, marker='s',
    #             markersize=9, color='g', err_style="bars", ci=None, err_kws={'capsize': 3}, label='Number offered incentives',  legend=False)
    handles, labels = axes.get_legend_handles_labels()
    print(handles, labels)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set(ylabel=r'$\Delta$')
    axes.set(xlabel=r'd')
    axes.set(ylabel='Expected fulfillment cost')
    axes.yaxis.get_major_locator().set_params(integer=True)
    plt.savefig(os.path.join(path_to_images, 'Saturation_choice_model_manypup.eps'), transparent=False,
                bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(ax=axes, data=df_bab, x='discount_value', y='num_disc_bab', linewidth=1, marker='s',
                 markersize=9, color='g', err_style="bars", ci=None, err_kws={'capsize': 3},
                 label=None, legend=False)
    #axes.legend()
    axes.set(xlabel=r'd')
    axes.set(ylabel='Number of offered incentives')
    axes.yaxis.get_major_locator().set_params(integer=True)
    plt.savefig(os.path.join(path_to_images, 'Saturation_choice_model_number_incentives.eps'), transparent=False,
                bbox_inches='tight')
    plt.show()

def exp_profile():
    file_folder = os.path.join(path_to_data, "output", "Profile", "02_03_profile.txt")
    file_folder = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm_manyPUP", "02_08_bab_exact.txt")

    df = parseProfile(file_folder)

    df['problem_class'] = df.apply(lambda x: 'low_determinism' if (x.p_home == 0.7)and (x.discount_rate== 0.06) else (
        'high_determinism' if (x.p_home == 0.1 ) and (x.discount_rate== 0.06)else (
            'high_disc' if x.discount_rate == 0.12 and (x.p_home== 0.4 ) else (
                'low_disc' if x.discount_rate == 0.03 and (x.p_home == 0.4 ) else (
                'normal' if x.discount_rate == 0.06 and (x.p_home == 0.4 ) else 'None')))), axis=1)

    df_results = pd.DataFrame(index=[10,15,19],#index=list(range(10, 21)),
                              columns=['sp1','n_bab_av', 'n_bab_min', 'n_bab_max', 'sp3',
                                       'pr_ins_nonleaf', 'pr_ins_leaf', 'sp4',
                                       'pr_cliques_nonleaf', 'pr_cliques_leaf',
                                       'sp6','pr_rs_nonleaf','pr_rs_leaf', 'sp5',
                                       'pr_bounds_nonleaf', 'pr_bounds_leaf',
                                       'tsp_t', 'bounds_t','lb_t','branch_t'])

    for problem_class in ['normal', 'low_determinism', 'high_determinism', 'low_disc', 'high_disc']:
        df1 = df[(df.problem_class == problem_class)].copy()
        #df1 = df.copy()
        print(problem_class)
        for nrCust in [10, 15, 19]:
        #for nrCust in range(10, 21):
            df_slice = df1[(df1.nrCust == nrCust)].copy()
            # df_slice = df[(df.nrCust == nrCust)& (df.p_pup == 0.2) & (df.discount_rate ==0.06)].copy()
            df_results.at[nrCust, 'sp1'] = int(nrCust)
            df_results.at[nrCust, 'n_bab_av'] = int(df_slice['nodes'].mean() + 0.5)
            df_results.at[nrCust, 'n_bab_min'] = int(df_slice['nodes'].min() + 0.5)
            df_results.at[nrCust, 'n_bab_max'] = int(df_slice['nodes'].max() + 0.5)
            df_results.at[nrCust, 'pr_ins_nonleaf'] = round(df_slice['pr_insertionCost_nonleaf'].mean())
            df_results.at[nrCust, 'pr_ins_leaf'] = round(df_slice['pr_insertionCost_leaf'].mean())
            df_results.at[nrCust, 'pr_cliques_nonleaf'] = round(df_slice['pr_cliques_nonleaf'].mean())
            df_results.at[nrCust, 'pr_cliques_leaf'] = round(df_slice['pr_cliques_leaf'].mean())
            df_results.at[nrCust, 'pr_bounds_nonleaf'] = round(df_slice['pr_bounds_nonleaf'].mean())
            df_results.at[nrCust, 'pr_bounds_leaf'] = round(df_slice['pr_bounds_leaf'].mean())
            df_results.at[nrCust, 'pr_rs_nonleaf'] = round(df_slice['pr_rs_nonleaf'].mean())
            df_results.at[nrCust, 'pr_rs_leaf'] = round(df_slice['pr_rs_leaf'].mean())
            # df_results.at[nrCust, 'tsp_t'] = round(df_slice['tsp_time'].mean())
            # df_results.at[nrCust, 'bounds_t'] = round(df_slice['time_exact_bounds'].mean())
            # df_results.at[nrCust, 'lb_t'] = round(df_slice['time_lb_addition'].mean())
            # df_results.at[nrCust, 'branch_t'] = round(df_slice['time_branch'].mean())

        df_results = df_results[
            ['sp1','n_bab_av', 'n_bab_min', 'n_bab_max', 'sp3',
                                       'pr_ins_nonleaf', 'pr_ins_leaf', 'sp4',
                                       'pr_cliques_nonleaf', 'pr_cliques_leaf',
                                        'sp5',
                                       'pr_bounds_nonleaf', 'pr_bounds_leaf']].copy()

        # df_results = df_results[
        #     ['tsp_t', 'bounds_t', 'lb_t', 'branch_t']].copy()
        print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))
        # print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))

        print("")


def compare_enumeration_no_Gurobi(folder ):
    #parseEnumeration(os.path.join(folder, "02_06_enumeration.txt"), folder, "02_06_enumeration")
    #parseEnumeration(os.path.join(folder, "02_07_enumeration_withGurobi.txt"), folder, "02_07_enumeration_withGurobi")
    withGurobi = pd.read_csv(os.path.join(folder, "02_07_enumeration_withGurobi.csv"))
    noGurobi = pd.read_csv(os.path.join(folder, "02_13_enumeration.csv"))

    withGurobi = withGurobi[['instance', 'time_running_enum']].copy()
    withGurobi.rename(columns={'time_running_enum': 'time_running_enum_withGurobi'}, inplace=True)
    df = noGurobi.merge(withGurobi, how='left', on='instance')

    df_results = pd.DataFrame(index=list(range(10, 21)),
                              columns=['t_enum_withG','t_enum_withG_min', 't_enum_withG_max' ,'sp2', 't_enum_noG', 't_enum_noG_min', 't_enum_noG_max' ])
    for nrCust in range(10, 21):
        df_slice = df[(df.nrCust_enum == nrCust)].copy()
        if nrCust < 21:
            df_results.at[nrCust, 't_enum_noG'] = df_slice['time_running_enum'].mean()
            df_results.at[nrCust, 't_enum_noG_min'] = df_slice['time_running_enum'].min()
            df_results.at[nrCust, 't_enum_noG_max'] = df_slice['time_running_enum'].max()

            df_results.at[nrCust, 't_enum_withG'] = df_slice['time_running_enum_withGurobi'].mean()
            df_results.at[nrCust, 't_enum_withG_min'] = df_slice['time_running_enum_withGurobi'].min()
            df_results.at[nrCust, 't_enum_withG_max'] = df_slice['time_running_enum_withGurobi'].max()


    # df_results = df_results[[ 'g_opt_3600','closed_3600','sp1','tto_best', 'tto_best_min', 'tto_best_max']].copy()  'n_bab_av', 'n_bab_min', 'n_bab_max'

    df_results = df_results[
       ['t_enum_withG','t_enum_withG_min', 't_enum_withG_max' ,'sp2', 't_enum_noG', 't_enum_noG_min', 't_enum_noG_max']].copy()
    print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))
    # print(df_results.to_latex(formatters=['{:0.2f}', None, None, '{:0.1f}','{:0.5f}','{:0.1f}'], na_rep=''))



def plot_clustered_stacked(dfall, title, savename, labels=None, H="/", **kwargs):
    # title="Cost of robsut and nominal solutions",
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    colors = [(36 / 255, 6 / 255, 117 / 255), (5 / 255, 115 / 255, 97 / 255), (8 / 255, 159 / 255, 21 / 255),
              (112 / 255, 196 / 255, 9 / 255), (250 / 255, 208 / 255, 146 / 255), ]
    colors = [(24 / 255, 8 / 255, 163 / 255, 0.8), (97 / 255, 193 / 255, 9 / 255), (0 / 255, 0 / 255, 0 / 255),
              (247 / 255, 110 / 255, 80 / 255), (250 / 255, 208 / 255, 146 / 255)]
    colors = [(30 / 255, 30 / 255, 50 / 255, 0.8),
              (190 / 255, 190 / 255, 200 / 255),
              (90 / 255, 120 / 255, 90 / 255),
              (230 / 255, 230 / 255, 255 / 255),
              (140 / 255, 140 / 255, 160 / 255)]
    for df in dfall:  # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False, color=colors,
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1.6))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation=0)

    axe.minorticks_on()
    # axe.set_xticklabels([r'$\alpha =1.0$', r'$\alpha =2.0$', r'$\alpha =3.0$', r'$\alpha =5.0$', r'$\alpha =\infty$'], rotation = 0)
    axe.set_title(title)
    axe.set(xlabel='' + r'$u$')
    axe.set(ylabel='Number of offered incentives')

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    # l1 = axe.legend(h[:n_col], ['1', '2', '3', '4', '5'], bbox_to_anchor=(0.01, 0.49), loc='upper left',
    #                borderaxespad=0., title ='Pickup point' )
    if labels is not None:
        l2 = plt.legend(n, labels, loc='upper right', bbox_to_anchor=(1.00, 1.0),
                        title=r'$p$')
    #plt.gcf().text(0.1, 0.05, r'$u$', fontsize=14)
    # axe.add_artist(l1)
    plt.savefig(os.path.join(constants.PATH_TO_IMAGES, savename), bbox_inches="tight")
    plt.show()
    return axe


def plot_clustered_stacked_temp(dfall, title, savename, labels=None, H="/", **kwargs):
    # title="Cost of robsut and nominal solutions",
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    colors = [(36 / 255, 6 / 255, 117 / 255), (5 / 255, 115 / 255, 97 / 255), (8 / 255, 159 / 255, 21 / 255),
              (112 / 255, 196 / 255, 9 / 255), (250 / 255, 208 / 255, 146 / 255), ]
    colors = [(24 / 255, 8 / 255, 163 / 255, 0.8), (97 / 255, 193 / 255, 9 / 255), (0 / 255, 0 / 255, 0 / 255),
              (247 / 255, 110 / 255, 80 / 255), (250 / 255, 208 / 255, 146 / 255)]
    colors = [(30 / 255, 30 / 255, 50 / 255, 0.8),
              (190 / 255, 190 / 255, 200 / 255),
              (90 / 255, 120 / 255, 90 / 255),
              (230 / 255, 230 / 255, 255 / 255),
              (140 / 255, 140 / 255, 160 / 255)]
    for df in dfall:  # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False, color=colors,
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1.6))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation=0)
    print((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2., df.index)
    for tick in axe.xaxis.get_major_ticks():
        tick.set_pad(17)

    axe.minorticks_on()
    axe.set_xticks([-0.125, 0.126, 0.375, 0.875, 1.126, 1.375, 1.875, 2.126, 2.375, 2.875, 3.126, 3.375, 3.875, 4.126, 4.375], minor=True)
    axe.set_xticklabels([1,3,5, 1,3,5,1,3,5,1,3,5, 1,3,5], rotation=0, minor=True)
    axe.set_yticklabels(['', 5,10,15,20,25] )
    # axe.set_xticklabels([r'$\alpha =1.0$', r'$\alpha =2.0$', r'$\alpha =3.0$', r'$\alpha =5.0$', r'$\alpha =\infty$'], rotation = 0)
    axe.set_title(title)
    axe.set(xlabel='')
    axe.set(ylabel='Number of offered incentives')

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], [r'$1^{st}$', r'$2^{nd}$', r'$3^{rd}$', r'$4^{th}$', r'$5^{th}$'], bbox_to_anchor=(0.99, 0.99), loc='upper right',
                   borderaxespad=0., title ='Pickup point' )
    #if labels is not None:
    #    l2 = plt.legend(n, labels, loc='upper right', bbox_to_anchor=(1.00, 1.0),
    #                    title=r'$p$')
    plt.gcf().text(0.125, 0.074, r'$p:$', fontsize=14)

    plt.gcf().text(0.125, 0.033, r'$u:$', fontsize=14)
    axe.add_artist(l1)
    plt.savefig(os.path.join(constants.PATH_TO_IMAGES, savename), bbox_inches="tight")
    plt.show()
    return axe

def managerial_effect_delta(folder):
    #parseBABHeuristic(os.path.join(folder, "02_18_bab_nodisc_rs_30.txt"), folder, "02_18_bab_nodisc_rs_30")
    #parseBAB(os.path.join(folder, "02_13_bab_exact_managerial.txt"), folder,
    #        "02_13_bab_exact_managerial")
    #parseBAB(os.path.join(folder, "02_13_bab_exact_managerial_constDisc.txt"), folder, "02_13_bab_exact_managerial_constDisc")
    #parseBAB(os.path.join(folder, "02_13_bab_exact_managerial_DistDeptDiscount.txt"), folder, "02_13_bab_exact_managerial_DistDeptDiscount")
    #df = pd.read_csv(os.path.join(folder, "02_13_bab_exact_managerial.csv"))
    #df = pd.read_csv(os.path.join(folder, "02_15_bab_nodisc_rs_managerial.csv"))
    df  = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
    df['p_accept'] = round(1 - df['p_home'], 2)
    #df['p_accept'] = round(1 - df['p_home'] , 2)
    #df['p_accept'] = round(1 - df['p_home'] + (df['discount_rate']/3 - 0.02), 2)
    #df['p_accept'] = round(1 - df['p_home'] + (df['nrPup']*0.01 - 0.03), 2)
    #df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)
    folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_managerial")
    #df['exp_discount_cost_bab'] = df.apply(
    #df['exp_discount_cost_bab'] = df.apply(
    #lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data), axis=1)
    #df = df[df.p_home < 1].copy()
    #df = df[df.nrPup == 1].copy()
    #df = df[df.instance_id == 1].copy()
    #f = df[ df['discount_rate'].isin([0.015, 0.03, 0.06, 0.12, 0.24])].copy()
    #df = df[ df['discount_rate'].isin([ 0.06 ])].copy()
    #df = df[df.p_home==0.4].copy()
    #df = df[df.p_accept == 0.6].copy()
    # df['in_pup_bab'] = df.apply(
    #         lambda x: calculate_exp_pup_utilization(x['policy_bab_ID'], x['instance'], folder_data),
    #         axis=1)

    sns.set()
    sns.set(font_scale=1.3)
    #sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
    #sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, sharex=True)
    # effect of delta on the fulfillment cost instances with distance dependent probability
    if False:
        fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        df['discount_rate_print'] = df['discount_rate'] + (df['nrPup'] * 0.001 - 0.003)

        iter1, iter2 =-1, -1
        for (l_min, l_max) in [(1, 3), (1, 8)]:
            iter2 = -1
            iter1+=1
            for (p_min, p_av, p_max) in [(0.95,0.5, 0.05), (0.95,0.5, 0.3), (0.8,0.5, 0.05),(0.8,0.5, 0.3) ]:
                iter2 +=1
                #axes[iter1, iter2].set_title(str(l_max)+' '+str(p_min)+' '+str(p_max))
                df1 = df[(df.l_min == l_min) & (df.l_max == l_max) &
                         (df.p_min == p_min) & (df.p_max == p_max) & (df.p_av == p_av)].copy()
                sns.lineplot(ax=axes[iter1, iter2], data=df1, x='discount_rate_print', y='num_offered_disc_bab', markers=True,
                             markersize=9, linewidth=1, hue = 'nrPup', style = 'nrPup',#discount_rate   nrPup
                             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3},
                             dashes=False, legend = False ).set(title=str(l_max)+' '+str(p_min)+' '+str(p_max))

        #plt.legend( loc='lower left', bbox_to_anchor=(1.0, 1.0))
        # axes[iter1,iter2].set(xlabel='' + 'discount rate')ht
        # sns.lineplot(ax=axes, data=df, x='discount_rate_print', y='obj_val_bab', markers=True,
        #             markersize=9, linewidth=1, hue = 'nrPup', style = 'nrPup',#discount_rate   nrPup
        #             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, dashes=False)
        # sns.scatterplot(ax=axes, data=df, x='discount_rate_print', y='obj_val_bab', markers=True,
        #             hue='nrPup', style='nrPup',  # discount_rate   nrPup
        #             palette="deep", legend = False)
        # axes.set(ylabel='Expected fulfillment cost')

        # sns.lineplot(ax=axes, data=df, x='discount_rate_print', y='num_offered_disc_bab', markers=True,
        #             markersize=9, linewidth=1, hue = 'nrPup', style = 'nrPup',#discount_rate   nrPup
        #             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, dashes=False)
        # sns.scatterplot(ax=axes, data=df, x='discount_rate_print', y='num_offered_disc_bab', markers=True,
        #             hue='nrPup', style='nrPup',  # discount_rate   nrPup
        #             palette="deep", legend = False)
        # axes.set(ylabel='NUmber of offered incentives')
        plt.show()
    # effect of delta on the fulfillment cost
    if False:
        fig, axes = plt.subplots(1, 1, sharex=True)
        ax2 = axes.twinx()
        df = df[df['discount_rate'].isin([0.06])].copy()
        df = df[df['nrPup'].isin([3])].copy()
        df['obj_val_bab_print'] = df['obj_val_bab']/10
        df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)
        df['p_accept'] = round(1 - df['p_home'], 2)
        df['p_accept_print'] = df['p_accept'] + 0.02
        #df['p_accept'] = df.apply(lambda x: 1 - x['p_home'] + 0.03 if x['nrPup']==5 else 1 - x['p_home']-0.03 if x['nrPup']==3 else 1 - x['p_home'], axis=1)
        #round(1 - df['p_home'] + (df['nrPup'] * 0.01 - 0.03), 2)
        # sns.lineplot(ax=axes, data=df, x='p_accept', y='obj_val_bab_print', markers=True,
        #             markersize=9, linewidth=1,  hue = 'nrPup', style = 'nrPup',#discount_rate   nrPup
        #             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, legend = None )
        #sns.scatterplot(ax=axes, data=df, x='p_accept', y='obj_val_bab', markers=True,#hue='nrPup', style='nrPup',  # discount_rate   nrPup
        #           palette="deep", legend = False)


        #axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        #plt.legend(title='Number of pickup points', loc='lower left', bbox_to_anchor=(0.0, 0.0),
        #          labels=['1', '3','5' ])

        sns.lineplot(ax=axes, data=df, x='p_accept', y='obj_val_bab_print', markers=True,
                     linewidth=1, hue='nrPup', style='nrPup', color='b',
                     palette="deep", errorbar='sd', err_kws={'capsize': 3}, err_style="bars", legend=False, alpha=0.4,
                     label=None)
        sns.lineplot(ax=axes, data=df, x='p_accept', y='obj_val_bab_print', marker='o',
                     markersize=10, label='fulfillment cost', color='b',
                     err_style="bars", errorbar=None)

        sns.lineplot(ax=ax2, data=df, x='p_accept_print', y='num_offered_disc_bab', markers=False,
                     linewidth=1, hue='nrPup', style='nrPup',
                     palette="deep", errorbar='sd', err_kws={'capsize': 3}, err_style="bars", color='k', legend=False,
                     alpha=0.4, label=None)

        sns.lineplot(ax=ax2, data=df, x='p_accept_print', y='num_offered_disc_bab', marker='d',
                     markersize=10, label='number of incentives',
                     err_style="bars", errorbar=None, linestyle='--', color='k')
        ax2.lines[0].set_linestyle("--")
        ax2.lines[1].set_linestyle("--")
        ax2.set_ylim(None, 28)
        axes.set_ylim(None, 100)
        ax2.legend(loc=1)
        axes.legend(loc=2)
        axes.set(xlabel='' + r'$\Delta$')

        axes.set(ylabel='Expected fulfillment cost')
        ax2.set(ylabel='Number of offered incentives')

        plt.savefig(os.path.join(path_to_images, 'Total_cost_delta_nr_pups_30.eps'), transparent=False,
                    bbox_inches='tight')
        plt.show()

    # effect of delta on the number of offered incentives
    if False:
        df = df[df['discount_rate'].isin([0.06])].copy()
        df = df[df['nrPup'].isin([3])].copy()
        #df['p_accept'] = round(1 - df['p_home'] + (df['nrPup'] * 0.01 - 0.03), 2)
        df['p_accept'] = round(1 - df['p_home'] , 2)
        df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)

        sns.lineplot(ax=axes, data=df, x='p_accept', y='num_offered_disc_bab',
                     linewidth=1, markersize=7, markers=True,  hue = 'nrPup', style = 'nrPup',
                     palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, legend = None)
        #sns.scatterplot(ax=axes, data=df, x='p_accept', y='num_offered_disc_bab',
        #             markers=True,   legend = False, #hue='nrPup', style='nrPup',
        #             palette="deep")
        axes.set(xlabel='' + r'$\Delta$')
        axes.set(ylabel='Number of offered incentives')
        #plt.legend(title='Number of pickup points', loc='upper left', bbox_to_anchor=(0.0, 1.0),
        #           labels=['1', '3', '5'])
        plt.savefig(os.path.join(path_to_images, 'Num_offered_incentives_delta_nr_pups_const_disc_30.eps'), transparent=False,
                    bbox_inches='tight')
        plt.show()
    # effect of number of pickup points on the number of offered incentives
    if False:
        sns.lineplot(ax=axes, data=df, x='p_accept', y='num_offered_disc_bab', #  exp_discount_cost_bab
                     linewidth=1, markersize=7, markers=True, marker='o', hue = 'nrPup', style = 'nrPup',
                     palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3})
        axes.set(xlabel='' + r'$\Delta$')
        axes.set(ylabel='Number of offered incentives')
        #axes.set(ylabel='Expected cost of incentives')
        # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$', r'$\Delta = 1.0$'])
        #plt.savefig(os.path.join(path_to_images, 'Number_incentives_delta.eps'), transparent=False, bbox_inches='tight')
        plt.show()
    # barchart number of customers per pickup point discount dependent
    if False:
        folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30")
        nr_customers = 30
        df = df[df.nrCust == nr_customers].copy()
        df = df[df.p_accept == 0.6].copy()
        df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)
        df = df[["p_accept", "p_home", 'discount_rate', 'num_offered_disc_bab', 'nrPup', 'policy_bab_ID', 'instance']].copy()

        df['num1']=0
        df['num2']=0
        df['num3']=0
        df['num4']=0
        df['num5']=0

        # df_temp = df[df.discount_rate == 0.03].copy()
        # list_instance = df_temp['instance'].unique()
        # for instance in list_instance:
        #     OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
        #     df.loc[len(df)] = {"p_accept":0.6, "p_home":0.4, 'discount_rate':0, 'num_offered_disc_bab':30,
        #                        'nrPup':OCVRPInstance.NR_PUP, 'policy_bab_ID':2**30-1, 'instance':instance}

        for index, row in df.iterrows():
            policy = row['policy_bab_ID']
            instance = row['instance']
            OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))

            #sort pickup points by the average distance to "assigned" customers
            dict_distances = {}
            for pup in OCVRPInstance.pups:
                dict_distances[pup.id] = len(pup.closest_cust_id)
                # for i in pup.closest_cust_id:
                #    dict_distances[pup.id] += OCVRPInstance.distanceMatrix[i,  pup.id]
                #
                # dict_distances[pup.id] = dict_distances[pup.id]/len(pup.closest_cust_id)
            dict_distances = {k: v for k, v in sorted(dict_distances.items(), key=lambda item: item[1],  reverse=True)}
            num_in_each_pup = []
            print()
            for k, v in sorted(dict_distances.items(), key=lambda item: item[1],  reverse=True):
                for pup in OCVRPInstance.pups:

                    if pup.id == k:
                        print(pup.id, v, len(pup.closest_cust_id))
                        num_in_pup_temp = 0
                        for i in pup.closest_cust_id:
                            if policy & (1 << i - 1):
                                num_in_pup_temp += 1
                num_in_each_pup.append(num_in_pup_temp)
                iter = 0
                if num_in_each_pup:
                    for num in num_in_each_pup:
                        if num:
                            iter += 1
                            df.at[index, 'num' + str(iter)] = num
        if True:
            df_temp = df.copy()
            df_temp.set_index('discount_rate', inplace=True, drop=True)
            df1 = df_temp[(df_temp.nrPup == 1)].copy()
            df1 = df1[['num1', 'num2', 'num3', 'num4', 'num5']].copy()
            df1= df1.groupby(level = 'discount_rate').mean()
            df3 = df_temp[(df_temp.nrPup == 3)].copy()
            df3 = df3[['num1', 'num2', 'num3', 'num4', 'num5']].copy()
            df3 = df3.groupby(level='discount_rate').mean()
            df5 = df_temp[(df_temp.nrPup == 5)].copy()
            df5 = df5[['num1', 'num2', 'num3', 'num4', 'num5']].copy()
            df5 = df5.groupby(level='discount_rate').mean()
            plot_clustered_stacked_temp([df1, df3, df5], "", "VRPDO_30"+".eps", ['1', '3', '5'])
    # barchart number of customers per pickup point delta dependent
    if False:
        df = df[df.nrCust == 30].copy()
        df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)
        if True:
            df = df[["p_accept", "p_home", 'discount_rate', 'num_offered_disc_bab', 'nrPup', 'policy_bab_ID',
                     'instance']].copy()
            df['num1'] = 0
            df['num2'] = 0
            df['num3'] = 0
            df['num4'] = 0
            df['num5'] = 0
            folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30")
            for index, row in df.iterrows():
                policy = row['policy_bab_ID']
                instance = row['instance']
                OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))

                num_in_each_pup = []
                for pup in OCVRPInstance.pups:
                    num_in_pup_temp = 0
                    for i in pup.closest_cust_id:
                        if policy & (1 << i - 1):
                            num_in_pup_temp += 1
                    num_in_each_pup.append(num_in_pup_temp)
                iter = 0
                if num_in_each_pup:
                    for num in sorted(num_in_each_pup, reverse=True):
                        if num:
                            iter += 1
                            df.at[index, 'num' + str(iter)] = num
            # if True:
            for p_home in [0.6]:
                # for discount_rate in [0.03, 0.06, 0.12]:
                df_temp = df.copy()
                #df_temp = df[(df.p_home == p_home)].copy()
                df_temp.set_index('p_accept', inplace=True, drop=True)
                #df_temp.set_index('discount_rate', inplace=True, drop=True)
                df1 = df_temp[(df_temp.nrPup == 1)].copy()
                df1 = df1[['num1', 'num2', 'num3', 'num4', 'num5']].copy()
                #df1 = df1.groupby(level='discount_rate').mean()
                df1 = df1.groupby(level='p_accept').mean()
                df3 = df_temp[(df_temp.nrPup == 3)].copy()

                df3 = df3[['num1', 'num2', 'num3', 'num4', 'num5']].copy()
                #df3 = df3.groupby(level='discount_rate').mean()
                df3 = df3.groupby(level='p_accept').mean()
                df5 = df_temp[(df_temp.nrPup == 5)].copy()
                df5 = df5[['num1', 'num2', 'num3', 'num4', 'num5']].copy()
                df5 = df5.groupby(level='p_accept').mean()
                #df5 = df5.groupby(level='discount_rate').mean()
                print()
                plot_clustered_stacked([df1, df3, df5], "", "barchart_p_accept_30" + ".eps", ['1', '3', '5'])
    # Impact of incenitve rate on the expected fulfillment cost hue nrPup
    if False:
        nr_customers = 30
        df = df[df.nrCust == nr_customers].copy()
        df = df[df.p_accept == 0.6].copy()

        df = df[df['nrPup'].isin([3])].copy()
        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
                                                   x['obj_val_uniform'])/10, axis=1)
        df['discount_rate_print'] = df['discount_rate'] + (df['nrPup'] * 0.001 - 0.007)
        df['discount_rate_print_nodisc'] = df['discount_rate']  #- 0.004
        df['obj_val_nodisc'] = df['obj_val_nodisc']/10
        df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)

        df = df[df.nrPup == 3].copy()
        #sns.lineplot(data=df, x='discount_rate_print_nodisc', y='obj_val_nodisc', marker='d', markersize=10,
        #             err_style="bars", errorbar=('ci', 0), linestyle='-.', label='0', color='k')
        # sns.lineplot(ax=axes, data=df, x='discount_rate_print_nodisc', y='obj_val_nodisc', markers=False,
        #              linewidth=1, errorbar='sd', err_kws={'capsize': 3}, linestyle='', err_style="bars", legend=False, alpha=0.4,
        #              color='k')
        fig, axes = plt.subplots(1, 1, sharex=True)
        ax2 = axes.twinx()

        sns.lineplot(ax=axes, data=df, x='discount_rate_print', y='objValPrint', markers=False,
                     linewidth=1, hue='nrPup', style='nrPup',color='b',
                    palette="deep", errorbar='sd', err_kws={'capsize': 3}, err_style="bars", legend = False, alpha = 0.4, label=None )

        sns.lineplot(ax=axes, data=df, x='discount_rate_print', y='objValPrint', marker='o',
                     markersize=10, label='fulfillment cost',color='b',
                     err_style="bars", errorbar=None)


        sns.lineplot(ax=ax2, data=df, x='discount_rate_print_nodisc', y='num_offered_disc_bab', markers=False,
                     linewidth=1, hue='nrPup', style='nrPup',
                     palette="deep", errorbar='sd', err_kws={'capsize': 3}, err_style="bars",color='k', legend=False, alpha=0.4, label=None)

        sns.lineplot(ax=ax2, data=df, x='discount_rate_print_nodisc', y='num_offered_disc_bab',marker='d', markersize=10, label='number of incentives',
                      err_style="bars", errorbar=None, linestyle='-.',color='k')
        ax2.lines[0].set_linestyle("--")
        ax2.lines[1].set_linestyle("--")
        #lines, labels = axes.get_legend_handles_labels()
        #lines2, labels2 = ax2.get_legend_handles_labels()
        #ax2.legend(lines + lines2, labels + labels2)
        axes.set_xticks([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24])
        ax2.set_ylim(-5, None)
        axes.set_ylim(50, None)
        #plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), title = 'Number of pickup points', ncol = 2)
        #plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), title=False, ncol=2)
        ax2.legend(loc=4)
        axes.legend(loc=3)
        #lines1, labels1 = axes.get_legend_handles_labels()
        #print(lines1, labels1 )
        #axes.legend(loc=3, labels = ['fulfillment cost'])
        axes.set(xlabel='' +  r'$u$')
        axes.set(ylabel='Expected fulfillment cost')
        ax2.set(ylabel='Number of offered incentives')

        plt.savefig(os.path.join(path_to_images, 'Expected_fulfillment_cost_disc_rate_30.eps'), transparent=False,
                   bbox_inches='tight')
        plt.show()
    # Impact of incenitve rate on the number of offered icnetnives
    if False:
        nr_customers = 30
        df = df[df.nrCust == nr_customers].copy()
        df = df[df.p_accept == 0.6].copy()

        df = df[df['nrPup'].isin([3])].copy()
        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
                                                   x['obj_val_uniform'])/10, axis=1)
        df['discount_rate_print'] = df['discount_rate'] + (df['nrPup'] * 0.001 - 0.003)
        df['discount_rate_print_nodisc'] = df['discount_rate']  - 0.004
        df['obj_val_nodisc'] = df['obj_val_nodisc']/10
        df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)



        # sns.lineplot(ax=axes, data=df, x='discount_rate_print', y='num_offered_disc_bab', markers=False,
        #              linewidth=1, hue='nrPup', style='nrPup',
        #             palette="deep", errorbar='pi', err_kws={'capsize': 3}, err_style="bars", legend = False, alpha = 0.4 )

        sns.lineplot(ax=axes, data=df, x='discount_rate_print', y='num_offered_disc_bab', markers=True,
                   markersize=10, linewidth=1, hue = 'nrPup', style = 'nrPup',legend = False,
                   palette="deep",    err_style=None )

        # ax2 = axes.twinx()
        # sns.lineplot(ax=ax2, data=df, x='discount_rate_print_nodisc', y='num_offered_disc_bab', markers=False,
        #              linewidth=1, hue='nrPup', style='nrPup',
        #              palette="deep", errorbar='sd', err_kws={'capsize': 3}, err_style="bars",color='k', legend=False, alpha=0.4)
        #
        # sns.lineplot(ax=ax2, data=df, x='discount_rate_print_nodisc', y='num_offered_disc_bab',marker='d', markersize=10,
        #               err_style="bars", errorbar=('ci', 0), linestyle='-.', label=None, color='k')
        # ax2.lines[0].set_linestyle("--")
        # ax2.lines[1].set_linestyle("--")
        # lines, labels = axes.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2)

        axes.set_xticks([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24])
        #ax2.set_ylim(0, 40)
        #plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), title = 'Number of pickup points', ncol = 2)
        axes.set(xlabel='' +  r'$u$')
        axes.set(ylabel='Number of offered incentives')
        #ax2.set(ylabel='Number of offered incentives')

        plt.savefig(os.path.join(path_to_images, 'Number_offered_incentives_disc_rate_30.eps'), transparent=False,
                   bbox_inches='tight')
        plt.show()

def sensitivity_comparison_nodisc_rs(folder):
    #parseBAB_RS_NODISC(os.path.join(folder, "02_18_bab_nodisc_rs_15.txt"), folder, "02_18_bab_nodisc_rs_15")
    #parseBABHeuristic(os.path.join(folder, "02_18_bab_nodisc_rs_30.txt"), folder, "02_18_bab_nodisc_rs_30")
    df = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
    #df['instance_type'] =  df['instance'].apply(lambda x: str(x).split('_')[0])
    #df = df[df.instance_type=='VRPDODistDepAccept2'].copy()
    #df = df[df.instance_type == 'VRPDO'].copy()
    #df['p_accept'] = round(1 - df['p_home'] + (df['nrPup'] * 0.01 - 0.03), 2)
    df['p_accept'] = round(1 - df['p_home'], 2)
    df = df[df.nrPup == 1].copy()

    df_copy = df[['instance', 'nrCust', 'policy_bab_ID', 'obj_val_bab']].copy()
    #parseBAB_REMOTE(os.path.join(folder, "remote_30.txt"), folder, "remote_30")
    df_remote = pd.read_csv(os.path.join(folder, "remote_30.csv"))
    df_remote = df_remote.merge(df_copy, on='instance')
    df_remote['gap_remote'] = 100 * (df_remote['obj_val_remote'] - df_remote['obj_val_bab']) / df_remote['obj_val_bab']


    df_results = pd.DataFrame(columns=['p_accept', 'discount_rate','nrPup', 'algo','savings','p_home' ])

    iter = 0
    for index, row in df.iterrows():
        for i in range(4):
            df_results.at[iter+i, 'instance'] = row['instance']
            df_results.at[iter + i, 'p_home'] = row['p_home']
            df_results.at[iter+i, 'p_accept'] = row['p_accept']  -0.02 + 0.02*i
            df_results.at[iter+i, 'discount_rate'] = row['discount_rate']
            df_results.at[iter + i, 'nrPup'] = row['nrPup']
            df_results.at[iter+i, 'discount_rate_print'] = row['discount_rate']  + 0.005*i-0.005
            if i==0:
                df_results.at[iter+i, 'algo'] = 1
                df_results.at[iter+i, 'savings'] =  max(row['gap_nodisc'], 0)
            elif i==1:
                if row['nrPup'] == 3:
                    pass
                else:
                    df_results.at[iter+i, 'algo'] = 2
                    df_results.at[iter+i, 'savings'] =  max(row['gap_rs'], 0)
            elif i==2:
                if row['nrPup'] == 3:
                    pass
                else:
                    df_results.at[iter+i, 'algo'] = 0
                    df_results.at[iter+i, 'savings'] =  max(row['gap_uniform'], 0)
            # elif i==3:
            #     if row['nrPup'] == 3:
            #         pass
            #     else:
            #         df_results.at[iter+i, 'algo'] = 3
            #         df_results.at[iter+i, 'savings'] = max(df_remote[df_remote['instance']==row['instance']]['gap_remote'].mean(), 0)
        iter += 3
    df_results.sort_values("algo")
    sns.set()
    sns.set(font_scale=1.2)
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})

    if True:
        # df_temp = df_results.copy()
        #df_temp = df_results[df_results.algo == 'rs'].copy()
        if False: # impact of delta on savings  and different agorithms

            df_temp = df_results[df_results.discount_rate == 0.06].copy()
            df_temp['discount_rate_print'] = df_temp['discount_rate'] + (df_temp['nrPup'] * 0.001 - 0.003)
            fig, axes = plt.subplots(1, 3, sharex=True,  figsize=(15, 5))
            iter = 0
            for algo in ['ALL', 'DI', 'NOI']:
                axes[iter].title.set_text(algo)
                df1 = df_temp[df_temp.algo == algo].copy()
                sns.lineplot(ax=axes[iter], data=df1, x='p_accept', y='savings', markers=True,
                             markersize=11, linewidth=1, hue='nrPup', style='nrPup',  # algo   nrPup
                             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 5}, legend = False)
                axes[iter].set(xlabel='' + r'$\Delta$')
                axes[iter].set(xlabel='' + r'$\Delta$')
                axes[iter].set_ylim(-0.2, None)
                iter += 1
            axes[0].set(ylabel='Savings (%)')
            plt.legend(title='Number of pickup points', loc='upper left', bbox_to_anchor=(0.0, 1.0),
                      labels=['1', '3', '5'])
            plt.savefig(os.path.join(path_to_images, 'Savings_ALL_NOI_DI.eps'), transparent=False,
                      bbox_inches='tight')
            plt.show()
        # impact of delta on savings for 1 pickup point and different agorithms
        if False:

            df_temp = df_results[df_results.discount_rate == 0.06].copy() #try 0.12
            fig, axes = plt.subplots(1, 1, sharex=True)
            df1 = df_temp.copy()
            sns.lineplot(ax=axes, data=df1, x='p_accept', y='savings', markers=["o","^","s"],
                         markersize=10,
                         hue='algo', style='algo',  # algo   nrPup
                         palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 5})
            # sns.scatterplot(ax=axes, data=df1, x='p_accept', y='savings', markers=True,
            #                hue='algo', style='algo',  #    nrPup algo
            #                palette="deep", legend=False)
            axes.set(xlabel='' + r'$\Delta$')
            axes.set(ylabel='Savings (%), log scale with 0-linearization')
            plt.legend(title=False)
            axes.set(yscale="symlog")
            axes.set_ylim(None, 110)
            plt.legend(title=False,    labels=['ALL', 'NOI', 'DI'],  markerscale=1.2)

            # lgnd = axes.legend(title=False, handles=handles, fontsize=16, markerscale=1.9,
            #                    labels=['ALL', 'NOI', 'DI', 'R'], loc='upper right',
            #                    bbox_to_anchor=(1.0, 1.0))
            plt.savefig(os.path.join(path_to_images, 'Savings_delta_ALL_NOI_DI_R_30.eps'), transparent=False,
                        bbox_inches='tight')
            plt.show()
        if False:
            df_temp = df_results[df_results.nrPup == 1].copy()
            for p_home in [0.4]:
                fig, axes = plt.subplots(1, 1, sharex=True)
                df1 = df_temp[df_temp.p_home == p_home].copy()
                sns.lineplot(ax=axes, data=df1, x='discount_rate', y='savings', markers=True,
                             markersize=11, linewidth=1, hue='algo', style='algo',  # algo   nrPup
                             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 5})
                #sns.scatterplot(ax=axes, data=df1, x='p_accept', y='savings', markers=True,
                #                hue='algo', style='algo',  #    nrPup algo
                #                palette="deep", legend=False)
                axes.set(xlabel='Discount rate, ' + r'$u$')
                axes.set(ylabel='Savings (%)')
                plt.legend(title =False)
                #axes.set(yscale="log")
                #plt.legend(title='Number of pickup points', loc='lower left', bbox_to_anchor=(0.0, 0.0),
                #           labels=['1', '3', '5'])
                plt.savefig(os.path.join(path_to_images, 'Savings_disc_rate_1pup_ALL_NOI_DI_15.eps'), transparent=False,
                          bbox_inches='tight')
                plt.show() #Effect of discount rate on savings for nrPup=1
        if False: # impact of discount cost on savings for 1 pickup point and different agorithms
            df_temp = df_results[df_results.nrPup == 1].copy()
            df_temp = df_results[df_results.p_accept == 0.6].copy()

            for p_accept in [0.6]:
                fig, axes = plt.subplots(1, 1, sharex=True)
                df1 = df_temp.copy()
               # df1 = df_temp[df_temp.p_accept == p_accept].copy()
                sns.lineplot(ax=axes, data=df1, x='discount_rate_print', y='savings', markers=True,
                             markersize=11, linewidth=1, hue='algo', style='algo',  # algo   nrPup
                             palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 5})
                #sns.scatterplot(ax=axes, data=df1, x='p_accept', y='savings', markers=True,
                #                hue='algo', style='algo',  #    nrPup algo
                #                palette="deep", legend=False)
                axes.set(xlabel='discount rate, ' + r'$u$')
                axes.set(ylabel='Savings (%)')
                plt.legend(title = 'None')
                #axes.set(yscale="log")
                plt.legend(title=False )
                plt.savefig(os.path.join(path_to_images, 'Savings_ALL_NOI_DI_discount_rate.eps'), transparent=False,
                          bbox_inches='tight')
                plt.show()

def large_exp(folder):
    #parseBABHeuristic(os.path.join(folder, "02_16_bab_nodisc_large.txt"), folder, "03_10_bab_nodisc_large")
    #parseBAB_RS_NODISC(os.path.join(folder, "temp.txt"), folder, "temp")
    #parseBABHeuristic(os.path.join(folder, "bab_large_temp.txt"), folder, "bab_large_temp")
    # df = pd.read_csv(os.path.join(folder, "bab_large_temp.csv"))
    # #df = pd.read_csv(os.path.join(folder, "03_10_bab_nodisc_large.csv"))
    # df['cost_per_order'] = df.apply(lambda x:  min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_uniform'],
    #                                                x['obj_val_nodisc']) / x['nrCust'],
    #                                    axis=1)
    # df['cost_per_order_nodisc'] = df.apply(  lambda x:  x['obj_val_nodisc'] / x['nrCust'], axis=1)
    # df['cost_per_order_all'] = df.apply(lambda x: x['obj_val_uniform'] / x['nrCust'], axis=1)
    # df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
    #                                            x['obj_val_uniform'])/10, axis=1)

    sns.set()
    sns.set(font_scale=1.4)
    #sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
    #sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        # impact of the problem size on the objective value hue number of pups
    if False: #impact of the problem size on the objective value hue number of pups
        df = df[df['nrCust'].isin([20, 25, 30, 35, 40, 45, 50])].copy()
        df = df[df.discount_rate == 0.06].copy()
        df = df[['nrCust','nrPup', 'objValPrint', 'time_bab' ]].copy()
        #parseBAB(os.path.join(folder, "03_10_bab_large_10_15.txt"), folder,  "03_10_bab_large_10_15")
        df_small = pd.read_csv(os.path.join(folder, "03_10_bab_large_10_15.csv"))
        df_small['objValPrint'] = df_small['obj_val_bab']/10
        df_small = df_small[['nrCust','nrPup', 'objValPrint','time_bab' ]].copy()
        df = pd.concat([df, df_small])

        df['nrCustPrint'] = df['nrCust']+ df['nrPup']*0.3-1

        #sns.scatterplot(ax=axes, data=df, x='nrCustPrint', y='objValPrint', markers=True,
        #              linewidth=1, hue='nrPup', style='nrPup',  # discount_rate   nrPup
        #             palette="deep", legend = False)time_bab   objValPrint
        sns.lineplot(ax=axes, data=df, x='nrCustPrint', y='time_bab', markers=True,
                     markersize=12, linewidth=1, hue='nrPup', style='nrPup',  # discount_rate   nrPup
                     palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3})
        #sns.lineplot(ax=axes, data=df, x='nrCustNodisc', y='obj_val_nodisc', markers=True,
        #           markersize=9, linewidth=1,   # discount_rate   nrPup
        #            err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, label='0')
        #sns.lineplot(ax=axes, data=df, x='nrCust', y='obj_val_uniform', markers=True,
        #            markersize=9, linewidth=1,  # discount_rate   nrPup
        #            err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, label='all')
        axes.set(xlabel='' + 'Problem size, n')
        # # axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set(ylabel='Expected fulfillment cost')
        #axes.set(yscale="log")
        plt.legend(title='Number of pickup points', loc='lower right', bbox_to_anchor=(1.0, 0.0))
        # plt.savefig(os.path.join(path_to_images, 'Total_cost_delta_nr_customers_nr_pups.eps'), transparent=False,
        #             bbox_inches='tight')
        plt.show()
    # impact of the problem size on the savings of BAB in comparison to rs and nodsic
    if False:
        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
                                                   x['obj_val_uniform']), axis=1)
        df['gap_nodisc'] = 100*(df['obj_val_nodisc'] - df['objValPrint'])/df['objValPrint']
        df['gap_rs'] = 100*(df['obj_val_rs'] - df['objValPrint']) / df['objValPrint']
        df['gap_uniform'] = 100*(df['obj_val_uniform'] - df['objValPrint']) / df['objValPrint']
        df_results = pd.DataFrame(columns=['p_accept','p_home', 'discount_rate', 'nrPup', 'nrCust', 'algo', 'savings'])
        iter = 0
        for index, row in df.iterrows():
            for i in range(3):
                df_results.at[iter + i, 'instance'] = row['instance']
                df_results.at[iter + i, 'p_home'] = row['p_home']
                #df_results.at[iter + i, 'p_accept'] = row['p_accept']
                df_results.at[iter + i, 'discount_rate'] = row['discount_rate']
                df_results.at[iter + i, 'nrPup'] = row['nrPup']
                df_results.at[iter + i, 'nrCust'] = row['nrCust'] + i*0.7 - 0.7
                if i == 0:
                    if   row['nrPup'] ==1:
                        pass
                    else:
                        df_results.at[iter + i, 'algo'] = 'NOI'
                        df_results.at[iter + i, 'savings'] = max(row['gap_nodisc'], 0)
                elif i == 1:
                    df_results.at[iter + i, 'algo'] = 'DI'
                    df_results.at[iter + i, 'savings'] = max(row['gap_rs'], 0)
                elif i == 2:
                    df_results.at[iter + i, 'algo'] = 'ALL'
                    df_results.at[iter + i, 'savings'] = max(row['gap_uniform'], 0)
            iter += 3
        df_results = df_results[df_results['discount_rate'].isin([0.06])].copy()
        df_results = df_results[df_results['nrPup'].isin([1,3])].copy()


        #sns.scatterplot(ax=axes, data=df_results, x='nrCust', y='savings', markers=True,
        #                linewidth=1, hue='algo', style='algo',  # discount_rate   nrPup
        #                palette="deep", legend=False)
        sns.lineplot(ax=axes, data=df_results, x='nrCust', y='savings', markers=True,
                     markersize=10, linewidth=1, hue='algo', style='algo',  # discount_rate   nrPup
                     palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3})

        axes.set(xlabel='' + 'Problem size, n')
        # # axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set(ylabel='Savings (%)')
        lgnd = plt.legend(title=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        #lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        plt.savefig(os.path.join(path_to_images, 'heuristic_improvement.eps'), transparent=False,
                    bbox_inches='tight')
        plt.show()
    # impact of the problem size on the savings of BAB in comparison to rs and nodsic and remote
    if False:
        #parseBABHeuristic(os.path.join(folder, "bab_large_temp.txt"), folder, "bab_large_temp")
        df = pd.read_csv(os.path.join(folder, "bab_large_temp.csv"))
        #parseBAB_REMOTE(os.path.join(folder, "08_03_remote.txt"), folder, "08_03_remote")
        df_remote = pd.read_csv(os.path.join(folder, "08_03_remote.csv"))
        df_remote = df_remote[['instance', 'nrCust_rem',"nrPup_rem", 'p_home_rem', 'discount_rate_rem', 'obj_val_remote', 'policy_remote_ID', 'instance_id_rem']].copy()
        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
                                                   x['obj_val_uniform']), axis=1)
        df_copy = df[['instance','nrCust', 'policy_bab_ID','objValPrint']].copy()
        df_remote = df_remote.merge(df_copy, on='instance')
        df_remote['gap_remote'] = 100 * (df_remote['obj_val_remote'] - df_remote['objValPrint']) / df_remote['objValPrint']

        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
                                                   x['obj_val_uniform']), axis=1)
        df['gap_nodisc'] = 100 * (df['obj_val_nodisc'] - df['objValPrint']) / df['objValPrint']
        df['gap_rs'] = 100 * (df['obj_val_rs'] - df['objValPrint']) / df['objValPrint']
        df['gap_uniform'] = 100 * (df['obj_val_uniform'] - df['objValPrint']) / df['objValPrint']
        df_results = pd.DataFrame(columns=['p_accept', 'p_home', 'discount_rate', 'nrPup', 'nrCust', 'algo', 'savings'])
        iter = 0
        for index, row in df.iterrows():
            if row['nrCust'] in [10, 15, 20, 25,30,35,40,45,50]:
                for i in range(4):
                    df_results.at[iter + i, 'instance'] = row['instance']
                    df_results.at[iter + i, 'p_home'] = row['p_home']
                    # df_results.at[iter + i, 'p_accept'] = row['p_accept']
                    df_results.at[iter + i, 'discount_rate'] = row['discount_rate']
                    df_results.at[iter + i, 'nrPup'] = row['nrPup']
                    df_results.at[iter + i, 'nrCust'] = row['nrCust']  + i * 0.9 - 0.9
                    if i == 0:
                        if row['nrPup'] == 1:
                            pass
                        else:
                            df_results.at[iter + i, 'algo'] = 1
                            df_results.at[iter + i, 'savings'] = max(row['gap_nodisc'], 0)
                    elif i == 1:
                        if row['nrPup'] == 3:
                            pass
                        else:
                            df_results.at[iter + i, 'algo'] = 2
                            df_results.at[iter + i, 'savings'] = max(row['gap_rs'], 0)
                    elif i == 2:
                        if row['nrPup'] == 3:
                            pass
                        else:
                            df_results.at[iter + i, 'algo'] = 0
                            df_results.at[iter + i, 'savings'] = max(row['gap_uniform'], 0)
                    # elif i==3:
                    #
                    #     df_results.at[iter + i, 'algo'] = 3
                    #     df_results.at[iter + i, 'savings'] = max(df_remote[df_remote['instance']==row['instance']]['gap_remote'].mean(), 0)
                iter += 3
        df_results = df_results[df_results['discount_rate'].isin([0.06])].copy()
        df_results = df_results[df_results['nrPup'].isin([1,3])].copy()
        df_results.sort_values("algo")
        # sns.scatterplot(ax=axes, data=df_results, x='nrCust', y='savings', markers=True,
        #                linewidth=1, hue='algo', style='algo',  # discount_rate   nrPup
        #                palette="deep", legend=False)
        sns.lineplot(ax=axes, data=df_results, x='nrCust', y='savings', markers=True,
                    markersize=12, hue='algo', style='algo',  # discount_rate   nrPup
                    palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, alpha = 0.5)
        sns.lineplot(ax=axes, data=df_results, x='nrCust', y='savings', markers=["o","^","s"],
                     markersize=12, hue='algo', style='algo',  # discount_rate   nrPup
                     palette="deep", ci=None)
        #plt.legend(title=False, labels=['ALL', 'NOI', 'DI', 'R'])
        #df_results.dropna(subset=['savings'], inplace = True)
        #axes = sns.barplot(data=df_results, x='nrCust', y='savings', hue = 'algo')

        axes.set_xticks([10, 15, 20, 25,30,35,40,45,50])

        axes.set(xlabel='' + 'Problem size, n')
        # # axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set(ylabel='Savings (%)')
        handles, labels = axes.get_legend_handles_labels()
        handles = [handles[3], handles[4], handles[5]] #handles[7]
        #axes.legend(handles, labels,  loc='upper right', bbox_to_anchor=(1.0, 1.0))
        lgnd = axes.legend( title=False,  handles=handles, fontsize=16,markerscale=1.9,
                            labels=[ 'ALL', 'NOI', 'DI'], loc='upper right',
                            bbox_to_anchor=(1.0, 1.0))
        #plt.legend(title=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        plt.savefig(os.path.join(path_to_images, 'heuristic_improvement.eps'), transparent=False,
                   bbox_inches='tight')
        plt.show()
    #parse remote
    if False:
        df = pd.read_csv(os.path.join(folder, "bab_large_temp.csv"))
        parseBAB_REMOTE(os.path.join(folder, "remote.txt"), folder, "08_03_remote_full")
        df_remote = pd.read_csv(os.path.join(folder, "08_03_remote.csv"))

        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_rs'], x['obj_val_nodisc'],
                                                   x['obj_val_uniform']), axis=1)
        df_copy = df[['instance', 'nrCust', 'policy_bab_ID', 'objValPrint']].copy()
        df_remote = df_remote.merge(df_copy, on='instance')
        df_remote['same_remote'] =''
        df_remote['babbin'] = ''
        df_remote['remotebin'] = ''
        for index, row in df_remote.iterrows():
            same_remote = 0

            try:
                df_remote['babbin'].at[index] = bin(int(row['policy_bab_ID']))
                df_remote['remotebin'].at[index] = bin(int(row['policy_remote_ID']))
                for cust in range(row['nrCust']):
                    if int(row['policy_bab_ID'])& (1<< cust) == int(row['policy_remote_ID'])& (1<< cust) :
                        same_remote += 1
                df_remote.at[index, 'same_remote'] = same_remote/row['nrCust']
            except:
                pass
        df_remote = df_remote.dropna(axis=0)
        df_results = pd.DataFrame(index=df_remote.nrCust.unique(), columns=['same_remote'])
        for n in df_remote.nrCust.unique():
            df_slice = df_remote[(df_remote.nrCust == n)].copy()
            df_results.at[n, 'same_remote'] = round(df_slice['same_remote'].mean(), 2)
        print(df_results.to_latex(float_format='{:0.2f}'.format, na_rep=''))
        print("")

    if True:
        df = pd.read_csv(os.path.join(folder, "17_07_23_bab_classes.csv"))
        #parseBAB_REMOTE(os.path.join(folder, "28_07_23_remote.txt"), folder, "28_07_23_remote")
        df_remote = pd.read_csv(os.path.join(folder, "28_07_23_remote.csv"))
        df_remote[['nrCust_rem', "p_home_rem", "nrPup_rem", 'discount_rate_rem']] = \
            df_remote[['nrCust_rem', "p_home_rem", "nrPup_rem", 'discount_rate_rem']].apply(pd.to_numeric)
        df_remote = df_remote[
            (df_remote.nrCust_rem == 18) & (df_remote.nrPup_rem == 3) & (df_remote.discount_rate_rem == 0.06) & (
                        df_remote.p_home_rem == 0.4)]

        df_copy = df[['instance', 'nrCust', 'policy_bab_ID', 'obj_val_bab']].copy()
        df_remote = df_remote.merge(df_copy, on='instance')
        df_remote['same_remote'] =''
        df_remote['babbin'] = ''
        df_remote['remotebin'] = ''

        for index, row in df_remote.iterrows():
            print(bin(row['policy_remote_ID']), bin(row['policy_bab_ID']), (row['obj_val_remote'] - row['obj_val_bab'])/row['obj_val_remote'])
            same_remote = 0
            try:
                df_remote['babbin'].at[index] = bin(int(row['policy_bab_ID']))
                df_remote['remotebin'].at[index] = bin(int(row['policy_remote_ID']))
                for cust in range(row['nrCust']):
                    if int(row['policy_bab_ID'])& (1<< cust) == int(row['policy_remote_ID'])& (1<< cust) :
                        same_remote += 1
                df_remote.at[index, 'same_remote'] = same_remote/row['nrCust']
            except:
                pass
        df_remote = df_remote.dropna(axis=0)
        df_results = pd.DataFrame(index=df_remote.nrCust.unique(), columns=['same_remote'])
        for n in df_remote.nrCust.unique():
            df_slice = df_remote[(df_remote.nrCust == n)].copy()
            df_results.at[n, 'same_remote'] = round(df_slice['same_remote'].mean(), 2)
        print(df_results.to_latex(float_format='{:0.2f}'.format, na_rep=''))
        print("")

def managerial_location(folder):
    #parseBABHeuristic(os.path.join(folder, "02_18_bab_nodisc_rs_30.txt"), folder, "02_18_bab_nodisc_rs_30")
    # effect of centrality and delta on the distribution of  offered incentives
    if True:
        df = pd.read_csv(os.path.join(folder, "17_07_23_bab_classes.csv"))
        folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPup_classes")
        df = df[(df.nrPup == 3) & (df.nrCust == 18)].copy()
        # df['class'] = df.apply(lambda x: 'low_determinism' if x['p_home'] == 0.7 else (
        #     'high_determinism' if x['p_home'] == 0.1 else (
        #         'high_disc' if x['discount_rate'] == 0.12 else (
        #             'low_disc' if x['discount_rate'] == 0.03 else 'base'))), axis=1)

        df['class_id'] = df.apply(lambda x: 2 if x['p_home'] == 0.7 else (
            3 if x['p_home'] == 0.1 else (
                5 if x['discount_rate'] == 0.12 else (
                    4 if x['discount_rate'] == 0.03 else 1))), axis=1)
        df = df[["class_id", 'discount_rate', 'nrPup', 'policy_bab_ID', 'instance']].copy()

        sns.set()
        sns.set(font_scale=1.2)
        sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        sns.set_style('ticks', {"xtick.bottom": False, "ytick.direction": "in"})
        fig, axes = plt.subplots(1, 1, sharey=True, sharex=True)
        df['instance_type'] = df['instance'].apply(lambda x: int(str(x).split('_')[10]))
        # df = df[df['instance_type'].isin([0,2,3,4])].copy()
        distances = [ 4,8,16]
        #distances = [4, 6, 16]
        #distances = [3.5, 7, 10.5,16]
        #distances = [9,12, 15,20]
        dict_distances = {}
        class_bins = [1, 2, 3, 4, 5]
        bottom_dict = {}
        for distance in distances:
            bottom_dict[distance] = [0] * (len(class_bins))
        for distance in distances:
            dict_distances[distance] = [0] * (len(class_bins))

        colors = [(30 / 255, 30 / 255, 50 / 255, 0.8),
                  (190 / 255, 190 / 255, 200 / 255),
                  #(90 / 255, 120 / 255, 90 / 255),
                 (230 / 255, 230 / 255, 255 / 255),
                  (140 / 255, 140 / 255, 160 / 255)]
        iter = -1
        list_farness = []

        df_temp = df[df.class_id == 1].copy()
        for index, row in df_temp.iterrows():
            instance = row['instance']
            OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
            df1 = df[(df.instance_type == row['instance_type'])].copy()
            temp_full = []
            for class_id in class_bins:
                temp = []
                df1 = df[(df.class_id == class_id) ].copy()
                policy = df1['policy_bab_ID'].iloc[0]
                for cust in OCVRPInstance.customers:
                    farness = OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id] / 10
                    if farness < distances[0]:
                        if policy & (1 << cust.id - 1):
                            temp.append(cust.id)
                temp_full.append(temp)
            print(temp_full)




        for class_id in class_bins:
            iter += 1
            if class_id ==0:
                df1 = df[df.class_id == 1].copy()
            else:
                df1 = df[df.class_id == class_id].copy()
            number_instances = 0
            for index, row in df1.iterrows():
                number_instances += 1
                instance = row['instance']
                OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
                temp =[]
                for cust in OCVRPInstance.customers:

                    # distancesf = [OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.customers if
                    #                    j is not cust] + [OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.pups] + [OCVRPInstance.distanceMatrix[cust.id, OCVRPInstance.depot.id]]
                    # farness = (sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.customers if
                    #                    j is not cust) + \
                    #                sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.pups) + \
                    #                OCVRPInstance.distanceMatrix[cust.id, OCVRPInstance.depot.id]) / (
                    #                           OCVRPInstance.NR_CUST + OCVRPInstance.NR_PUP)/10
                    farness = OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id] / 10
                    #farness = (sum(distancesf) + OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id]) / 180
                    list_farness.append(farness)
                    for distance in distances:
                        if farness < distance:
                            if class_id>0:
                                if row['policy_bab_ID'] & (1 << cust.id - 1):
                                    dict_distances[distance][iter] += 1

                                break
                            else:
                                dict_distances[distance][iter] += 1
                                break
            for distance in distances:
                dict_distances[distance][iter] = dict_distances[distance][iter] / number_instances
        print(sum(list_farness) / len(list_farness))
        print("max farness", max(list_farness),  min(list_farness),  np.percentile(list_farness, 60), np.percentile(list_farness, 90))
        for distance in distances:
            print( distance, dict_distances[distance][0])


        distances.reverse()
        for distance in distances:
            iter = -1
            for iter_temp in class_bins:
                iter += 1
                for dist_temp in distances:
                    if dist_temp < distance:
                        bottom_dict[distance][iter] += dict_distances[dist_temp][iter]
        #lable_dict = {4: '0 - 4: out of 6.6 customers', 8: '4 - 8: out of 6.1 customers',16: '  > 8: out of 5.3 customers'}
        lable_dict = {4: '0 - 4', 8: '4 - 8', 16: '  > 8'}
        #lable_dict = {3: '0 - 3: out of 5.3 customers', 6: '3 - 6: out of 4.5 customers',16: '  > 6: out of 8.2 customers'}

        class_bins_print = [ 0.7, 1.7, 2.7, 3.7, 4.7]
        for index, distance in enumerate(distances):
            print("distance", distance)
            axes.bar(class_bins_print, dict_distances[distance], bottom=bottom_dict[distance], width=0.6, align="edge",
                     color=colors[index], label=lable_dict[distance] ) #, label=lable_dict[distance]
            # str()discount_rate
            # axes.bar(delta_bins, percent, width=1, align="edge",  label=r'$\Delta = $'+str(round(1-p_home,1)), hatch=pattern)  # str()discount_rate

        axes.set(xlabel='Parameter combination')
        axes.set(ylabel='Number of incentives')
        # #plt.gca().set_xticks([round(i+0.5,1) for i in bins_all[:-1]]) r'$(\Delta = 0.6, u=0.06)$
        # plt.gca().set_xticks(class_bins, ['C1\n ' +r'$\Delta = 0.6$'  +'\n'+r'$u=0.06$',
        #                                   'C2\n ' +r'$\Delta = 0.3$'  +'\n'+r'$u=0.06$',
        #                                   'C3\n ' +r'$\Delta = 0.9$'  +'\n'+r'$u=0.06$',
        #                                   'C4\n ' +r'$\Delta = 0.6$'  +'\n'+r'$u=0.03$',
        #                                   'C5\n ' +r'$\Delta = 0.6$'  +'\n'+r'$u=0.12$'])
        plt.gca().set_xticks(class_bins, ['C1','C2',  'C3', 'C4', 'C5'])
        plt.yticks(np.arange(0, 11 + 1, 2.0))
        #axes.set_ylim(0, 23)

        lines, labels = axes.get_legend_handles_labels()
        print(lines, labels)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Distance to \npickup point',handles = reversed(lines), labels=reversed(labels))
        plt.savefig(os.path.join(path_to_images, 'centrality_delta_2.eps'), transparent=False, bbox_inches='tight')
        plt.show()
    if False:
        df = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
        folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30")
        df = df[df.discount_rate == 0.06].copy()
        df = df[df.nrPup == 3].copy()
        df = df[["p_home", 'discount_rate', 'nrPup', 'policy_bab_ID', 'instance']].copy()
        sns.set()
        sns.set(font_scale=1.2)
        sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        fig, axes = plt.subplots(1, 1, sharey=True, sharex=True)
        df['instance_type'] = df['instance'].apply(lambda x: int(str(x).split('_')[10]))
        #df = df[df['instance_type'].isin([0,2,3,4])].copy()
        #distances = [3, 6, 9, 12, 15]
        distances = [4,8,10]
        dict_distances = {}
        delta_bins = [ 0.2, 0.4, 0.6, 0.8, 1]
        bottom_dict = {}
        for distance in distances:
            bottom_dict[distance] = [0] * (len(delta_bins))
        for distance in distances:
            dict_distances[distance] = [0]*(len(delta_bins))

        delta_bins = [ 0.2, 0.4, 0.6, 0.8, 1]
        colors = [(30 / 255, 30 / 255, 50 / 255, 0.8),
                  (190 / 255, 190 / 255, 200 / 255),
                  (90 / 255, 120 / 255, 90 / 255),
                  (230 / 255, 230 / 255, 255 / 255),
                  (140 / 255, 140 / 255, 160 / 255)]
        iter = -1
        list_farness =[]
        for delta in delta_bins:
            iter +=1

            p_home = round(1-delta,1)
            df1 = df[df.p_home == p_home].copy()
            number_instances = 0
            for index, row in df1.iterrows():
                number_instances += 1
                instance = row['instance']
                OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
                for cust in OCVRPInstance.customers:
                    farness =  OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id]/10

                    for distance in distances:
                        if farness < distance and farness >1:
                            if row['policy_bab_ID'] & (1 << cust.id - 1):
                                list_farness.append(farness)
                                dict_distances[distance][iter] += 1
                            break
            print("iter",delta, iter, number_instances)
            for distance in distances:
                dict_distances[distance][iter] = dict_distances[distance][iter]/number_instances
        print("max farness", max(list_farness))
        distances.reverse()
        for distance in distances:
            iter = -1
            for delta in delta_bins:
                iter += 1
                for dist_temp in distances:
                    if dist_temp<distance:
                        bottom_dict[distance][iter] += dict_distances[dist_temp][iter]
        delta_bin_print = [0.15, 0.35, 0.55, 0.75, 0.95]
        for index, distance in enumerate(distances):
            print("distance", distance)
            axes.bar(delta_bin_print, dict_distances[distance],bottom=bottom_dict[distance], width=0.1, align="edge", color=colors[index], label=str(distance-3) + "-" +
                                                                                                                                                 str(distance))
                # str()discount_rate
            #axes.bar(delta_bins, percent, width=1, align="edge",  label=r'$\Delta = $'+str(round(1-p_home,1)), hatch=pattern)  # str()discount_rate

        axes.set(xlabel=r'$\Delta$')
        axes.set(ylabel='Number of offered incentives')
        # #plt.gca().set_xticks([round(i+0.5,1) for i in bins_all[:-1]])
        # plt.gca().set_xticks([2, 5, 8, 11, 14], ['0-3', '3-6', '6-9', '9-12', '12-15'])
        plt.yticks(np.arange(0, 19 + 1, 2.0))
        plt.legend( loc='upper left', title='Distance to pickup point')
        plt.savefig(os.path.join(path_to_images, 'centrality_delta_.eps'), transparent=False, bbox_inches='tight')
        plt.show()
    if False:
        df = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
        folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30")
        df = df[df.discount_rate == 0.06].copy()
        df = df[df.nrPup == 3].copy()
        df = df[["p_home", 'discount_rate', 'nrPup', 'policy_bab_ID', 'instance']].copy()
        sns.set()
        sns.set(font_scale=1.2)
        sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        fig, axes = plt.subplots(1, 1, sharey=True, sharex=True)
        df['instance_type'] = df['instance'].apply(lambda x: int(str(x).split('_')[10]))
        df = df[df['instance_type'].isin([0,2,3,4])].copy()

        for p_home in [0.2,0.8]:
            df1 = df[df.p_home == p_home].copy()
            closeness = []
            all = []
            num_bins = 5
            percent = [0] * num_bins
            percent_all = [0] * num_bins
            number_instances = 0
            for index, row in df1.iterrows():
                number_instances += 1
                instance = row['instance']
                OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
                for cust in OCVRPInstance.customers:
                    farness = OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id]
                    # farness = (sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.customers if
                    #                    j is not cust) + \
                    #                sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.pups) + \
                    #                OCVRPInstance.distanceMatrix[cust.id, OCVRPInstance.depot.id]) / (
                    #                           OCVRPInstance.NR_CUST + OCVRPInstance.NR_PUP)
                    all.append(farness / 10)
                    if row['policy_bab_ID'] & (1 << cust.id - 1):
                        closeness.append(farness/10)

            num_bins = [0,3,6,9,12, 15]
            heights, bins = np.histogram(closeness, bins=num_bins)
            heights_all, bins_all = np.histogram(all, bins=num_bins)

            percent = [i / number_instances for i in heights]
            percent_all = [i / number_instances for i in heights_all]
            axes_bins = [1.5, 4.5, 7.5, 10.5, 13.5]
            if p_home==0.8:
                bins = [i+ 0.5 for i in axes_bins]
                pattern = "/"
                label_all = None
            else:
                label_all = 'all customers'
                #percent[2] -=2
                #percent[3] += 2
                bins = [i - 0.5 for i in axes_bins]
                pattern = 'X'

            axes.bar(bins, percent_all, width=1, align="edge",  label=label_all,color='lightgrey')  # str()discount_rate
            axes.bar(bins, percent, width=1, align="edge",  label=r'$\Delta = $'+str(round(1-p_home,1)), hatch=pattern)  # str()discount_rate

        axes.set(xlabel='Distance to the closest pickup point')
        axes.set(ylabel='Number of offered incentives')
        #plt.gca().set_xticks([round(i+0.5,1) for i in bins_all[:-1]])
        plt.gca().set_xticks([2, 5, 8, 11, 14], ['0-3', '3-6', '6-9', '9-12', '12-15'])

        plt.legend( loc='upper right')
        plt.savefig(os.path.join(path_to_images, 'centrality_delta.eps'), transparent=False, bbox_inches='tight')
        plt.show()
    # effect of centrality and discount on the distribution of  offered incentives
    if False:
        df = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
        folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30")
        df = df[df.p_home == 0.4].copy()
        df = df[df.nrPup == 3].copy()
        df = df[["p_home", 'discount_rate', 'nrPup', 'policy_bab_ID', 'instance']].copy()
        sns.set()
        sns.set(font_scale=1.2)
        sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        fig, axes = plt.subplots(1, 1, sharey=True, sharex=True)
        for discount_rate in [0.03, 0.06]:
            df1 = df[df.discount_rate == discount_rate].copy()
            num_bins = 5
            number_instances = 0
            all = []
            closeness = []
            for index, row in df1.iterrows():

                number_instances += 1
                instance = row['instance']
                OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
                for cust in OCVRPInstance.customers:
                    #if True:
                    distance_to_closest = [OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.customers if j is not cust]
                    farness = OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id]#+ \
                              #OCVRPInstance.distanceMatrix[cust.id, OCVRPInstance.depot.id]) / 2
                    # farness = (sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.customers if
                    #                    j is not cust) + \
                    #                sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.pups) + \
                    #                OCVRPInstance.distanceMatrix[cust.id, OCVRPInstance.depot.id]) / (
                    #                           OCVRPInstance.NR_CUST + OCVRPInstance.NR_PUP)
                    all.append(farness / 10)
                    if row['policy_bab_ID'] & (1 << cust.id - 1):
                        closeness.append(farness/10)
            print(closeness)
            num_bins = [0, 3, 6, 9, 12, 15]
            heights, bins = np.histogram(closeness, bins=num_bins)
            heights_all, bins_all = np.histogram(all, bins=num_bins)
            percent = [ i/ number_instances for i in heights]
            percent_all = [ i/ number_instances for i in heights_all]
            print(bins)
            axes_bins = [1.5, 4.5, 7.5, 10.5, 13.5]
            if discount_rate == 0.06:
                bins = [i + 0.5 for i in axes_bins]
                pattern = "x"
                label_all = None
            else:
                label_all = 'all customers'
                # percent[2] -=2
                # percent[3] += 2
                bins = [i - 0.5 for i in axes_bins]
                pattern = "\\"

            # if discount_rate==0.06:
            #     pattern = "x"
            #     bins_print = [i+1 for i in bins_print]
            #     label_all = None
            # else:
            #     pattern = "\\"
            #     bins_print = [i-0.5 for i in bins]
            #     label_all = 'all customers'
            axes.bar(bins, percent_all, width=1, align="edge", color = 'lightgrey', edgecolor=None, label =label_all)  # str()discount_rate
            axes.bar(bins, percent, width=1, align="edge",  hatch = pattern, label =r'$u=$'+str(discount_rate) ) # str()discount_rate
        #axes.set_xticks([1.7, 4.6, 7.4, 10.3, 13.7])
        plt.gca().set_xticks([2, 5, 8, 11, 14], ['0-3', '3-6', '6-9', '9-12', '12-15'])
        axes.set(xlabel='Distance to the closest pickup point')
        axes.set(ylabel='Number of offered incentives')
        plt.legend( loc='upper right')
        #plt.legend(labels=['all customers',r'$u = 0.03$', r'$u = 0.06$'], loc='upper right')
        plt.savefig(os.path.join(path_to_images, 'centrality_discount_rate.eps'), transparent=False, bbox_inches='tight')
        plt.show()

    # fig, axes = plt.subplots(1, 1, sharey=True, sharex=True)
    #
    # df = pd.read_csv(os.path.join(folder, "02_18_bab_nodisc_rs_30.csv"))
    # folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_30")
    # for p_home in [ 0.2, 0.8]:
    #     df1 = df[df.p_home == p_home].copy()
    #     df1['av_dist_pup_incentivized'] = ''
    #     df1['av_dist_pup'] = ''
    #     dist_pup_incentivized = []
    #     dist_pup = []
    #     closeness = []
    #     num_bins = 4
    #     percent = [0]*num_bins
    #     number_instances = 0
    #     for index, row in df1.iterrows():
    #         number_instances +=1
    #         policy = row['policy_bab_ID']
    #         instance = row['instance']
    #         OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
    #         #dist_pup_incentivized = []
    #         #dist_pup = []
    #         for cust in OCVRPInstance.customers:
    #             if row['policy_bab_ID']& (1 << cust.id-1):
    #                 dist_pup_incentivized.append(OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id])
    #                 farness = (sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.customers if j is not cust)+\
    #                             sum(OCVRPInstance.distanceMatrix[cust.id, j.id] for j in OCVRPInstance.pups)+ \
    #                             OCVRPInstance.distanceMatrix[cust.id, OCVRPInstance.depot.id])/(OCVRPInstance.NR_CUST+OCVRPInstance.NR_PUP)
    #                 closeness.append(1/farness)
    #             dist_pup.append(OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id])
    #         if dist_pup_incentivized:
    #             df1.at[index,'av_dist_pup_incentivized'] =  sum(dist_pup_incentivized)/len(dist_pup_incentivized)
    #         df1.at[index,'av_dist_pup'] =  sum(dist_pup)/len(dist_pup)
    #         heights, bins = np.histogram(closeness, bins=num_bins)
    #         #heights, bins = np.histogram(dist_pup_incentivized, bins=num_bins)
    #         heights_all, bins_all = np.histogram(dist_pup, bins=num_bins)
    #         for i in range(num_bins):
    #            # if heights_all[i]:
    #            #     percent[i] += heights[i]/heights_all[i] *100
    #             #else:
    #             #    percent[i] = 0
    #             percent[i] +=heights[i]/sum(heights) * 100
    #     for i in range(num_bins):
    #         percent[i] =percent[i]/number_instances
    #     axes.bar(bins[:-1], percent, width=0.001, align="edge", alpha = 0.5, label = round(1 -p_home,1))#str()discount_rate
    # axes.set(xlabel='Vertex centrality')
    # axes.set(ylabel='Percentage of offered incentives (%)')
    # plt.legend( labels=[r'$\Delta = 0.8$',r'$\Delta = 0.2$'], loc='upper right')
    # #plt.hist(dist_pup_incentivized, alpha=0.5, density=True, bins=20, label='incentivized')
    # plt.show()
    # plt.savefig(os.path.join(path_to_images, 'centrality_delta.eps'), transparent=False,
    #             bbox_inches='tight')


if __name__ == "__main__":
    folder = os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison")
    folder_data_disc = os.path.join(path_to_data, "data", "i_VRPDO_old")
    folder_data_prob = os.path.join(path_to_data, "data", "i_VRPDO_prob")
    folder_large = os.path.join(path_to_data, "output", "VRPDO_2segm_large")
    #experiment_heuristic_parameters_variation(folder_large)
    #large_exp(folder_large)
    #managerial_effect_delta(folder_large)
    #sensitivity_disc_size_comparison_nodisc(folder, folder_data_disc)
    #sensitivity_comparison_nodisc_rs(os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison"))
    #print_convergence_gap()
    folder_2segm = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm")

    folder_2segm_manyPUP = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm_manyPUP")
    #new remote:
    large_exp(folder_2segm_manyPUP)
    #compare_enumeration_no_Gurobi(folder_2segm_manyPUP)
    #experiment_variation_nrcust_heuristic(folder)
    #experiment_variation_nrcust(folder_2segm_manyPUP)


    #exp_profile()
    #managerial_effect_delta(folder)
    #large_exp(os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison"))
    #managerial_location(os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison"))
    #managerial_effect_delta(os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison"))
    #experiment_bab_solution_time_classes(folder_2segm_manyPUP)
    #experiment_bab_solution_time_classes_pups(folder_2segm_manyPUP)
    #managerial_location(folder_2segm_manyPUP)
    # parseBAB(os.path.join(folder, "bab_7types_nrCust.txt"), folder, "bab_7types_nrCust")

    # experiment_heuristic_general()
    # concorde_gurobi_comparison()
    # experiment_heuristic_time_limit()
    # experiment_Santini_comparison()

    #experiment_varying_discount(folder)
    #experiment_saturation_choice_model()



    # folder_eps = os.path.join(path_to_data, "output", "nr_cust_small", "eps")
    # folder_sample = os.path.join(path_to_data, "output", "nr_cust_small", "sample")
    # experiment_heuristic_table(folder_eps, folder_sample)
    # experiment_heuristic_cliques()
    # experiment_heuristic_rsub()
    # parseBAB_time_limit()
    # experiment_heuristic_sample()

    # folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "i_7_types_nr_cust")
    # folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "nr_cust_small")
    # experiment_nrcust_heuristic_comparison_rs_nodisc(folder)

    # folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "sensitivity_disc_locdep_size")
    # folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "sensitivity_disc_size")
    # folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "artificial_clustered")
    # experiment_sensitivity_disc_size(folder)

    # folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "sensitivity_p_home")
    # experiment_sensitivity_p_home(folder)

    # runtime_comparison(enum_txt_file,bab_txt_file)

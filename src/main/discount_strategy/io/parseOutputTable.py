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
            try:
            #if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])
                    p_home = float(instance.split('_')[4])
                    discount = float((instance.split('_')[8]).split('.txt')[0])
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

                    time_running = float(lines[idx + 7].split(':')[1])
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

                    # data.append([eps, nrCust,p_home, p_pup, discount, time_running, time_first_opt, nodes, num_tsps, optimal,gap, obj_val, sd,
                    #               policy_ID, num_disc, instance])
                    data.append(
                        [nrCust, nrPup, p_home,  discount, time_running, time_first_opt, nodes, num_tsps,
                          obj_val,  policy_ID, num_disc, instance,
                         pruned_by_cliques_nonleaf*100/nodes, pruned_by_cliques_leaf*100/nodes, pruned_by_rs_nonleaf*100/nodes, pruned_by_rs_leaf*100/nodes,
                         pruned_by_insertionCost_nonleaf*100/nodes, pruned_by_insertionCost_leaf*100/nodes, pruned_by_bounds_nonleaf*100/nodes,
                         pruned_by_bounds_leaf*100/nodes, tsp_time*100/time_running, time_exact_bounds*100/time_running, time_lb_addition*100/time_running, time_branch*100/time_running ])
            except:
                #data.append([nrCust, nrPup, p_home,  discount, "", "", "", "", "", "", "",
                #             "", "", "", instance])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
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
                         obj_val,'', policy_ID, num_disc, instance,instance_id, p_pup])
            except:
                data.append([eps, nrCust, nrPup, p_home,  discount, "", "", "", "", "", "", "",
                             "", "", "", instance, p_pup])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['eps', 'nrCust',"nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'optimal', 'gap', 'obj_val_bab', '2sd_bab', 'policy_bab_ID', 'num_disc_bab', 'instance', 'instance_id','p_pup', ]
    writer(os.path.join(folder, output_name + ".csv"), data, rowTitle)


def parseBAB_RS_NODISC(file_path, folder, output_name):
    # print("whatch out, I print the best known LB instead on Obj.Val")
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                # if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    eps = round(float(lines[idx + 2].split(':')[1].split()[0]), 4)
                    p_home = instance.split('_')[4]
                    p_pup = instance.split('_')[6]
                    discount_rate = (instance.split('_')[8]).split('.txt')[0]

                    time_bab = float(lines[idx + 3].split(':')[1])
                    time_tb = float(lines[idx + 9].split(':')[1])
                    nodes = float(lines[idx + 4].split(':')[1])
                    num_tsps = float(lines[idx + 5].split(':')[1])
                    optimal = int(lines[idx + 6].split(':')[1])
                    obj_val_bab = float(lines[idx + 12].split('[')[1].split(',')[0])
                    # 2sd:
                    sd_bab = float(lines[idx + 12].split('[')[1].split(',')[0]) - float(
                        lines[idx + 12].split('[')[1].split(',')[1])
                    policy_ID_bab = int(lines[idx + 10].split(':')[1])
                    try:
                        obj_val_rs = float(lines[idx + 14].split('[')[1].split(',')[0])
                        # 2sd:
                        sd_rs = float(lines[idx + 14].split('[')[1].split(',')[0]) - float(
                            lines[idx + 14].split('[')[1].split(',')[1])
                        policy_ID_rs = int(lines[idx + 13].split(':')[1])
                        obj_val_nodisc = float(lines[idx + 15].split('[')[1].split(',')[0])
                        sd_nodisc = float(lines[idx + 15].split('[')[1].split(',')[0]) - float(
                            lines[idx + 15].split('[')[1].split(',')[1])
                        obj_val_uniform = float(lines[idx + 16].split('[')[1].split(',')[0])
                        sd_uniform = float(lines[idx + 16].split('[')[1].split(',')[0]) - float(
                            lines[idx + 16].split('[')[1].split(',')[1])

                        gap_rs = (obj_val_rs - obj_val_bab) / obj_val_rs * 100
                        gap_nodisc = (obj_val_nodisc - obj_val_bab) / obj_val_nodisc * 100
                        gap_uniform = (obj_val_uniform - obj_val_bab) / obj_val_uniform * 100
                        data.append(
                            [instance, nrCust, p_home, p_pup, discount_rate, policy_ID_bab, obj_val_bab, sd_bab, time_bab,
                             time_tb, nodes,
                             num_tsps, optimal, policy_ID_rs, obj_val_rs, sd_rs, gap_rs, obj_val_nodisc, sd_nodisc,
                             gap_nodisc,
                             obj_val_uniform, sd_uniform, gap_uniform, eps])
                    except:
                        data.append(
                            [instance, nrCust, p_home, p_pup, discount_rate, policy_ID_bab, obj_val_bab, sd_bab,
                             time_bab,
                             time_tb, nodes,
                             num_tsps, optimal, "", "", "", "", "", "",  "", "", "", "", eps])

            except:
                data.append([eps, nrCust, p_home, p_pup, discount_rate, "", "", "", "", "", "", "",
                             "", "", "", instance])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['instance', 'nrCust', 'p_home', 'p_pup', 'discount_rate', 'policy_bab_ID', 'obj_val_bab', '2sd_bab',
                'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'optimal', 'policy_ID_rs', 'obj_val_rs', '2sd_rs', 'gap_rs', 'obj_val_nodisc', '2sd_nodisc',
                'gap_nodisc', 'obj_val_uniform', '2sd_uniform', 'gap_uniform', 'eps']
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
            #try:
            if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    eps = round(float(lines[idx -1].split(':')[1]), 3)
                    p_home = instance.split('_')[4]
                    p_pup = instance.split('_')[6]
                    discount = (instance.split('_')[8]).split('.txt')[0]
                    sample = float(lines[idx -4].split(':')[1])
                    time_limit = float(lines[idx -3].split(':')[1])

                    time_bab = float(lines[idx + 3].split(':')[1])
                    nodes = float(lines[idx + 4].split(':')[1])
                    num_tsps = float(lines[idx + 5].split(':')[1])
                    obj_val_bab = float(lines[idx + 9].split('[')[1].split(',')[1])
                    # 2sd:
                    sd_bab = float(lines[idx + 9].split('[')[1].split(',')[0]) - float(
                        lines[idx + 9].split('[')[1].split(',')[1])
                    policy_ID_bab = int(lines[idx + 7].split(':')[1])
                    try:
                        obj_val_rs = float(lines[idx + 11].split('[')[1].split(',')[2].split(']')[0])
                        # 2sd:
                        sd_rs = float(lines[idx + 11].split('[')[1].split(',')[0]) - float(
                            lines[idx + 11].split('[')[1].split(',')[1])

                        policy_ID_rs = int(lines[idx + 10].split(':')[1])
                        obj_val_nodisc = float(lines[idx + 12].split('[')[1].split(',')[0])
                        sd_nodisc = float(lines[idx + 12].split('[')[1].split(',')[0]) - float(
                            lines[idx + 12].split('[')[1].split(',')[1])
                        obj_val_uniform = float(lines[idx + 13].split('[')[1].split(',')[0])
                        sd_uniform = float(lines[idx + 13].split('[')[1].split(',')[0]) - float(
                            lines[idx + 13].split('[')[1].split(',')[1])

                        gap_rs = (obj_val_rs - obj_val_bab) / obj_val_rs * 100
                        gap_nodisc = (obj_val_nodisc - obj_val_bab) / obj_val_nodisc * 100
                        gap_uniform = (obj_val_uniform - obj_val_bab) / obj_val_uniform * 100
                        data.append(
                            [instance, nrCust, p_home, p_pup, discount, policy_ID_bab, obj_val_bab, sd_bab, time_bab,
                             nodes,
                             num_tsps,
                             policy_ID_rs, obj_val_rs, sd_rs, gap_rs, obj_val_nodisc, sd_nodisc, gap_nodisc,
                             obj_val_uniform, sd_uniform, gap_uniform, eps, sample, time_limit])
                    except:

                        data.append(
                            [instance, nrCust, p_home, p_pup, discount, policy_ID_bab, obj_val_bab, sd_bab, time_bab, nodes,
                             num_tsps,
                             "", "", "", "", "", "", "",
                             "", "", "", eps, sample, time_limit])
            else:
            #except:
                data.append([instance, nrCust, p_home, p_pup, discount])
                print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['instance', 'nrCust', 'p_home', 'p_pup', 'discount_rate', 'policy_bab_ID', 'obj_val_bab', '2sd_bab',
                'time_bab', 'nodes',
                'num_tsps', 'policy_ID_rs', 'obj_val_rs', '2sd_rs', 'gap_rs', 'obj_val_nodisc', '2sd_nodisc',
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


def sensitivity_prob_comparison_nodisc(folder, folder_data):
    # parseBAB_RS_NODISC(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_prob.txt"), folder, "bab_rs_nodisc_i_VRPDO_prob")
    df = pd.read_csv(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_prob.csv"))
    df['instance_type'] = df['instance'].apply(lambda x: str(x).split('_')[0])
    df['in_pup_bab'] = df.apply(lambda x: calculate_exp_pup_utilization(x['policy_bab_ID'], x['instance'], folder_data),
                                axis=1)

    df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID']) ), axis=1)
    df['num_offered_disc_rs'] = df.apply(lambda x: round(bitCount(x['policy_ID_rs']) / 15 * 100), axis=1)
    df['num_offered_disc_uniform'] = 100

    df['in_pup_rs'] = df.apply(lambda x: calculate_exp_pup_utilization(x['policy_ID_rs'], x['instance'], folder_data),
                               axis=1)
    df['in_pup_uniform'] = df.apply(lambda x: calculate_exp_pup_utilization(32767, x['instance'], folder_data), axis=1)
    df['p_accept'] = round(1 - df['p_pup'] - df['p_home'], 2)

    df['exp_discount_cost_bab'] = df.apply(
        lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data), axis=1)
    df['upper_bound'] = df.apply(lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data),
                                 axis=1)
    df['exp_discount_cost_bab_percent'] = (df['exp_discount_cost_bab'] / df['obj_val_bab']) * 100

    df['exp_discount_cost_rs'] = df.apply(
        lambda x: calculate_exp_disc_cost(x['policy_ID_rs'], x['instance'], folder_data), axis=1)
    df['exp_discount_cost_rs_percent'] = (df['exp_discount_cost_rs'] / df['obj_val_rs']) * 100

    df['exp_discount_cost_uniform'] = df.apply(lambda x: calculate_exp_disc_cost(32767, x['instance'], folder_data),
                                               axis=1)
    df['exp_discount_cost_uniform_percent'] = (df['exp_discount_cost_uniform'] / df['obj_val_uniform']) * 100

    df['nodisc_bab, %'] = round((df['obj_val_nodisc'] - df['obj_val_bab']) / df['obj_val_bab'] * 100, 1)
    df['nodisc_rs, %'] = round((df['obj_val_nodisc'] - df['obj_val_rs']) / df['obj_val_nodisc'] * 100, 1)
    df['nodisc_uniform, %'] = round((df['obj_val_nodisc'] - df['obj_val_uniform']) / df['obj_val_nodisc'] * 100, 1)
    df['rs_bab_comp,%'] = df.apply(lambda x: (x['obj_val_rs'] - x['obj_val_bab']) / x['obj_val_bab'] * 100, axis=1)
    df['all_bab_comp,%'] = df.apply(lambda x: (x['obj_val_uniform'] - x['obj_val_bab']) / x['obj_val_bab'] * 100,
                                    axis=1)

    # loss of using the best policy among d-ds, all, BM
    df['loss_best_comp,%'] = df.apply(
        lambda x: min(x['nodisc_bab, %'], (x['obj_val_rs'] - x['obj_val_bab']) / x['obj_val_rs'] * 100,
                      (x['obj_val_nodisc'] - x['obj_val_bab']) / x['obj_val_nodisc'] * 100), axis=1)
    for instance_type in df.instance_type.unique():
        print("instance_type", instance_type)
        df1 = df[df.instance_type == instance_type].copy()
        discount_rates = []

        rs_saving = []
        uniform_savings = []
        # key(p_pup):p_delta
        dict_probabilities = {0.0: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                              0.2: [0.0, 0.2, 0.4, 0.6, 0.8],
                              0.4: [0.0, 0.2, 0.4, 0.6],
                              0.6: [0.0, 0.2, 0.4],
                              0.8: [0.0, 0.2],
                              1.0: [0.0]}

        # upper and lower bounds:
        df['lower_bound'] = df.apply(lambda x: calculate_lower_bounds(x['instance'], folder_data), axis=1)
        df['upper_bound'] = df.apply(lambda x: calculate_upper_bounds(x['instance'], folder_data), axis=1)
        df['improvement'] = df.apply(lambda x: 100*(x['upper_bound']-x['obj_val_bab'])/x['upper_bound'], axis=1)

        # df_temp = df[
        #     (df.p_pup == 0.0) | (df.p_pup == 0.2) | (df.p_pup == 0.4) | (df.p_pup == 0.6) | (df.p_pup == 0.8)   ].copy()
        # sns.set()
        # sns.set(font_scale=1.2)
        # #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        #
        # x = [ 0.2, 0.4, 0.6, 0.8, 1]
        # cmap = sns.color_palette("deep")
        # m = np.array(['o', 'P', 's', '^', 'D',"+"])
        # lns = np.array(['-', '--', '-.', ':', 'dashdot'])
        # change_x = [-0.04, -0.02, 0, 0.02,0.04]
        # labels = [r'$p = 0.8$', r'$p = 0.6$', r'$p = 0.4$',  r'$p = 0.2$',  r'$p = 0.0$',]
        # p_pups =  [0.8, 0.6, 0.4, 0.2, 0.0]
        # for iter in [0,1,2,3,4]:
        #     df_temp2 = df_temp[df_temp['p_pup'] == p_pups[iter]].copy()
        #     y = []
        #     yerr = []
        #     ymin = []
        #     ymax = []
        #     x_print = []
        #     for delta in x:
        #         df_temp3 = df_temp2[df_temp2.p_accept == delta].copy()
        #         x_print.append(delta + change_x[iter])
        #         y.append(df_temp3['obj_val_bab'].mean())
        #         ymin.append(df_temp3['obj_val_bab'].min())
        #         ymax.append(df_temp3['obj_val_bab'].max())
        #
        #         yerr.append([df_temp3['obj_val_bab'].mean() - df_temp3['obj_val_bab'].min(),
        #                      df_temp3['obj_val_bab'].max() - df_temp3['obj_val_bab'].mean()])
        #
        #     yerr = np.transpose(yerr)
        #     # plt.scatter(x_print, y_min,  marker = '-', s= 16,  label = class_name[class_id-1])
        #     # plt.errorbar(x_print, y_min, yerr=None, marker = m[class_id-1],  ms=3, linestyle = '', mec = cmap[class_id-1],mfc= cmap[class_id-1], c = cmap[class_id-1],)
        #
        #     plt.errorbar(x_print, y, yerr=yerr, marker=m[iter], ms=7, linewidth=0.5,
        #                  label=labels[iter],
        #                  mec=cmap[iter], mfc=cmap[iter], c=cmap[iter], elinewidth=1, capsize=3)


        #sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='obj_val_bab', hue='p_pup', style="p_pup", markers=True,
        #             markersize=9, linewidth=1,
        #             palette="deep", err_style="bars", ci=100, err_kws={'capsize': 3}, dashes=False)



        # file_path = os.path.join(folder, "bab_rs_nodisc_i_VRPDO_prob_bounds.xlsx")
        # writer = pd.ExcelWriter(file_path, engine='openpyxl')
        # df.to_excel(writer, index=False)
        # writer.save()
        #axes.fill_between(x, df['upper_bound'].min(), df['upper_bound'].max(), alpha=0.2)
        #axes.axhline(df['upper_bound'].mean(), ls='--')
        # lower bound
        #axes.axhline(df['lower_bound'].mean(), ls='--')

        # axes.set(xlabel='' + r'$\Delta$')
        # axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        # axes.set(ylabel='Expected fulfillment cost')
        # plt.legend(title=False, loc='lower right',ncol=2, bbox_to_anchor=(0.99, 0.01))
        # #plt.legend(title=False, loc='lower right', bbox_to_anchor=(0.99, 0.1),
        # #           labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$', r'$p = 0.8$', r'$p = 1.0$'])
        # plt.savefig(os.path.join(path_to_images, 'Total_cost_hue_ppup_bounds.eps'), transparent=False,
        #             bbox_inches='tight')
        # plt.show()


        # sns.set()
        # sns.set(font_scale=1.1)
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='exp_discount_cost_bab', hue='p_accept', style="p_accept", linewidth=1,markers=True, markersize=7,
        #             palette="deep",  err_style="bars", ci=68, err_kws={'capsize': 3})
        # axes.set(xlabel='' + r'$p$')
        # axes.set(ylabel='Expected incentive cost')
        # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$'])
        # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$',
        #                                 r'$\Delta = 1.0$'])
        # plt.savefig(os.path.join(path_to_images, 'Incentive_cost_hue_ppup.eps'), transparent=False, bbox_inches='tight')
        # plt.show()
        #

        # sns.set()
        # df_temp = df[(df.p_pup == 0.0)].copy()
        # sns.set(font_scale=1.3)
        # #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        #
        # cmap = sns.color_palette("deep")
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # x = [ 0.2, 0.4, 0.6, 0.8, 1]
        # y=[]
        # yerr =[]
        # for p_accept in x:
        #     df_temp2 = df_temp[df_temp.p_accept == p_accept].copy()
        #     y.append(df_temp2['num_offered_disc_bab'].mean())
        #     yerr.append([df_temp2['num_offered_disc_bab'].mean() - df_temp2['num_offered_disc_bab'].min(),
        #                                        df_temp2['num_offered_disc_bab'].max() - df_temp2['num_offered_disc_bab'].mean()])
        # yerr = np.transpose(yerr)
        # plt.errorbar(x, y, yerr=yerr, marker='D', ms=7, capsize=3,mec=cmap[4], mfc=cmap[4], c=cmap[4] )
        #
        # # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='num_offered_disc_bab',
        # #              linewidth=1, markersize=7, markers=True, marker='o',
        # #              palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # axes.set(xlabel='' + r'$\Delta$')
        # axes.set(ylabel='Number of offered incentives')
        # # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$', r'$\Delta = 1.0$'])
        # plt.savefig(os.path.join(path_to_images, 'Number_incentives_delta.eps'), transparent=False, bbox_inches='tight')
        # plt.show()




        # fontsize = 15
        # sns.set(font_scale=1.4)
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # sns.color_palette("Spectral", as_cmap=True)
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='rs_bab_comp,%',markers=True,
        #              markersize=8,linewidth=1,palette="deep", err_style="bars", ci=96, err_kws={'capsize': 3})
        # #sns.relplot(data=df_temp, x='p_pup', y='p_accept',hue='gap_nodisc_mean', s=100, palette="Spectral")
        # plt.show()

        # fontsize = 15
        # sns.set(font_scale=1.4)
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='nodisc_bab, %', hue='p_home',  style="p_home",markers=True, markersize=8,
        #              linewidth=1,palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # plt.show()
        # sns.set(font_scale=1.4)
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.directionp_accep": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='rs_bab_comp,%', hue='p_home', style="p_home", markers=True,
        #              markersize=8,
        #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # plt.show()
        # sns.set(font_scale=1.4)
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='all_bab_comp,%', hue='p_home', style="p_home", markers=True,
        #              markersize=8,
        #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # plt.show()
        #

        # fontsize = 15
        # sns.set(font_scale=1.4)
        # df_temp = df[(df.p_pup == 0.0)| (df.p_pup==0.4) | (df.p_pup ==0.8)].copy()
        # df_temp['1p_accept'] = 1-df_temp['p_accept']
        #
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='nodisc_bab, %', hue='p_home', style='p_home',
        #              markers=True, markersize=8, linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # axes.set(xlabel=r'$\Delta$')
        # axes.set(ylabel='Improvement over NOI (%)')
        # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.4$', r'$p = 0.8$'])
        # plt.savefig(os.path.join(path_to_images, 'Bab_nodisc_comparison.eps'), transparent=False, bbox_inches='tight')
        # plt.show()
        #
        # sns.set(font_scale=1.4)
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='rs_bab_comp,%',hue='p_home', style='p_home',
        #              markers=True, markersize=8,
        #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.4$', r'$p = 0.8$'])
        # axes.set(xlabel=r'$\Delta$')
        # axes.set(ylabel='Improvement over D-DS (%)')
        # plt.savefig(os.path.join(path_to_images, 'Bab_rs_comparison.eps'), transparent=False, bbox_inches='tight')
        # plt.show()
        #
        # sns.set(font_scale=1.4)
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_accept', y='all_bab_comp,%',  markers=True, markersize=8,hue='p_home', style='p_home',
        #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 4})
        # axes.set(xlabel=r'$\Delta$')
        # axes.set(ylabel='Improvement over ALL (%)')
        # #plt.legend(title=False, labels=[r'$p = 0.0$', r'$p = 0.4$', r'$p = 0.8$'])
        # plt.savefig(os.path.join(path_to_images, 'Bab_all_comparison.eps'), transparent=False, bbox_inches='tight')
        # plt.show()


        fig, axes = plt.subplots(1, 4, sharey=True, figsize=(19, 5), gridspec_kw={'width_ratios': [10, 8, 6, 4]})
        sns.set(font_scale=1.4)

        rc = {'font.sans-serif': 'Computer Modern Sans Serif'}
        sns.set_context(rc=rc)
        sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        plt.rcParams.update(**rc)
        cmap = sns.color_palette("deep")
        m = np.array(['o', '^', 'D', '^', 'D', "+"])

        iter = -1
        for p_pup_param in [0.0, 0.2, 0.4, 0.6]:
            iter += 1
            df_temp = df[df.p_pup == p_pup_param].copy()

            x = [0.2, 0.4, 0.6, 0.8, 1.0]
            if iter == 1:
                x = [0.2, 0.4, 0.6, 0.8]
                #axes[iter].set_xlim(None, 0.82)
            if iter == 2:
                x = [0.2, 0.4, 0.6]
                #axes[iter].set_xlim(None, 0.62)
            if iter == 3:
                x = [ 0.2, 0.4]
                #axes[iter].set_xlim(0.18, 0.45)
                #axes[iter].set_xticks([0.0, 0.2, 0.4])
            y = []
            yerr = []

            y_d = []
            yerr_d = []

            y_noi = []
            yerr_noi = []

            x_all = [temp-0.02 for temp in x]
            x_noi = [temp+0.02 for temp in x]

            for p_accept in x:
                df_temp2 = df_temp[df_temp.p_accept == p_accept].copy()
                y.append(df_temp2['all_bab_comp,%'].mean())
                yerr.append([df_temp2['all_bab_comp,%'].mean() - df_temp2['all_bab_comp,%'].min(),
                                                   df_temp2['all_bab_comp,%'].max() - df_temp2['all_bab_comp,%'].mean()])

                y_d.append(df_temp2['rs_bab_comp,%'].mean())
                yerr_d.append([df_temp2['rs_bab_comp,%'].mean() - df_temp2['rs_bab_comp,%'].min(),
                             df_temp2['rs_bab_comp,%'].max() - df_temp2['rs_bab_comp,%'].mean()])

                y_noi.append(df_temp2['nodisc_bab, %'].mean())
                yerr_noi.append([df_temp2['nodisc_bab, %'].mean() - df_temp2['nodisc_bab, %'].min(),
                               df_temp2['nodisc_bab, %'].max() - df_temp2['nodisc_bab, %'].mean()])

            yerr = np.transpose(yerr)
            yerr_d = np.transpose(yerr_d)
            yerr_noi = np.transpose(yerr_noi)

            axes[iter].errorbar(x_all, y, yerr=yerr, marker=m[2], ms=7, linewidth=0.3,
                         label="ALL", mec=cmap[2], mfc=cmap[2], c=cmap[2], elinewidth=1.5, capsize=4)

            axes[iter].errorbar(x_noi, y_noi, yerr=yerr_noi, marker=m[0], ms=7, linewidth=0.3,
                         label="NOI", mec=cmap[0], mfc=cmap[0], c=cmap[0], elinewidth=1.5, capsize=4)

            axes[iter].errorbar(x, y_d, yerr=yerr_d, marker=m[1], ms=7, linewidth=0.3,
                         label="DI", mec=cmap[1], mfc=cmap[1], c=cmap[1], elinewidth=1.5, capsize=4)

            axes[iter].set(xlabel='Problem size, n')

            # sns.lineplot(ax=axes[iter], data=df_temp, x='p_accept', y='all_bab_comp,%', marker="s", markersize=7,
            #              linewidth=1,
            #              palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3}, label='ALL', legend=0)
            # sns.lineplot(ax=axes[iter], data=df_temp, x='p_accept', y='rs_bab_comp,%', marker="o", markersize=8,
            #              linewidth=1,
            #              palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3}, label='DI', legend=0)
            # sns.lineplot(ax=axes[iter], data=df_temp, x='p_accept', y='nodisc_bab, %',  marker="^", markersize=9, linewidth=1,
            #             palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3}, label = 'NOI', legend=0)
            # plt.legend(title=False,loc = 'upper right', bbox_to_anchor=(0.995, 0.995))

            axes[iter].set_title(r'$p =$' + str(p_pup_param))
            # if iter == 1:
            #     axes[iter].set_xlim(None, 0.82)
            # if iter == 2:
            #     axes[iter].set_xlim(None, 0.62)
            # if iter == 3:
            #     axes[iter].set_xlim(None, 0.45)
            axes[iter].set_xlabel(r'$\Delta$', fontsize=18)

        handles, labels = axes[iter].get_legend_handles_labels()

        axes[0].set_ylabel('Savings (%)', fontsize = 19)
        axes[0].legend(handles, labels,loc = 'upper left', bbox_to_anchor=(0.01, 0.99),  title=False)

        plt.savefig(os.path.join(path_to_images, 'Improvement_threparcee_plots.eps'), transparent=False,
                    bbox_inches='tight')
        plt.show()
        #
        # sns.set(font_scale=1.4)
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='rs_bab_comp,%',
        #              markers=True, markersize=8,
        #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 3})
        # #axes.set(xlabel=r'$p')
        # axes.set(ylabel='Improvement over D-DS (%)')
        # plt.savefig(os.path.join(path_to_images, 'Bab_rs_comparison.eps'), transparent=False, bbox_inches='tight')
        # plt.show()
        #
        # sns.set(font_scale=1.4)
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
        # sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
        # sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
        # fig, axes = plt.subplots(1, 1, sharex=True)
        # sns.lineplot(ax=axes, data=df_temp, x='p_pup', y='all_bab_comp,%', markers=True, markersize=8,
        #              linewidth=1, palette="deep", err_style="bars", ci=68, err_kws={'capsize': 4})
        # #axes.set(xlabel=r'$p$')
        # axes.set(ylabel='Improvement over ALL (%)')
        # plt.savefig(os.path.join(path_to_images, 'Bab_all_comparison.eps'), transparent=False, bbox_inches='tight')
        # plt.show()

        # print("bab")
        # for p_pup in dict_probability_print:
        #     bab_savings = []
        #     for p_delta in dict_probability_print[p_pup]:
        #         #bab_savings.append( str(int(round(df1.loc[(df1['p_pup'] == p_pup) & (df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_bab'].mean(),0))))
        #         bab_savings.append(str(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                     df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'nodisc_bab, %'].mean(), 1)))
        #         #in_pup_bab
        #         bab_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                         df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'num_offered_disc_bab'].mean(),0)))
        #         bab_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'exp_discount_cost_bab_percent'].mean(), 0)))
        #         bab_savings.append("ydali")
        #         #append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_bab'].mean(),0)))
        #     print("&$p=",p_pup,"$&", ' & '.join(map(str, bab_savings)), "\\")
        # print("rs")
        # for p_pup in dict_probability_print:
        #     rs_savings = []
        #     for p_delta in dict_probability_print[p_pup]:
        #         # rs_savings.append(str(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #         #         df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_rs'].mean(),
        #         #                                 0))))
        #         rs_savings.append(str(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'nodisc_rs, %'].mean(), 1)))
        #
        #         rs_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'num_offered_disc_rs'].mean())))
        #         rs_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'exp_discount_cost_rs_percent'].mean(), 0)))
        #         rs_savings.append("ydali")
        #         #rs_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #         #        df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_rs'].mean())))
        #
        #     print("&$p=",p_pup,"$&",' & '.join(map(str, rs_savings)), "\\")
        # print("uniform")
        # for p_pup in dict_probability_print:
        #     uniform_savings = []
        #     for p_delta in dict_probability_print[p_pup]:
        #         #uniform_savings.append(str(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #         #        df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_uniform'].mean(),
        #         #                                     0))))
        #         uniform_savings.append(str(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                     df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'nodisc_uniform, %'].mean(), 1)))
        #
        #         uniform_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                     df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'num_offered_disc_uniform'].mean())))
        #         uniform_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #                 df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'exp_discount_cost_uniform_percent'].mean(), 0)))
        #         uniform_savings.append("ydali")
        #         #uniform_savings.append(int(round(df1.loc[(df1['p_pup'] == p_pup) & (
        #         #        df1['p_home'] == round((1 - p_delta - p_pup), 1)), 'obj_val_uniform'].mean())))
        #
        #     print("&$p=",p_pup,"$&", ' & '.join(map(str, uniform_savings)), "\\")


def print_convergence_gap():
    mainDirStorage = os.path.join(path_to_data, "output")
    convergence = os.path.join(mainDirStorage, 'convergence.txt')
    # matplotlib.rcParams['text.usetex'] = True
    # rc('text', usetex=True)
    # plt.rcParams['font.size'] = '16'
    with open(convergence, 'rb') as file:
        time = pickle.load(file)
        lbPrint = pickle.load(file)
        ubPrint = pickle.load(file)
    fig = plt.figure()
    sns.set()
    sns.set(font_scale=1.2)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})

    gap = []
    # for i, lb in enumerate(lbPrint):
    #    gap.append((ubPrint[i] - lbPrint[i]) / ubPrint[i])
    # plt.plot(time, gap, '-', label='Gap')
    bab_obj = 65.1999
    plt.axhline(y=bab_obj, color='r', linestyle='--', label='optimal objective value')

    plt.plot(time, ubPrint, '-', label='best known upper bound ')
    plt.plot(time, lbPrint, '-', label='best known lower bound ')

    # plt.plot(time, ubPrint, '-',  label='best upper bound ' + r'$(\overline{z}^{*})$')
    # plt.plot(time, lbPrint, '-', label='best lower bound '+r'$(\underline{z}^{*})$')

    plt.xlabel("Running time (sec)")
    plt.ylabel("Expected fulfillment cost")
    plt.legend()
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
    # plt.legend(title=False, loc='lower right', bbox_to_anchor=(0.99, 0.12),
    #           labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$', r'$p = 0.8$', r'$p = 1.0$'])
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
    #parseBAB(os.path.join(folder, "02_08_bab_exact.txt"), folder, "02_08_bab_exact")
    #parseBAB(os.path.join(folder, "02_06_bab_exact_dominance_check.txt"), folder, "02_06_bab_exact_dominance_check")
    #parseEnumeration(os.path.join(folder, "02_13_enumeration.txt"), folder, "02_13_enumeration")
    df_enum = pd.read_csv(os.path.join(folder, "02_13_enumeration.csv"))
    df_bab = pd.read_csv(os.path.join(folder, "02_08_bab_exact.csv"))
    df_bab_dominance_check = pd.read_csv(os.path.join(folder, "02_06_bab_exact_dominance_check.csv"))
    df_bab_dominance_check = df_bab_dominance_check[['instance','nodes', 'time_bab']].copy()
    df_bab_dominance_check.rename(columns={'nodes': 'nodes_domcheck', 'time_bab': 'time_bab_domcheck'}, inplace=True)
    df_enum.drop(['nrCust_enum'], axis=1, inplace=True)
    df = df_bab.merge(df_enum, how='left', on='instance')
    df = df.merge(df_bab_dominance_check, how='left', on='instance') #how=left
    nr_cust_variation(df)

    #parseBAB(os.path.join(folder, "bab_VRPDO_disc_proportional_small_02_02_006.txt"), folder, "bab_VRPDO_disc_proportional_small_02_02_006")
    #df_bab = pd.read_csv(os.path.join(folder, "i_VRPDO_time.csv"))
    #df_bab_tl = pd.read_csv(os.path.join(folder, "bab_3600_VRPDO_disc_proportional_02_02_006.csv"))
    #df_bab_tl = pd.read_csv(os.path.join(folder, "bab_VRPDO_discount_proportional_3600.csv"))
    # df_bab_tl.drop(['p_home', 'p_pup', 'discount_rate', 'time_bab', 'time_tb', 'nodes', 'num_tsps', 'policy_ID_rs',
    #                 'obj_val_rs', '2sd_rs', 'gap_rs', 'obj_val_nodisc', '2sd_nodisc', 'gap_nodisc', 'obj_val_uniform',
    #                 '2sd_uniform', 'gap_uniform', 'eps'], axis=1, inplace=True)

    #df_bab_tl.drop(['p_home', 'p_pup', 'discount_rate', 'time_bab',  'nodes', 'num_tsps', 'eps', 'nrCust','time_tb'], axis=1, inplace=True)

    #df_bab_tl.rename(columns={'policy_bab_ID': 'policy_h_ID', 'obj_val_bab': 'obj_val_h', '2sd_bab': '2sd_h',
    #                          'optimal': 'optimal_h'},
    #                 inplace=True)

    #df = df.merge(df_bab_tl, how='left', on='instance')
    # df['opt_gap_h'] = df.apply(
    #     lambda x: 0 if (x['optimal_h'] == x['policy_bab_ID'] or x['time_bab']<3600) else (x['obj_val_h'] - x['obj_val_bab']) / x[
    #         'obj_val_h'] * 100, axis=1)
    #df['solved_h'] = df.apply(
    #    lambda x: 0 if x['time_bab'] >=3600 else 1, axis=1)
    # file_path = os.path.join( folder,  "Comparison_full_table.xlsx")
    # writer = pd.ExcelWriter(file_path, engine='openpyxl')
    # df.to_excel(writer, index=False)
    # writer.save()
    #df = pd.read_excel(os.path.join(folder, "Comparison_full_table.xlsx"))
    #nr_cust_variation(df)


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
    #parseBAB(os.path.join(folder, "h_rs_VRPDO_discount_proportional.txt"), folder, "h_rs_VRPDO_disc_proportional")
    df_rs = pd.read_csv(os.path.join(folder, "h_rs_VRPDO_disc_proportional.csv"))

    #parseBAB(os.path.join(folder, "bab_VRPDO_discount_proportional_3600.txt"), folder, "bab_VRPDO_discount_proportional_3600")
    df_bab = pd.read_csv(os.path.join(folder, "bab_3600_VRPDO_disc_proportional_02_02_006.csv"))

    #parseBABHeuristic(os.path.join(folder, "h_VRPDO_disc_proportional_02_02.txt"), folder, "h_VRPDO_disc_proportional_02_02")
    df_h = pd.read_csv(os.path.join(folder, "h_VRPDO_disc_proportional_02_02.csv"))

    nr_cust = [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 35, 40, 45, 50]
    #nr_cust = [10, 15, 18, 20, 25, 30, 35, 40, 45,50]
    df_results = pd.DataFrame(index=nr_cust,
                        columns=['exactt','exactn',"s1",'s500t','s500node', 's500gap','s2', 's200t',  's200node', 's200gap',
                        's3', 's100t', 's100node', 's100gap', 's4','s50t', 's50node','s50gap', 's5', 's20t', 's20node',
                        's20gap','s6','rst','rsn','rsg'])

    df_bab = df_bab[(df_bab['discount_rate'] == 0.06) ].copy()
    df_h = df_h[(df_h['discount_rate'] == 0.06)].copy()
    df_bab = df_bab[['instance', 'policy_bab_ID', 'obj_val_bab', '2sd_bab', 'time_bab','nodes', 'nrCust']]
    df_rs = df_rs[['instance', 'policy_bab_ID', 'obj_val_bab', 'time_bab','nodes','nrCust']]
    df_rs['time_bab'] = df_rs.apply(lambda x: 3600 if x['time_bab'] >3600 else x['time_bab'], axis=1)

    df_bab_small = pd.read_csv(os.path.join(folder, "bab_VRPDO_disc_proportional_small_02_02_006.csv"))
    df_bab_small = df_bab_small[['instance', 'time_bab']].copy()
    df_bab_small.rename(columns={'time_bab': 'time_bab_small'}, inplace=True)
    df_bab = df_bab.merge(df_bab_small, how='outer', on='instance')
    df_bab['time_bab'] = df_bab.apply(lambda x: x['time_bab_small'] if x['nrCust'] < 19 else x['time_bab'], axis=1)
    df_bab['time_bab'] = df_bab.apply(lambda x: 3600 if x['time_bab'] >3600 else x['time_bab'], axis=1)

    df_bab = df_bab.dropna(subset=['time_bab'])

    df_bab.rename(
        columns={'policy_bab_ID': 'policy_bab_ID_exact', 'obj_val_bab': 'obj_val_bab_exact', '2sd_bab': '2sd_bab_exact',
                 'time_bab': 'time_bab_exact', 'nrCust': 'nrCust_exact', 'nodes':'nodes_exact'}, inplace=True)

    df_h = df_h[['instance', 'nrCust', 'time_bab', 'nodes', 'obj_val_bab', '2sd_bab', 'policy_bab_ID','eps', 'sample','time_limit']]
    df_h['time_bab'] = df_h.apply(lambda x: 3600 if x['time_bab']>3600 else x['time_bab'], axis=1)
    df_h = df_h.merge(df_bab, on='instance')
    df_rs = df_rs.merge(df_bab, on='instance')

    for df in [df_h,df_rs]:
        # df['gap_ub'] = (df['obj_val_bab'] + df['2sd_bab'] - (
        #         df_h['obj_val_bab_exact'] - df['2sd_bab_exact'])) / df_h[
        #                      'obj_val_bab'] * 100
        df['gap_av'] = (df['obj_val_bab'] - df['obj_val_bab_exact']) / df['obj_val_bab'] * 100
        df.loc[df['policy_bab_ID'] == df['policy_bab_ID_exact'], 'gap_av'] = 0
        #df.loc[df['policy_bab_ID'] == df['policy_bab_ID_exact'], 'gap_ub'] = 0
        df.drop(['obj_val_bab_exact', 'policy_bab_ID_exact', '2sd_bab_exact'], axis=1, inplace=True)

    for n in nr_cust:
        df_bab_temp = df_bab[df_bab["nrCust_exact"] == n].copy()
        df_temp_sample = df_h[df_h["nrCust"] == n].copy()
        df_temp_rs = df_rs[df_rs["nrCust"] == n].copy()
        df_results.at[n, 'exactt'] = round(df_bab_temp['time_bab_exact'].mean(), 1)
        df_results.at[n, 'exactn'] = round(df_bab_temp['nodes_exact'].mean())

        df_temp_s500 = df_temp_sample[(df_temp_sample['sample'] == 500) & (df_temp_sample['eps'] == 0)].copy()
        df_temp_s200 = df_temp_sample[(df_temp_sample['sample'] == 200) & (df_temp_sample['eps'] == 0)].copy()
        df_temp_s100 = df_temp_sample[(df_temp_sample['sample'] == 100) & (df_temp_sample['eps'] == 0)].copy()
        df_temp_s50 = df_temp_sample[(df_temp_sample['sample'] == 50) & (df_temp_sample['eps'] == 0)].copy()
        df_temp_s20 = df_temp_sample[(df_temp_sample['sample'] == 20) & (df_temp_sample['eps'] == 0)].copy()

        df_results.at[n, 'rst'] = round(df_temp_rs['time_bab'].mean(), 1)
        df_results.at[n, 'rsn'] = round(df_temp_rs['nodes'].mean())
        df_results.at[n, 'rsg'] = round(df_temp_rs['gap_av'].mean(), 2)

        df_results.at[n, 's500t'] = round(df_temp_s500['time_bab'].mean(), 1)
        df_results.at[n, 's500node'] = round(df_temp_s500['nodes'].mean())
        df_results.at[n, 's500gap'] = round(df_temp_s500['gap_av'].mean(), 2)

        df_results.at[n, 's200t'] = round(df_temp_s200['time_bab'].mean(), 1)
        df_results.at[n, 's200node'] = round(df_temp_s200['nodes'].mean())
        df_results.at[n, 's200gap'] = round(df_temp_s200['gap_av'].mean(), 2)

        df_results.at[n, 's100t'] = round(df_temp_s100['time_bab'].mean(), 1)
        df_results.at[n, 's100node'] = round(df_temp_s100['nodes'].mean())
        df_results.at[n, 's100gap'] = round(df_temp_s100['gap_av'].mean(), 2)

        df_results.at[n, 's50t'] = round(df_temp_s50['time_bab'].mean(), 1)
        df_results.at[n, 's50node'] = round(df_temp_s50['nodes'].mean())
        df_results.at[n, 's50gap'] = round(df_temp_s50['gap_av'].mean(), 2)

        df_results.at[n, 's20t'] = round(df_temp_s20['time_bab'].mean(), 1)
        df_results.at[n, 's20node'] = round(df_temp_s20['nodes'].mean())
        df_results.at[n, 's20gap'] = round(df_temp_s20['gap_av'].mean(), 2)


    #print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))
    df_results = df_results[['exactt',  'exactn', "s1",  's200t', 's200node','s200gap', 's3',
                              's100t', 's100node','s100gap','s4',  's20t', 's20node','s20gap','s2', 'rst', 'rsn', 'rsg']].copy()
    print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep=''))


def experiment_bab_solution_time_classes(folder):
    #parseBAB(os.path.join(folder, "bab_VRPDO_disc_proportional_small_not_finished.txt"), folder, "bab_VRPDO_discount_proportional_small")
    df_bab = pd.read_csv(os.path.join(folder, "01_30_bab_exact_small.csv"))
    #df_bab = df_bab[df_bab['p_pup']!=0.35].copy()
    nr_cust = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for n in nr_cust:
        df_bab_temp = df_bab[(df_bab["nrCust"] == n)&(df_bab['discount_rate']==0.06)].copy()
        print(n, df_bab_temp['time_bab'].mean())

    df_bab['class'] = df_bab.apply( lambda x: 'low_determinism' if x['p_home']==0.9 else (
        'high_determinism' if x['p_home']==0.2 else (
            'high_disc' if x['discount_rate']==0.12 else (
                'low_disc' if x['discount_rate']==0.03 else 'normal'))), axis=1)

    df_bab['class_id'] = df_bab.apply(lambda x: 2 if x['p_home'] == 0.9 else (
        3 if x['p_home'] == 0.2 else (
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
    x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

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
    plt.savefig(os.path.join(path_to_images, 'Solution_time_classes_2segm.eps'), transparent=False,
            bbox_inches='tight')
    plt.show()

def average_discount(instance_name):
    file_instance = os.path.join(path_to_data, "data", "i_VRPDO_saturation",
                                 instance_name+".txt")
    instance = OCVRPParser.parse(file_instance)

    discount_value_av = []
    for cust in instance.customers:
        discount_value_av.append(instance.shipping_fee[cust.id])
    print(discount_value_av, instance_name, sum(discount_value_av)/len(discount_value_av))
    return sum(discount_value_av)/len(discount_value_av)

def experiment_saturation_choice_model():
    folder = os.path.join(path_to_data, "output", "VRPDO_saturation")
    #parseBAB(os.path.join(folder, "bab_VRPDO_saturation2.txt"), folder, "bab_VRPDO_saturation2")
    df_bab = pd.read_csv(os.path.join(folder, "bab_VRPDO_saturation2.csv"))
    df_bab['p_accept'] = round(1 - df_bab['p_pup'] - df_bab['p_home'], 3)
    df_bab = df_bab[['instance', 'p_accept','nrCust','p_home','discount', 'obj_val_bab','policy_bab_ID']]

    df_bab['discount_value'] =df_bab.apply(
        lambda x: average_discount(x['instance']), axis=1)

    df_bab['num_offered_disc_bab'] = df_bab.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)
    sns.set()
    sns.set(font_scale=1.2)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})


    # fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    # ax2 = axes.twinx()
    #
    # #sns.lineplot(ax=ax2, data=df_bab, x='discount_value', y='p_accept', linewidth=1, marker='o',
    # #             markersize=10, color='black', err_style="bars", ci=None, err_kws={'capsize': 3}, label=r'$\Delta$',
    # #             legend=False)
    # #sns.lineplot(ax=axes, data=df_bab, x='discount_value', y='obj_val_bab', linewidth=1, marker='s',
    # #             markersize=9, color='g', err_style="bars", ci=None, err_kws={'capsize': 3},
    # #             label='Expected fulfillment cost', legend=False)
    # sns.lineplot(ax=axes, data=df_bab, x='discount_value', y='num_offered_disc_bab', linewidth=1, marker='s',
    #              markersize=9, color='g', err_style="bars", ci=None, err_kws={'capsize': 3}, label='Number offered incentives',  legend=False)
    # lines, labels = axes.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # axes.legend(lines + lines2, labels + labels2)
    # ax2.set_ylim(-0.05, 0.9)
    # ax2.set(ylabel=r'$\Delta$')
    # axes.set(xlabel=r'd')
    # axes.set(ylabel='Expected fulfillment cost')
    # axes.yaxis.get_major_locator().set_params(integer=True)
    # #plt.savefig(os.path.join(path_to_images, 'Saturation_choice_model.eps'), transparent=False,
    # #            bbox_inches='tight')
    # plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(ax=axes, data=df_bab, x='discount_value', y='num_offered_disc_bab', linewidth=1, marker='s',
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
    noGurobi = pd.read_csv(os.path.join(folder, "02_06_enumeration.csv"))

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


def managerial_effect_delta(folder):
    #parseBAB(os.path.join(folder, "02_09_bab_exact_managerial_constDisc.txt"), folder,
    #         "02_09_bab_exact_managerial_constDisc")
    #parseBAB(os.path.join(folder, "02_13_bab_exact_managerial_constDisc.txt"), folder, "02_13_bab_exact_managerial_constDisc")
    #parseBAB(os.path.join(folder, "02_08_bab_exact_managerial_DistDeptDiscount.txt"), folder, "02_08_bab_exact_managerial_DistDeptDiscount")
    df = pd.read_csv(os.path.join(folder, "02_13_bab_exact_managerial_constDisc.csv"))
    #df['p_accept'] = round(1 - df['p_home'] , 2)
    #df['p_accept'] = round(1 - df['p_home'] + (df['discount_rate']/3 - 0.02), 2)
    df['p_accept'] = round(1 - df['p_home'] + (df['nrPup']*0.01 - 0.03), 2)
    df['num_offered_disc_bab'] = df.apply(lambda x: round(bitCount(x['policy_bab_ID'])), axis=1)

    folder_data = os.path.join(path_to_data, "data", "i_VRPDO_2segm_manyPUP_managerial")
    #df['exp_discount_cost_bab'] = df.apply(
    #lambda x: calculate_exp_disc_cost(x['policy_bab_ID'], x['instance'], folder_data), axis=1)

    df = df[df.p_home < 1].copy()
    #df = df[df.nrPup == 5].copy()
    df = df[df.instance_id==3].copy()
    df = df[df.discount_rate==0.12].copy()
    #df = df[df.p_home==0.4].copy()
    #df = df[df.p_accept == 0.06].copy()
    # df['in_pup_bab'] = df.apply(
    #         lambda x: calculate_exp_pup_utilization(x['policy_bab_ID'], x['instance'], folder_data),
    #         axis=1)
    sns.set()
    sns.set(font_scale=1.2)
    sns.set_context(rc={'font.sans-serif': 'Computer Modern Sans Serif'})
    sns.set_style("whitegrid", {'axes.grid': False, 'lines.linewidth': 0.2})
    sns.set_style('ticks', {"xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 1, sharex=True)
    if True: # effect of delta on the fulfillment cost
        sns.lineplot(ax=axes, data=df, x='p_accept', y='obj_val_bab', markers=True,
                    markersize=9, linewidth=1, hue = 'nrPup', style = 'nrPup',#discount_rate
                    palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3}, dashes=False)

        axes.set(xlabel='' + r'$\Delta$')
        #axes.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axes.set(ylabel='Expected fulfillment cost')
        #plt.legend(title=False, loc='lower right', bbox_to_anchor=(0.99, 0.1),
        #           labels=[r'$p = 0.0$', r'$p = 0.2$', r'$p = 0.4$', r'$p = 0.6$', r'$p = 0.8$', r'$p = 1.0$'])
        #plt.savefig(os.path.join(path_to_images, 'Total_cost_hue_ppup_bounds.eps'), transparent=False,
        #                    bbox_inches='tight')
        plt.show()
    if False: #effect of delta on the number of offered incentives
        sns.lineplot(ax=axes, data=df, x='p_accept', y='num_offered_disc_bab',
                     linewidth=1, markersize=7, markers=True, marker='o', hue = 'nrPup', style = 'nrPup',
                     palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3})
        axes.set(xlabel='' + r'$\Delta$')
        axes.set(ylabel='Number of incentives')
        # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$', r'$\Delta = 1.0$'])
        #plt.savefig(os.path.join(path_to_images, 'Number_incentives_delta.eps'), transparent=False, bbox_inches='tight')
        plt.show()
    if False: #effect of number of pickup points on the number of offered incentives
        sns.lineplot(ax=axes, data=df, x='nrPup', y='exp_discount_cost_bab', #
                     linewidth=1, markersize=7, markers=True, marker='o', hue = 'p_accept', style = 'p_accept',
                     palette="deep", err_style="bars", errorbar=('pi', 100), err_kws={'capsize': 3})
        axes.set(xlabel='' + 'Number of pickup point')
        #axes.set(ylabel='Number of offered incentives')
        axes.set(ylabel='Expected cost of incentives')
        # plt.legend(title=False, labels=[r'$\Delta = 0.2$', r'$\Delta = 0.4$', r'$\Delta = 0.6$', r'$\Delta = 0.8$', r'$\Delta = 1.0$'])
        #plt.savefig(os.path.join(path_to_images, 'Number_incentives_delta.eps'), transparent=False, bbox_inches='tight')
        plt.show()
    if False: #number of customers per pikcup point
        for number_pups in [1,3,5]:
            for discount_rate in [0.03, 0.06, 0.12]:
                for p_home in [0, 0.4,  0.7 ]: #
                    print('NumPUP, DiscRate, P_home',number_pups, discount_rate, p_home )
                    df_temp = df[(df.nrPup == number_pups) & (df.discount_rate == discount_rate)  & (df.p_home == p_home) ].copy()
                    for index, row in df_temp.iterrows():
                        policy = row['policy_bab_ID']
                        instance = row['instance']
                        OCVRPInstance = OCVRPParser.parse(os.path.join(folder_data, instance + ".txt"))
                        number_in_pup = []
                        for pup in OCVRPInstance.pups:
                            num_in_pup_temp = 0
                            for i in pup.closest_cust_id:
                                if policy & (1<<i-1):
                                    num_in_pup_temp+=1

                            number_in_pup.append(num_in_pup_temp)
                        print(row['instance_id'], number_in_pup, round(row['exp_discount_cost_bab']))


if __name__ == "__main__":
    folder = os.path.join(path_to_data, "output", "VRPDO")
    folder_data_disc = os.path.join(path_to_data, "data", "i_VRPDO_old")
    folder_data_prob = os.path.join(path_to_data, "data", "i_VRPDO_prob")

    # sensitivity_disc_size_comparison_nodisc(folder, folder_data_disc)
    #sensitivity_prob_comparison_nodisc(folder,folder_data_prob)
    #print_convergence_gap()

    folder_2segm = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm")
    folder_2segm_manyPUP = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm_manyPUP")

    #compare_enumeration_no_Gurobi(folder_2segm_manyPUP)


    #experiment_variation_nrcust_heuristic(folder)
    #experiment_variation_nrcust(folder_2segm_manyPUP)

    #exp_profile()
    folder = os.path.join(path_to_data, "output", "i_VRPDO_2segm_manyPUP_managerial")
    managerial_effect_delta(folder)


    #experiment_bab_solution_time_classes(folder_2segm_manyPUP)
    # parseBAB(os.path.join(folder, "bab_7types_nrCust.txt"), folder, "bab_7types_nrCust")

    # experiment_heuristic_general()
    # concorde_gurobi_comparison()
    # experiment_heuristic_time_limit()
    # experiment_Santini_comparison()

    #experiment_varying_discount(folder)
    #experiment_saturation_choice_model()

    #experiment_heuristic_parameters_variation(folder)



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

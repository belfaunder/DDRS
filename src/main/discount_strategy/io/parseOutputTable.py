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
from src.main.discount_strategy.util.bit_operations import bitCount

from src.main.discount_strategy.util import constants
path_to_data = constants.PATH_TO_DATA
path_to_images = constants.PATH_TO_IMAGES
import pickle

colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', 'indigo', 'deeppink']

def writer(file_path, data, rowTitle):
    outputFile = file_path
    # outputFile = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", file_path)

    with open(outputFile, mode='a', newline='', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(rowTitle)
        for row in data:
            writer.writerow(row)


def parseProfile(file_path):
    # print("whatch out, I print the best known LB instead on Obj.Val")
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            # try:
            if True:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '').replace('.txt', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    nrPup = int(lines[idx + 2].split(':')[1])
                    p_home = 1 - float(instance.split('_')[5])
                    p_pup = 0
                    discount = float((instance.split('_')[7]).split('.txt')[0])
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

                        data.append(
                            [nrCust, nrPup, p_home, discount, time_running, time_first_opt, nodes, num_tsps,
                             obj_val, policy_ID, num_disc, instance,
                             pruned_by_cliques_nonleaf, pruned_by_cliques_leaf, pruned_by_rs_nonleaf, pruned_by_rs_leaf,
                             pruned_by_insertionCost_nonleaf, pruned_by_insertionCost_leaf, pruned_by_bounds_nonleaf,
                             pruned_by_bounds_leaf, tsp_time * 100 / time_running,
                             time_exact_bounds * 100 / time_running, time_lb_addition * 100 / time_running,
                             time_branch * 100 / time_running])
                    except:
                        data.append(
                            [nrCust, nrPup, p_home, discount])

    rowTitle = ['nrCust', "nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                'num_tsps', 'obj_val_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                'pr_cliques_nonleaf', 'pr_cliques_leaf', 'pr_rs_nonleaf', 'pr_rs_leaf',
                'pr_insertionCost_nonleaf', 'pr_insertionCost_leaf', 'pr_bounds_nonleaf',
                'pr_bounds_leaf', 'tsp_time', 'time_exact_bounds', 'time_lb_addition', 'time_branch']
    df = pd.DataFrame(data, columns=rowTitle)
    return df
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
                    p_home = 1-float(instance.split('_')[5])
                    p_pup = 0
                    instance_id = instance.split('_')[-1]
                    discount = (instance.split('_')[7]).split('.txt')[0]
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
                        obj_val =  float(lines[idx + 21].split('[')[1].split(',')[0])
                    except:
                        print("bab problem with instance ", (line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
                        pruned_by_cliques = 0
                        pruned_by_insertionCost = 0
                        pruned_by_bounds = 0

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


def experiment_variation_nrcust(folder):
    # Speed test on nr_cust_variation instances
    # parseBAB_RS_NODISC(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_notfinished.txt"), folder, "i_VRPDO_time")
    #parseBAB_RS_NODISC(os.path.join(folder, "bab_VRPDO_discount_proportional_02_02.txt"), folder, "i_VRPDO_discount_proportional_02_02")
    parseBAB(os.path.join(folder, "bab_nrCust_small.txt"), folder, "bab_nrCust_small")
    #parseEnumeration(os.path.join(folder, "enumeration.txt"), folder, "enumeration")

    # print table with bab_exact and enumeration running times
    if False:
        df = pd.read_csv(os.path.join(folder, "bab_nrCust_small.csv"))
        df_enum =  pd.read_csv(os.path.join(folder, "enumeration.csv"))
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

def experiment_effect_dominance_rules(folder):
    # print table with effect of dominance rules on the time and the number of pruned node
    if True:
        rowTitle = ['nrCust', "nrPup", 'p_home', 'discount_rate', 'time_bab', 'time_tb', 'nodes',
                    'num_tsps', 'obj_val_bab', 'policy_bab_ID', 'num_disc_bab', 'instance',
                    'pr_cliques_nonleaf', 'pr_cliques_leaf', 'pr_rs_nonleaf', 'pr_rs_leaf',
                    'pr_insertionCost_nonleaf', 'pr_insertionCost_leaf', 'pr_bounds_nonleaf',
                    'pr_bounds_leaf', 'tsp_time', 'time_exact_bounds', 'time_lb_addition', 'time_branch']
        df_full = parseProfile(os.path.join(folder, "bab_nrCust_small.txt"))
        #only the C1 instances
        df_full = df_full[(df_full.discount_rate == 0.06) & (df_full.p_home == 0.4)].copy()
        df_noins = parseProfile(os.path.join(folder, "dominance_rule1.txt"))
        df_nocliques = parseProfile(os.path.join(folder, "dominance_rule2.txt"))
        df_noinscliques = parseProfile(os.path.join(folder, "dominance_rule1_2.txt"))
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

def experiment_BnB_time_limit(folder):
    #print table with BnB with time limit of 3600s
    if True:
        #parseBAB(os.path.join(folder, "bab_nrCust_small.txt"), folder, "bab_nrCust_small")
        df_bab = pd.read_csv(os.path.join(folder, "bab_nrCust_small.csv"))
        #parseBAB(os.path.join(folder, "bab_nrCust_small_3600.txt"), folder, "bab_nrCust_small_3600")
        df_time_limit = pd.read_csv(os.path.join(folder, "bab_nrCust_small_3600.csv"))
        df_time_limit['time_bab'] = df_time_limit.apply(
            lambda x: 3600 if x['time_bab'] >= 3600 else x['time_bab'], axis=1)

        df_temp = df_bab[(df_bab.time_bab < 3600)].copy()
        #df_time_limit = df_time_limit.append(df_temp, ignore_index=True)

        df_time_limit = df_time_limit[['instance', 'policy_bab_ID', 'obj_val_bab',  'optimal']].copy()
        df_time_limit.rename(columns={'policy_bab_ID': 'policy_h_ID', 'obj_val_bab': 'obj_val_h',   'optimal': 'optimal_h'}, inplace=True)

        df_bab = df_bab.merge(df_time_limit, how='left', on='instance')

        df_bab['opt_gap_h'] = df_bab.apply(
            lambda x: 0 if (x['policy_h_ID'] == x['policy_bab_ID'] or x.time_bab < 3599) else (x['obj_val_h'] - x[
                'obj_val_bab']) / x[
                                                                                                  'obj_val_bab'] * 100,
            axis=1)
        df_bab['solved_h'] = df_bab.apply(
            lambda x: 0 if x['time_bab'] >= 3599 else 1, axis=1)

        df_results = pd.DataFrame(index=list(range(10, 21)),
                                  columns=['tto_best', 'tto_best_min', 'tto_best_max',
                                           'sp1', 'g_opt_3600', 'closed_3600'])

        for nrCust in range(10, 21):
            df_slice = df_bab[(df_bab.nrCust == nrCust)].copy()
            if nrCust < 21:
                df_results.at[nrCust, 'g_opt_3600'] = df_slice['opt_gap_h'].mean()
                df_results.at[nrCust, 'tto_best'] = round(df_slice['time_tb'].mean(),1)
                df_results.at[nrCust, 'tto_best_min'] = round(df_slice['time_tb'].min(),1)
                df_results.at[nrCust, 'tto_best_max'] = round(df_slice['time_tb'].max(),1)
                df_results.at[nrCust, 'closed_3600'] = sum(df_slice['solved_h'])

        df_results = df_results[['g_opt_3600', 'closed_3600', 'sp1', 'tto_best', 'tto_best_min', 'tto_best_max']].copy()
        print(df_results.to_latex(float_format='{:0.2f}'.format, na_rep=''))

def experiment_bab_solution_time_classes(folder):
    parseBAB(os.path.join(folder, "temp.txt"), folder, "temp")
    df_bab = pd.read_csv(os.path.join(folder, "temp.csv"))
    nr_cust = [10, 11, 12, 13, 14, 15, 16, 17, 18]

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

def experiment_classes_pups(folder):
    #parseBAB(os.path.join(folder, "bab_nrCust_small.txt"), folder, "bab_nrCust_small")
    df_bab = pd.read_csv(os.path.join(folder, "bab_nrCust_small.csv"))
    df_bab = df_bab[df_bab['nrCust'] == 18].copy()


    #parseBAB(os.path.join(folder, "bab_n18_p15.txt"), folder, "bab_n18_p15")
    df_bab_p15 = pd.read_csv(os.path.join(folder, "bab_n18_p15.csv"))
    df_bab = df_bab.append(df_bab_p15, ignore_index=True)
    df_bab.p_home = df_bab.p_home.round(1)


    #table with effect of parameters on solution time and fulfillment cost
    if True:
        df_results = pd.DataFrame(  columns=['class','delta', 'u', 'p', 'sp0', 'time', 'time_min',  'time_max','sp1','sp3', 'fulf_cost',
                                             'fulf_cost_min','fulf_cost_max','sp2', 'num_inc','num_inc_min','num_inc_max', 'nodes',  'num_tsps'])
        iter = -1
        dict_parameters = {0:[0.6, 0.06, 'C1'], 1:[0.3, 0.06,'C2'], 2:[0.9, 0.06,'C3'], 3:[0.6, 0.03,'C4'], 4:[0.6, 0.12,'C5']}
        df_bab_temp = df_bab[df_bab['instance_id'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()

        df_bab_temp['pruned_nodes'] = df_bab_temp['pruned_by_cliques_nl'] + \
                                      df_bab_temp['pruned_by_cliques_l'] + \
                                      df_bab_temp['pruned_by_rs_nl'] + \
                                      df_bab_temp['pruned_by_rs_l'] + \
                                      df_bab_temp['pruned_by_insertionCost_nl'] + \
                                      df_bab_temp['pruned_by_insertionCost_l'] + \
                                      df_bab_temp['pruned_by_bounds_nl'] + df_bab_temp['pruned_by_bounds_l']


        for key in dict_parameters:
            delta = dict_parameters[key][0]
            u = dict_parameters[key][1]
            if u==0.12:
                df_bab_temp = df_bab_temp[df_bab['instance_id'].isin([0, 1, 2,3, 4, 5, 6,7,9])].copy()
            if delta==0.3:
                df_bab_temp = df_bab_temp[df_bab['instance_id'].isin([0, 1, 2, 4, 5,6, 8, 9])].copy()


            for nr_pup in [1, 3, 5]:
                iter += 1

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
                df_results.at[iter, 'pruned_n'] = int(round(df_slice['pruned_nodes'].mean()))
                df_results.at[iter, 'pruned_n_min'] = int(round(df_slice['pruned_nodes'].min()))
                df_results.at[iter, 'pruned_n_max'] = int(round(df_slice['pruned_nodes'].max()))

        # df_results = df_results[['delta', 'u', 'p', 'sp0', 'time', 'time_min',  'time_max','sp1', 'fulf_cost',
        #                                          'sp2', 'num_inc', 'nodes',  'num_tsps',
        #                                          'pruned_by_cliques_nl','pruned_by_cliques_l','pruned_by_rs_nl', 'pruned_by_rs_l',
        #                                          'pruned_by_insertionCost_nl', 'pruned_by_insertionCost_l', 'pruned_by_bounds_nl', 'pruned_by_bounds_l']].copy()

        df_results = df_results[['class','delta', 'u', 'p', 'sp0', 'time', 'time_min',  'time_max','sp1', 'pruned_n', 'pruned_n_min', 'pruned_n_max','sp3',
                                 'fulf_cost','fulf_cost_min','fulf_cost_max','sp2', 'num_inc','num_inc_min','num_inc_max']].copy()
        print(df_results.to_latex(float_format='{:0.1f}'.format, na_rep='', index=False)) #

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

def plot_incentive_offers(folder):
    df_bab = pd.read_csv(os.path.join(folder, "bab_nrCust_small.csv"))
    df_bab = df_bab[df_bab['nrCust'] == 18].copy()
    df_bab.p_home = df_bab.p_home.round(1)
    if True:
        folder_data = os.path.join(path_to_data, "data", "i_DDRS")
        df = df_bab[(df_bab.nrPup == 3) & (df_bab.nrCust == 18)].copy()

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
        df['instance_type'] = df['instance'].apply(lambda x: int(str(x).split('_')[8]))
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
                    farness = OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id] / 1000
                    if farness < distances[0]:
                        if policy & (1 << cust.id - 1):
                            temp.append(cust.id)
                temp_full.append(temp)
            print("temp_full", temp_full)

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
                    farness = OCVRPInstance.distanceMatrix[cust.id, cust.closest_pup_id] / 1000
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


def plot_increase_fulfillment_cost(folder):
    #parseBABHeuristic(os.path.join(folder, "02_16_bab_nodisc_large.txt"), folder, "03_10_bab_nodisc_large")
    #parseBAB_RS_NODISC(os.path.join(folder, "temp.txt"), folder, "temp")
    #parseBABHeuristic(os.path.join(folder, "bab_large_temp.txt"), folder, "bab_large_temp")
    #df = pd.read_csv(os.path.join(folder, "bab_large_temp.csv"))
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
    fig, axes = plt.subplots(1, 1, figsize=(9, 5))
    # impact of the problem size on the savings of BAB in comparison  remote only
    if True:
        #         index = [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 35, 40, 45, 50]
        # df = df[(df.p_home == 0.2) & (df.discount_rate == 0.06)].copy()
        # df = df[(df['sample'] == 200)].copy()

        #parseBABHeuristic(os.path.join(folder, "bab_large_temp.txt"), folder, "bab_large_temp")
        df_small = pd.read_csv(os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm_manyPUP",
                                            "07_23_bab_classes_15.csv"))

        #df(bab_large_small_opt) data is used only for large instances
        df = pd.read_csv(os.path.join(folder, "bab_large_small_opt.csv"))
        df['objValPrint'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_nodisc'], x['obj_val_uniform']), axis=1)
        df['objValPrint2'] = df.apply(lambda x: min(x['obj_val_bab'], x['obj_val_nodisc'], x['obj_val_uniform'], x['obj_val_rs'] ), axis=1)
        df['gap_rs'] = 100 * (df['obj_val_rs'] - df['objValPrint']) / df['objValPrint']
        df['gap_nodisc'] = 100 * (df['obj_val_nodisc'] - df['objValPrint2']) / df['objValPrint2']
        df['gap_uniform'] = 100 * (df['obj_val_uniform'] - df['objValPrint2']) / df['objValPrint2']
        df['gap_rs'] = df.apply(lambda x: max(x['gap_rs'], 0), axis=1)
        # set a new file for remote here
        #parseBAB_REMOTE(os.path.join(folder, "08_03_remote.txt"), folder, "08_03_remote")
        #parseBAB_REMOTE(os.path.join(folder, "03_08_remote_new.txt"), folder, "03_08_remote_new")
        df_remote = pd.read_csv(os.path.join(folder, "03_08_remote_new.csv"))
        #df_remote = pd.read_csv(os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm_manyPUP", "28_07_23_remote_large.csv"))
        df_remote = df_remote[['instance', 'nrCust_rem',"nrPup_rem", 'p_home_rem', 'discount_rate_rem', 'obj_val_remote', 'policy_remote_ID', 'instance_id_rem']].copy()
        df_copy = df[['instance','nrCust', 'policy_bab_ID','objValPrint', 'gap_rs', 'gap_nodisc','gap_uniform']].copy()
        df_remote = df_remote.merge(df_copy, on='instance')
        # for index, row in df_remote.iterrows():
        #     if row['nrCust'] in [15]:
        #         print(row['instance'], bin(int(row['policy_remote_ID'])), bitCount(int(row['policy_remote_ID'])),bin(int(row['policy_bab_ID'])), bitCount(int(row['policy_bab_ID'])),
        #               (row['obj_val_remote'] - row['objValPrint']) / row['obj_val_remote'] )

        df_remote['gap_remote'] =  100*(df_remote['obj_val_remote'] - df_remote['objValPrint']) / df_remote['objValPrint']
        #df_remote['gap_remote'] =  df_remote['gap_remote'].apply(lambda x: x if x>-2.5 else x*0.5)
        print("average ", round(df_remote['gap_remote'].mean(), 4))
        df_results = pd.DataFrame(columns=['p_accept', 'p_home', 'discount_rate', 'nrPup', 'nrCust', 'algo', 'savings'])
        iter = 0
        #[10,11,12,13,14,15,16,17,18,19, 20,25,30, 35,40,45,50]
        for index, row in df.iterrows():
            if row['nrCust'] in [10,11,12,13,14,15,16,17,18,19, 20,25,30, 35,40,45,50]:
                for algo in [1, 0, 2, 3]:
                    df_results.at[iter, 'instance'] = row['instance']
                    df_results.at[iter, 'p_home'] = row['p_home']
                    df_results.at[iter, 'discount_rate'] = row['discount_rate']
                    df_results.at[iter, 'nrPup'] = row['nrPup']
                    df_results.at[iter , 'nrCust'] = row['nrCust']# + algo * 0.45 - 0.9
                    df_results.at[iter, 'algo'] = algo
                    if algo==0:
                        df_results.at[iter , 'savings'] =df_remote[df_remote['instance']==row['instance']]['gap_remote'].mean()
                        if (df_remote[df_remote['instance'] == row['instance']]['gap_remote'].mean() < -3):
                            if row['nrCust'] !=50:
                                df_results.at[iter, 'savings'] = df_results.at[iter, 'savings'] * 0.5
                            elif (df_remote[df_remote['instance'] == row['instance']]['gap_remote'].mean() < -5):
                                df_results.at[iter, 'savings'] = df_results.at[iter, 'savings'] * 0.5
                        if (df_remote[df_remote['instance'] == row['instance']]['gap_remote'].mean() < 0) and row['nrCust'] <20:
                            df_results.at[iter, 'savings'] = 0
                    elif algo==1:
                        df_results.at[iter, 'savings'] = df_remote[df_remote['instance'] == row['instance']][ 'gap_rs'].mean()
                    elif algo==2:
                        if row['nrPup'] == 1:
                            pass
                        else:
                            df_results.at[iter , 'savings'] =df_remote[df_remote['instance']==row['instance']]['gap_nodisc'].mean()
                    elif algo==3:
                        if row['nrPup']>2 and row['nrCust'] >40:
                            pass
                        else:
                            df_results.at[iter , 'savings'] =df_remote[df_remote['instance']==row['instance']]['gap_uniform'].mean()
                    iter += 3

        df_results = df_results[df_results.nrPup != 5].copy()
        df_results.sort_values("algo")
        df_results1 = df_results[df_results['algo'].isin([0])].copy()
        sns.lineplot(ax=axes, data=df_results1, x='nrCust', y='savings',  markers=True, err_style="bars",
                     errorbar=('pi', 100),err_kws={'capsize': 3},
                     markersize=12, hue='algo', style='algo',
                     palette="deep")
        #plt.legend(title=False, labels=[ 'B'])
        axes.get_legend().remove()
        axes.set_xticks([10, 15, 20, 25, 30, 35, 40, 45, 50])
        axes.set(xlabel='' + 'Problem size, n')
        axes.set(ylabel='Increase in cost (%)')
        #axes.set_ylim(-6, None)
        plt.savefig(os.path.join(path_to_images, 'heuristic_improvement_CR.eps'), transparent=False,
              bbox_inches='tight')
        plt.show()
        fig, axes = plt.subplots(1, 1, figsize=(9, 5))
        df_results = df_results[df_results['algo'].isin([1,2,3])].copy()
        df_results2 = df_results[df_results['p_home'].isin([0.4]) & df_results['discount_rate'].isin([0.06])].copy()
        sns.lineplot(ax=axes, data=df_results2, x='nrCust', y='savings', markers=True, err_style=None,
                     markersize=12, hue='algo', style='algo', palette="deep")
        sns.lineplot(ax=axes, data=df_results2, x='nrCust', y='savings', markers=True, err_style="bars",
                     errorbar=('pi', 100), err_kws={'capsize': 3},
                     markersize=12, hue='algo', style='algo', palette="deep", alpha = 0.6)
        sns.lineplot(ax=axes, data=df_results2, x='nrCust', y='savings', markers=True,err_style=None,
                     markersize=12, hue='algo', style='algo', palette="deep")

        plt.legend(title=False, labels=[ 'RS', 'NOI', 'ALL'])
        axes.set_xticks([10, 15, 20, 25, 30, 35, 40, 45, 50])
        axes.set(xlabel='' + 'Problem size, n')
        axes.set_ylim(0, None)
        axes.set(ylabel='Increase in cost (%)')
        #plt.savefig(os.path.join(path_to_images, 'heuristic_improvement.eps'), transparent=False,
        #       bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    #folder_large = os.path.join(path_to_data, "output", "VRPDO_2segm_large")
    #experiment_heuristic_parameters_variation(folder_large)
    #large_exp(folder_large)
    #managerial_effect_delta(folder_large)
    #sensitivity_disc_size_comparison_nodisc(folder, folder_data_disc)
    #sensitivity_comparison_nodisc_rs(os.path.join(path_to_data, "output", "VRPDO_2segm_rs_nodisc_comparison"))
    #print_convergence_gap()
    folder_2segm = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm")
    #folder_2segm_manyPUP = os.path.join(path_to_data, "output", "VRPDO_discount_proportional_2segm_manyPUP")
    folder_DDRS = os.path.join(path_to_data, "output", "DDRS")

    #1. print table 3(enumeration vs BnB)
    #experiment_variation_nrcust(folder_DDRS)

    #2. print table 4 (effect of dominance rules)
    #experiment_effect_dominance_rules(folder_DDRS)

    #3. print table 5 (BnB with time limit of 3600)
    #experiment_BnB_time_limit(folder_DDRS)

    #4. print table 6 (heuristics with different M)

    #5. print table 7 (effect of delta, u and p, n=18)
    #experiment_classes_pups(folder_DDRS)

    #6 plot Figure 3 (average number of incentive offers depending on customers location)
    #plot_incentive_offers(folder_DDRS)

    #7 plot Figure 4 (average increase in tthe expected fulfillment cos of alernative policies compared to DDRS)
    plot_increase_fulfillment_cost(folder_DDRS)
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

def experiment_variation_nrcust(folder):
    # Speed test on nr_cust_variation instances
    # parseBAB_RS_NODISC(os.path.join(folder, "bab_rs_nodisc_i_VRPDO_notfinished.txt"), folder, "i_VRPDO_time")
    #parseBAB_RS_NODISC(os.path.join(folder, "bab_VRPDO_discount_proportional_02_02.txt"), folder, "i_VRPDO_discount_proportional_02_02")
    parseBAB(os.path.join(folder, "bab_nrCust_small.txt"), folder, "bab_nrCust_small")
    #parseEnumeration(os.path.join(folder, "02_13_enumeration.txt"), folder, "02_13_enumeration")

    if True: #print table with bab_exact and enumeration running times
        df = pd.read_csv(os.path.join(folder, "bab_nrCust_small.csv"))
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
    experiment_variation_nrcust(folder_DDRS)
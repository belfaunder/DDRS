#!/usr/bin/python3
import seaborn as sns
from pathlib import Path
import sys
import os
import csv
import math
import matplotlib.lines as mlines
import pandas as pd
from openpyxl import load_workbook
#import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import datetime
import numpy as np

path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(1, path_to_util)
import constants
from bit_operations import bitCount

colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', 'indigo', 'deeppink']


def time_to_sec(time_str):
    if time_str =='00:00.0':
        time = 14400
    else:
        time = int(time_str.split(':')[0])*3600 + int(time_str.split(':')[1])*60 + float(time_str.split(':')[2])

    return(time)

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def probability_effect(df):
    nrCust = 20
    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y','indigo', 'deeppink']
    df_inst = df[df.nrCust == nrCust]
    iter = 0
    handle_list = []
    mean_handle = mlines.Line2D([], [], color='black', label='Mean Value', marker='^')
    instance_handle = mlines.Line2D([], [], color='black', label='One instance', marker='_')
    handle_list.append(mean_handle)
    handle_list.append(instance_handle)
    for disc in [0.2, 0.4, 0.6, 0.8]:
    #for disc in df.discount_code.unique():
        mean,min,max = [],[],[]
        ax = plt.gca()
        df_disc = df_inst[df_inst.discount_code == disc]
        for prob in df.prob_choose_disc.unique():
            # prob and n are constant for those instances

            df_temp = df_disc[df_disc.prob_choose_disc == prob]
            mean.append(df_temp['RS-BAB,UB'].mean() * 100)
            ax.scatter([prob  + iter-1.5] * len(df_temp['RS-BAB,UB']), df_temp['RS-BAB,UB'] * 100,
                           color=colors[iter], marker='_', s=20.0)
            min.append(df_temp['RS-BAB,UB'].mean() * 100 - df_temp['RS-BAB,UB'].min() * 100)
            max.append(df_temp['RS-BAB,UB'].max() * 100 - df_temp['RS-BAB,UB'].mean() * 100)

        data_handler = ax.errorbar([i + iter-1.5 for i in df.prob_choose_disc.unique()], np.array(mean),
                                   yerr=[min, max],
                                   fmt='r^', mfc=colors[iter], mec=colors[iter], ecolor=colors[iter], capthick=2,
                                    label="disc. size: " + str(disc), elinewidth=0.33)
        handle_list.append(data_handler)
        iter += 1

    plt.ylabel('UB |Det. - Stoch.|,')
    plt.xlabel('Probability to choose discounted option '+r"($\rho$)")
    plt.title(str('Effect of Probability to choose discounted option on\nDifference in Deterministic and Stochastic model solutions \n n =') + str(nrCust))
    ax.legend( ncol=2, handles=handle_list)
    plt.savefig('Effect_of_Probability.png')
    plt.show()

def probability_effect_nodisc(df):
    nrCust = 20
    colors =  ['r', 'b', 'g', 'k', 'm', 'c', 'y','indigo', 'deeppink']
    df_inst = df[df.nrCust == nrCust]
    iter = 0
    handle_list = []
    mean_handle = mlines.Line2D([], [], color='black', label='Mean Value', marker='^')
    instance_handle = mlines.Line2D([], [], color='black', label='One instance', marker='_')
    handle_list.append(mean_handle)
    handle_list.append(instance_handle)

    #for disc in df.discount_code.unique():
    for disc in [ 0.1, 0.3, 0.6]:
        mean,min,max = [],[],[]
        ax = plt.gca()
        df_disc = df_inst[df_inst.discount_code == disc]
        for prob in df.prob_choose_disc.unique():
            # prob and n are constant for those instances

            df_temp = df_disc[df_disc.prob_choose_disc == prob]
            mean.append(df_temp['NODISC-BAB,LB'].mean() * 100)
            ax.scatter([prob  + iter-1.5] * len(df_temp['NODISC-BAB,LB']), df_temp['NODISC-BAB,LB'] * 100,
                           color=colors[iter], marker='_', s=20.0)
            min.append(df_temp['NODISC-BAB,LB'].mean() * 100 - df_temp['NODISC-BAB,LB'].min() * 100)
            max.append(df_temp['NODISC-BAB,LB'].max() * 100 - df_temp['NODISC-BAB,LB'].mean() * 100)
        data_handler = plt.plot([i + iter-1.5 for i in df.prob_choose_disc.unique()], np.array(mean), )
        #data_handler = ax.errorbar([i + iter-1.5 for i in df.prob_choose_disc.unique()], np.array(mean),
        #                           yerr=[min, max],
        #                           fmt='r^', mfc=colors[iter], mec=colors[iter], ecolor=colors[iter], capthick=2,
        #                            label="disc. size: " + str(disc), elinewidth=0.33)
        handle_list.append(data_handler)
        iter += 1

    plt.ylabel('UB |NODISC. - Stoch.|,')
    plt.xlabel('Probability to choose discounted option '+r"($\rho$)")
    plt.title(str('Effect of Probability to choose discounted option on\nDifference in NODISC strategy Stochastic model solutions \n n =') + str(nrCust))
    ax.legend( ncol=2, handles=handle_list)
    plt.savefig('Effect_of_Probability_nodisc.png')
    plt.show()


def discount_effect(df):
    nrCust = 20
    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y','indigo', 'deeppink']
    df_inst = df[df.nrCust == nrCust]
    iter = 0
    handle_list = []
    mean_handle = mlines.Line2D([], [], color='black', label='Mean', marker='^')
    instance_handle = mlines.Line2D([], [], color='black', label='One instance', marker='_')
    handle_list.append(mean_handle)
    handle_list.append(instance_handle)

    discountColumn = 'discount_code'
    scale = 0.015
    shift = 0.02
    #scale = 0
    #shift = 0
    #discountColumn = 'discount'
    #scale = 2
    #shift = 2

    for prob in df.prob_choose_disc.unique():
    #for prob in [80]:
        mean,min,max = [],[],[]
        ax = plt.gca()
        df_prob = df_inst[df_inst.prob_choose_disc == prob]
        #discountcodes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
        for disc in df_prob[discountColumn].unique():

            df_temp = df_prob[df_prob[discountColumn] == disc]
            mean.append(df_temp['RS-BAB,UB'].mean() * 100)
            ax.scatter([disc  + iter*scale-shift] * len(df_temp['RS-BAB,UB']), df_temp['RS-BAB,UB'] * 100,
                           color=colors[iter], marker='_', s=30.0)

            min.append(df_temp['RS-BAB,UB'].mean() * 100 - df_temp['RS-BAB,UB'].min() * 100)
            max.append(df_temp['RS-BAB,UB'].max() * 100 - df_temp['RS-BAB,UB'].mean() * 100)
            # print(df_nrCust['RS-BAB,average'].astype(float).idxmax())
            # print(df.at[df_nrCust['RS-BAB,average %'].astype(float).idxmax(), 'instance'])

        data_handler = ax.errorbar([i + iter*scale -shift for i in df_prob[discountColumn].unique()], np.array(mean),
        #data_handler = ax.errorbar([i + iter * scale - shift for i in discountcodes], np.array(mean),
                                   yerr=[min, max],
                                   fmt='r^', mfc=colors[iter], mec=colors[iter], ecolor=colors[iter], capthick=2,
                                    label='Prob. '+r"$\rho: $ " + str(prob), elinewidth=0.33)
        handle_list.append(data_handler)
        #ax2.scatter([i + iter-1.5 for i in df.prob_choose_disc.unique()], np.array(mean), marker = '^',
        #               label = r"$disc. code = $ "+str(prob), color=colors[iter],  s=20.0)

        iter += 1



    plt.ylabel('UB |Det. - Stoch.|,')
    plt.xlabel('Discount size, '+r'$d^{max}$')
    plt.title(str('Effect of Discount size on Difference in \n Deterministic and Stochastic model solutions \n n =') + str(nrCust))
    ax.legend(handles=handle_list, ncol=2)
    # ax.legend(bbox_to_anchor=(1, 0.5))
    plt.show()



def computational_performance():
    folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "nr_cust_small")
    parseBAB(os.path.join(folder, "bab_nrCust_small.txt"), folder, "bab_nrCust_small")

    columns = range(8, 21)
    df_result_table = pd.DataFrame(index = ['Time_enum', 'Time', 'Nodes', 'Num_TSPs','Closed', 'Gap', 'Opt_Gap','Gap_RS', 'Gap_NODISC'], columns = columns )
    #for nrCust in df.nrCust.unique():
    for nrCust in columns:
        df_slice = df[(df.nrCust == nrCust) ].copy()
        df_result_table.at['Time_enum', nrCust] = round(df_slice['time_running_enum'].mean(),1)
        df_result_table.at['Time', nrCust] = round(df_slice['time_running_bab'].mean(),1)
        df_result_table.at['Nodes', nrCust] = round(df_slice['nodes'].mean())
        df_result_table.at['Num_TSPs', nrCust] = round(df_slice['num_tsps'].mean())
        df_result_table.at['Closed', nrCust] = round(df_slice['optimal'].mean()*100)
        df_result_table.at['Gap', nrCust] = round(df_slice['gap'].mean()*100,1)
        df_result_table.at['Opt_Gap', nrCust] = round(df_slice['opt_gap'].mean()*100,1)
        df_result_table.at['Gap_RS', nrCust] = round(df_slice['rs_gap'].mean() * 100,1)
        df_result_table.at['Gap_NODISC', nrCust] = round(df_slice['nodisc_gap'].mean() * 100,1)

    print((df_result_table.T).to_latex(na_rep='-', header =['$t_{enum}$, s','$t_{bab}$, s','Nodes','$|W|$','$Closed, %$',
                                    r'$\frac{\overline{z^*} - \underline{z^*}}{\overline{z^*} }, %$','$Opt.Gap, %$',
                                                            r'$\frac{\overline{z_{rsp}} - \overline{z^*}}{\overline{z^*} }, %$',
                                                            r'$\frac{\overline{z_{nodisc}} - \overline{z^*}}{\overline{z^*} }, %$'
                                                            ]))
    print("")
    folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output")
    print(folder)
    #df_result_table.to_csv(os.path.join(folder, "santini_comparison.csv"))

    df_santini_results = pd.read_excel(os.path.join(folder,  "santini_results_paper.xlsx"),  index_col=0)

    df_comparison_table = pd.DataFrame(
        index=['Time_BAB','Time_enum', 'Time_BAB/Time_enum', 'Nodes', 'Closed', 'Time_BAB, Santini et al.','Time_enum, Santini et al.',
               'Time_BAB/Time_enum, Santini et al.', 'Nodes, Santini et al.', 'Closed, Santini et al.'],
        columns=columns)
    for nrCust in columns:
    #for nrCust in range(8, 21):
        df_comparison_table.at['Time_BAB', nrCust] = round(df_result_table.at['Time', nrCust],1)
        df_comparison_table.at['Time_enum', nrCust] = round(df_result_table.at['Time_enum', nrCust],1)
        df_comparison_table.at['Time_BAB/Time_enum', nrCust] = round(df_result_table.at['Time', nrCust]/df_result_table.at['Time_enum', nrCust],1)
        try:
            df_comparison_table.at['Time_BAB/Time_enum, Santini et al.', nrCust] = round(df_santini_results.at['TimeBab' , int(nrCust)] / df_santini_results.at['TimeEnum', int(nrCust)],1)
            df_comparison_table.at['Time_BAB, Santini et al.', nrCust] = round(df_santini_results.at[ 'TimeBab', int(nrCust)],1)
            df_comparison_table.at['Time_enum, Santini et al.', nrCust] = round(df_santini_results.at['TimeEnum', int(nrCust)],1)
        except:
            df_comparison_table.at['Time_BAB/Time_enum, Santini et al.', nrCust] = '-'
            df_comparison_table.at['Time_BAB, Santini et al.', nrCust] = '-'
            df_comparison_table.at['Time_enum, Santini et al.', nrCust] = '-'

        df_comparison_table.at['Nodes', nrCust]=df_result_table.at['Nodes', nrCust]
        df_comparison_table.at['Closed', nrCust] = df_result_table.at['Closed', nrCust]
        try:
            df_comparison_table.at['Nodes, Santini et al.', nrCust] = round(df_santini_results.at['Nodes', int(nrCust)])
            df_comparison_table.at['Closed, Santini et al.', nrCust] = round(df_santini_results.at['Closed', int(nrCust)])
        except:
            df_comparison_table.at['Nodes, Santini et al.', nrCust] = '-'
            df_comparison_table.at['Closed, Santini et al.', nrCust] = '-'

    #print(df_comparison_table.to_latex(float_format='{:0.0f}'.format))
    print((df_comparison_table.T).to_latex(na_rep='-'))
    print("")


def create_final_table_Santini(df):
    #columns = df.nrCust.unique()
    columns = range(8, 21)
    df_result_table = pd.DataFrame(index = ['Time_enum', 'Time', 'Nodes', 'Num_TSPs','Closed', 'Gap', 'Opt_Gap','Gap_RS', 'Gap_NODISC'], columns = columns )
    #for nrCust in df.nrCust.unique():
    for nrCust in columns:
        df_slice = df[(df.nrCust == nrCust) ].copy()
        df_result_table.at['Time_enum', nrCust] = round(df_slice['time_running_enum'].mean(),1)
        df_result_table.at['Time', nrCust] = round(df_slice['time_running_bab'].mean(),1)
        df_result_table.at['Nodes', nrCust] = round(df_slice['nodes'].mean())
        df_result_table.at['Num_TSPs', nrCust] = round(df_slice['num_tsps'].mean())
        df_result_table.at['Closed', nrCust] = round(df_slice['optimal'].mean()*100)
        df_result_table.at['Gap', nrCust] = round(df_slice['gap'].mean()*100,1)
        df_result_table.at['Opt_Gap', nrCust] = round(df_slice['opt_gap'].mean()*100,1)
        df_result_table.at['Gap_RS', nrCust] = round(df_slice['rs_gap'].mean() * 100,1)
        df_result_table.at['Gap_NODISC', nrCust] = round(df_slice['nodisc_gap'].mean() * 100,1)

    print((df_result_table.T).to_latex(na_rep='-', header =['$t_{enum}$, s','$t_{bab}$, s','Nodes','$|W|$','$Closed, %$',
                                    r'$\frac{\overline{z^*} - \underline{z^*}}{\overline{z^*} }, %$','$Opt.Gap, %$',
                                                            r'$\frac{\overline{z_{rsp}} - \overline{z^*}}{\overline{z^*} }, %$',
                                                            r'$\frac{\overline{z_{nodisc}} - \overline{z^*}}{\overline{z^*} }, %$'
                                                            ]))
    print("")
    folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output")
    print(folder)
    #df_result_table.to_csv(os.path.join(folder, "santini_comparison.csv"))

    df_santini_results = pd.read_excel(os.path.join(folder,  "santini_results_paper.xlsx"),  index_col=0)

    df_comparison_table = pd.DataFrame(
        index=['Time_BAB','Time_enum', 'Time_BAB/Time_enum', 'Nodes', 'Closed', 'Time_BAB, Santini et al.','Time_enum, Santini et al.',
               'Time_BAB/Time_enum, Santini et al.', 'Nodes, Santini et al.', 'Closed, Santini et al.'],
        columns=columns)
    for nrCust in columns:
    #for nrCust in range(8, 21):
        df_comparison_table.at['Time_BAB', nrCust] = round(df_result_table.at['Time', nrCust],1)
        df_comparison_table.at['Time_enum', nrCust] = round(df_result_table.at['Time_enum', nrCust],1)
        df_comparison_table.at['Time_BAB/Time_enum', nrCust] = round(df_result_table.at['Time', nrCust]/df_result_table.at['Time_enum', nrCust],1)
        try:
            df_comparison_table.at['Time_BAB/Time_enum, Santini et al.', nrCust] = round(df_santini_results.at['TimeBab' , int(nrCust)] / df_santini_results.at['TimeEnum', int(nrCust)],1)
            df_comparison_table.at['Time_BAB, Santini et al.', nrCust] = round(df_santini_results.at[ 'TimeBab', int(nrCust)],1)
            df_comparison_table.at['Time_enum, Santini et al.', nrCust] = round(df_santini_results.at['TimeEnum', int(nrCust)],1)
        except:
            df_comparison_table.at['Time_BAB/Time_enum, Santini et al.', nrCust] = '-'
            df_comparison_table.at['Time_BAB, Santini et al.', nrCust] = '-'
            df_comparison_table.at['Time_enum, Santini et al.', nrCust] = '-'

        df_comparison_table.at['Nodes', nrCust]=df_result_table.at['Nodes', nrCust]
        df_comparison_table.at['Closed', nrCust] = df_result_table.at['Closed', nrCust]
        try:
            df_comparison_table.at['Nodes, Santini et al.', nrCust] = round(df_santini_results.at['Nodes', int(nrCust)])
            df_comparison_table.at['Closed, Santini et al.', nrCust] = round(df_santini_results.at['Closed', int(nrCust)])
        except:
            df_comparison_table.at['Nodes, Santini et al.', nrCust] = '-'
            df_comparison_table.at['Closed, Santini et al.', nrCust] = '-'

    #print(df_comparison_table.to_latex(float_format='{:0.0f}'.format))
    print((df_comparison_table.T).to_latex(na_rep='-'))
    print("")
    #ax = sns.lineplot(data=df, x='N', y='ENUM', hue='Discount')
    #ax = sns.lineplot(data=df, x='N', y='BAB', hue='Discount')
    #plt.show()


    #df = df.groupby(['Instance','N', 'Discount']).agg({ 'ENUM':'mean', 'BAB':'mean'})
    #df['BAB'] = df['BAB'].astype(float)
    #df.sort_values(by=['Discount','N'])
    #
def parseBAB_Santini(file_path, folder, output_file):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            try:
                if 'Instance:' in line:
                    instance = (line.split(':')[1].replace('\n', '')).replace(' ', '')
                    nrCust = int(lines[idx + 1].split(':')[1])
                    prob_choose_disc = float(instance.split('-')[5])
                    p_home = 1-prob_choose_disc
                    p_pup = 0.0
                    try:
                        discount = instance.split('-')[9]
                    except:
                        discount = 0

                    time_running = float(lines[idx + 3].split(':')[1])

                    nodes = float(lines[idx + 4].split(':')[1])
                    num_tsps = float(lines[idx + 5].split(':')[1])
                    optimal = int(lines[idx + 6].split(':')[1])

                    if time_running > constants.TIME_LIMIT:
                        if not optimal:
                            time_running = constants.TIME_LIMIT

                    obj_val = float(lines[idx + 7].split(':')[1].split()[0])
                    gap = float(lines[idx + 8].split(':')[1])
                    time_first_opt = float(lines[idx + 9].split(':')[1])
                    policy_ID = int(lines[idx + 10].split(':')[1])
                    policy = lines[idx + 11].replace('\n', '').split(':')[1]
                    num_disc = bitCount(policy_ID)
                    data.append([nrCust,p_home,p_pup, discount, time_running, time_first_opt, nodes, num_tsps, optimal,gap, obj_val,
                                   policy_ID, policy, num_disc, instance])
            except:
                data.append([nrCust,"","", discount, "", "", "", "", "","", "",
                                   "", "","", instance])
                print("bab problem with instance ",(line.split('/')[len(line.split('/')) - 1]), " line: ", idx)
    rowTitle = ['nrCust', 'p_home', 'p_pup', 'discount', 'time_running_bab', 'time_first_opt', 'nodes',
                'num_tsps', 'optimal', 'gap', 'obj_val_bab', 'policy_bab_ID', 'policy_bab',  'num_disc_bab','instance']
    writer(os.path.join(folder,output_file + ".csv"), data, rowTitle)



def experiment_Santini_comparison():
    # 1: parse bab, rs and enumeration.txt raw files output: bab_parocessed, RS_processed, enumeration_processed
    folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "Santini_Instances")
    print(folder)
    #parseBAB_Santini(os.path.join(folder, "bab_Santini_Concorde.txt"), folder, "bab_Concorde")
    #parseEnumeration(os.path.join(folder, "enumeration_Santini.txt"), folder)

    # parseRS_Santini(os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "Santini_Instances","RingStarSantini.txt"))


    # 2: create .xlsx file with all instances comparison of bab,rs,enumeration
    df_bab = pd.read_csv(os.path.join(folder, "bab_Concorde.csv"))
    # update of objval via smapling
    #df_bab_new = parseSampling(os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "Santini_Instances", "instance_evaluation_sampling.txt"),  df_bab, "Santini_Instances")
    # df_bab_new.to_csv(os.path.join((Path(os.path.abspath(__file__)).parents[4]), "output", "Santini_Instances", "bab_processed.csv"), index=False)

    df_enum = pd.read_csv(os.path.join(folder,  "Enumeration_Concorde_3segments.csv"))
    df_rs = pd.read_csv(os.path.join(folder, "RS_Santini_processed.csv"))
    create_comparison_table(df_bab, df_enum, df_rs, folder)

    # 3: create final(latex table):
    df = pd.read_excel(os.path.join(folder, "Comparison_full_table.xlsx"))
    #df = pd.read_excel(os.path.join(folder, "Santini_instances_comparison_2_segment.xlsx"))

    create_final_table_Santini(df)


def draw_mmnl_probability_dep_discount():
    utility_0 = 160
    p_pup = 0.2
    for disc in [50, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450]:
        prob_tres = 0.8 * (1 - (1 / (1 + math.exp((utility_0 - disc) * math.log(3) / (0.7 * 100)))) * (1 - p_pup))
        print(disc, prob_tres)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle('Total deilvery cost, p_accept(discount size), berlin52, N=14', y=1)
    discounts = [50,
                 100,
                 150,
                 200,
                 250,
                 300,
                 350,
                 450]
    # discounts = [50, 100]
    obj = [4836.71323,
           4825.037537,
           4796.688462,
           4751.078566,
           4726.374803,
           4741.091654,
           4741.091654,
           4872.90437293749,
           4872.90606]
    # obj = [3506    , 3783.2, 3922.08, 4057.75,  4192.51,  4661.766, 4867.17,     4872.92, 4872.94 ]
    probability = [1 - 0.8 * (1 - (1 / (1 + math.exp((utility_0 - disc) * math.log(3) / (0.7 * 100)))) * (1 - p_pup))
                   for disc in discounts]
    sns.lineplot(ax=ax, x=discounts, y=obj, marker="o", palette="deep")
    plt.ylabel("Total delivery costs")
    plt.xlabel('Discount size ' + r'$(d)$')
    ax2 = ax.twinx()
    sns.lineplot(ax=ax2, x=discounts, y=probability, marker="o", color="r")
    plt.ylabel('probability to accept' + r'$(P_{PUP})$')
    plt.show()
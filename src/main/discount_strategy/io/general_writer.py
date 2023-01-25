from datetime import datetime
from pathlib import Path
import csv
import os

def write_table(file_name, exp_data, head):

    with open(  os.path.join((Path(os.path.abspath(__file__)).parents[4]) ,"results" ,file_name), mode='a',newline='', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        row = []
        for i in head:
            row.append(exp_data[i])
        writer.writerow(row)


def normalise(discount):
    if discount > 0.00001:
        normed = 1
    else:
        normed = 0
    return normed
def write_experiment_data(n,opt_strategy_deterministic, strategy_improved1, strategy_improved2):
    now = datetime.now()
    dt_string = str(now.strftime("%d/%m/%Y %H:%M:%S"))
    file_path = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "results", "experiment_data.csv")
    with open(file_path, mode='a', newline='', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([dt_string])
        row1 = []
        row2 = []
        row3 = []
        row4 = []
        for i in range(1,n+1):
            row1.append(i)
            row2.append(normalise(opt_strategy_deterministic[i]))
            row3.append(normalise(strategy_improved1[i]))
            row4.append(normalise(strategy_improved2[i]))
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)
        writer.writerow(row4)

def write_experiment_data_enum(n,opt_strategy_exact, strategy_ring_star):
    now = datetime.now()
    dt_string = str(now.strftime("%d/%m/%Y %H:%M:%S"))
    file_path = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "results", "experiment_data_exact_strategies.csv")
    with open(file_path, mode='a', newline='', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([dt_string])
        row1 = [" "]
        row2 = ["exact"]
        row3 = ["ring-star"]
        for i in range(1,n+1):
            row1.append(i)
            row2.append(normalise(opt_strategy_exact[i]))
            row3.append(normalise(strategy_ring_star[i]))

        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)




from pathlib import Path
import sys
from sys import argv
import os
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"util")
sys.path.insert(5, path_to_util)
import constants


from time import process_time
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f})".format(int(hours),int(minutes),seconds))

path_to_io = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"io")
sys.path.insert(1, path_to_io)
import OCVRPParser
from print_functions import Painter

path_to_branchAndBound = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact", "bab")
sys.path.insert(2, path_to_branchAndBound)
from BAB_exact import BABExact

path_to_enumeration = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact","enumeration")
sys.path.insert(3, path_to_enumeration)
from enumeration_scenarios import ScenarioEnumerationSolver

path_to_exact= os.path.join((Path(os.path.abspath(__file__)).parents[1]), "algorithms", "exact")
sys.path.insert(4, path_to_exact)
from ring_star_without_TW import ring_star_deterministic_no_TW
#from sample_average import sampleAverageApproximation


prefix="tag: "

def calculate_exp_disc_cost(policy,   file_instance):
    folder = os.path.join((Path(os.path.abspath(__file__)).parents[4]), "data","i_Solomon_disc")
    file_instance = os.path.join(folder, file_instance+".txt")

    OCVRPInstance = OCVRPParser.parse(file_instance)
    OCVRPInstance.calculateInsertionBounds()

    discount_cost = 0
    for i in OCVRPInstance.customers:
        print(i.id)
        if policy & (1 << (i.id - 1)):
            print("yes")
            disount_cost += (1-OCVRPInstance.p_home[i.id])* OCVRPInstance.shipping_fee[i.id]
    return discount_cost

calculate_exp_disc_cost(0,"solomonC100_size_14_phome_0.2_ppup_0.2_incrate_0.1_1")



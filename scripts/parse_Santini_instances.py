from pathlib import Path
import sys
import os
import csv
import math
from collections import defaultdict

def adapt_file(file_instance, outputDir):

    with open(file_instance, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        name = lines[0].rstrip().split(" : ")[1]
        nr_cust = int(lines[2].split(" : ")[1])-1
        # flat_rate_shipping_fee = float(general_info[1])
        # deviationProb = float(general_info[2])/100
        vertices = {}

        for line in lines[6:6 + nr_cust]:
            custInfo = line.split()
            cust_id = int(custInfo[0])-1
            vertices[cust_id] = {}
            vertices[cust_id]['x'] = float(custInfo[1])
            vertices[cust_id]['y'] = float(custInfo[2])
            vertices[cust_id]['prob_accept'] = float(  lines[7+nr_cust+cust_id])
            vertices[cust_id]['shipping_fee'] =  float(  lines[9+2*nr_cust+cust_id])

        vertices[0] = {}
        vertices[0]['x'] = float(lines[5].split()[1])
        vertices[0]['y'] = float(lines[5].split()[2])

        vertices[nr_cust + 1] = {}
        vertices[nr_cust + 1]['x'] = float(lines[5].split()[1])
        vertices[nr_cust + 1]['y'] = float(lines[5].split()[2])


    instanceName = os.path.join(outputDir, name+'.txt')
    with open(instanceName, 'w+', encoding='utf-8') as file:
        file.write("NAME: {}\n".format(name))
        file.write("SIZE: {}\n\n".format(nr_cust))

        file.write("LOCATION COORDINATES:\n\n")

        file.write("DEPOT V.   XCOORD.   YCOORD.\n")
        file.write("{} {} {}\n\n".format(0, vertices[0]['x'], vertices[0]['y']))

        file.write("PUP V.   XCOORD.   YCOORD.\n")
        file.write("{} {} {}\n\n".format(nr_cust + 1, vertices[nr_cust + 1]['x'], vertices[nr_cust + 1]['y']))

        file.write("CUST V.   XCOORD.   YCOORD.   PROB_ALWAYS_HOME   PROB_ALWAYS_PUP   SHIPPING_FEE\n")
        for i in range(1, nr_cust + 1):
            file.write("{} {} {} {} {} {}\n".format(i, vertices[i]['x'], vertices[i]['y'], vertices[i]['prob_accept'],0.0,  vertices[i]['shipping_fee']))

    instanceList = os.path.join(outputDir, 'list.txt')

    with open(instanceList, 'a+', encoding='utf-8') as file:
        file.write("{}\n".format(name))



if __name__ == "__main__":
    inputDir = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "instances_Santini_original")
    outputDir = os.path.join((Path(os.path.abspath(__file__)).parents[1]), "data", "instances_Santini_adapted")

    files_tsp = [f for f in os.listdir(inputDir) if f.endswith(".txt")]
    #files = [f for f in os.listdir(inputDir) if ('sz-14' in f) or ('sz-15' in f) or ('sz-16' in f) or ('sz-17' in f) or  ('sz-18' in f) or ('sz-19' in f) or ('sz-20' in f) or ('sz-21' in f) ]
    for f in files_tsp:
        adapt_file(os.path.join(inputDir,f),outputDir)

from pathlib import Path
import os
# == == == == == = SAA PARAMS == == == == == == ==
#SAMPLE_SIZE = 5000
SAMPLE_SIZE = 1000
NUMBER_OF_SAMPLES = 10

# == == == == == = VRP MODEL PARAMS == == == == == == ==
# fleet size (=number of virtual PUPs)
FLEET_SIZE = 1
PUP_CAPACITY =60
VEHICLE_CAPACITY = 50
PROB_PLACE_ORDER = 1
#number of  segments of customers(from nonparametric rank approach)
#NUMBER_SEGMENTS = 3


# == == == == == == GENERAL == == == == == == == == ==
# EPS - precicion of calculations, does not depend on the algorithm requirements
EPS = 10e-6
BIGM = 1000
MIPGAP = 1*10e-2
LIMIT_CORES = 20
TIME_LIMIT = 3600
# == == == == == == HEURISTICS == == == == == == == == ==
HEURISTIC_TIME_LIMIT = 360000
HEURISTIC_SAMPLE_SIZE = 200
NEIGHBOURHOOD_HEURISTIC = 1
REMOTE_HEURISTIC_SAMPLE = 500
# == == == == == == Branch and Bound == == == == == == == == ==

#for epsilon-optimality of nodes(we prune the nodes that are epsilon-close to the current best node)
EPSILON =0
EPSILON_H = 0.01
EPSILONRS = 0

# file_images = os.path.join((Path(os.path.abspath(__file__)).parents[1]),"results","images")
#segment_probability[0] - proability of always home

PATH_TO_DATA = os.path.join((Path(os.path.abspath(__file__)).parents[6]),"OneDrive - TU Eindhoven", "~code_data", "OCTSP")
PATH_TO_IMAGES = os.path.join(PATH_TO_DATA, "images")
PREFIX = "tag: "
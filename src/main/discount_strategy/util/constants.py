from pathlib import Path
import os
# == == == == == = SAA PARAMS == == == == == == ==
SAMPLE_SIZE = 5000
NUMBER_OF_SAMPLES = 10

# == == == == == = VRP MODEL PARAMS == == == == == == ==
FLEET_SIZE = 1
PUP_CAPACITY = 60
VEHICLE_CAPACITY = 50
PROB_PLACE_ORDER = 1
SCALING_FACTOR = 1000
# == == == == == == GENERAL == == == == == == == == ==
EPS = 10e-6
BIGM = 1000
BIGM_POLICY_COST = 10**10
MIPGAP = 1*10e-2
LIMIT_CORES = 20
TIME_LIMIT = 80000
# == == == == == == HEURISTICS == == == == == == == == ==
HEURISTIC_TIME_LIMIT = 3600
HEURISTIC_SAMPLE_SIZE = 200
NEIGHBOURHOOD_HEURISTIC = 1
REMOTE_HEURISTIC_SAMPLE = 500
# == == == == == == Branch and Bound == == == == == == == == ==

#for epsilon-optimality of nodes(we prune the nodes that are epsilon-close to the current best node)
EPSILON =0
EPSILON_H = 0.01
EPSILONRS = 0

PATH_TO_DATA = os.path.join((Path(os.path.abspath(__file__)).parents[4]),"data")
PATH_TO_IMAGES = os.path.join(PATH_TO_DATA, "output")
PREFIX = "tag: "
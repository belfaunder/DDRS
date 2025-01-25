import math
def set_distance(x1,y1,x2,y2):
    # EUC_2D : rounded Euclidean distances metric from TSPLIN format
    return int(math.sqrt((x1 - x2) ** 2 +(y1 -y2) ** 2) + 0.5)

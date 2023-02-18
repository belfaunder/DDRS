import math
# input: integer
# output: number of 1s in the binary reporesentation of the integer
def set_distance(x1,y1,x2,y2):

    return int(math.sqrt((x1 - x2) ** 2 +
                  (y1 -y2) ** 2) + 0.5)

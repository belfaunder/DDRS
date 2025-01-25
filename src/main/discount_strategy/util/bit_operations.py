
# input: integer
# output: number of 1s in the binary reporesentation of the integer
def bitCount(int_type):
    return bin(int_type).count("1")
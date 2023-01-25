
# input: integer
# output: number of 1s in the binary reporesentation of the integer
def bitCount(int_type):
    count = 0
    while (int_type):
        int_type &= int_type - 1
        count += 1
    return (count)
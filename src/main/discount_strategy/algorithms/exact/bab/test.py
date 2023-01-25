
def probability_with_recursion( prev_seeds, code_intersection):
    weight_intersect = 0
    if prev_seeds:
        prev_seeds_current = []
        for s in prev_seeds:
            weight_intersect += probability_with_recursion(prev_seeds_current, code_intersection & s)
            prev_seeds_current.append(s)

    return setWeightCovered(code_intersection) - weight_intersect

#returns some float
def setWeightCovered(scenario):
    weightCovered = 1
    policy = 7
    # cumulative weight of all scenarios that more expensive than this lbScenario
    for offset in range(3):
        mask = 1 << offset
        if not (scenario & mask):
            if (policy & mask):
                weightCovered *= 0.3
            else:
                weightCovered *= 0.7
        # if (scenario & mask) < (policy & mask):
        #    probTemp += 1
    #weightCovered = p_dev ** (probTemp)
    return weightCovered

# seet_set is: (011), (110), (111)
if __name__ == "__main__":
    print(probability_with_recursion([3,6], 7))

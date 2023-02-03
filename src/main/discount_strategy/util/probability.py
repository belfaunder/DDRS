from src.main.discount_strategy.util import constants
# probability that the customer choose PUP being provided discount t =
# =probability of segment 0(customers who always choose PUP)
# +probability of segments with lowest duiscount < t

#only for 2 segment model
#TODO:check function
def scenarioPossible(scenario, policy, num_defined,n, p_pup):
    #return True
    if sum(p_pup)<n*constants.EPS:
        #TODO: array that have 1 if p_pup_i = 0, and then in all conditions: &array
        if (~policy & scenario):
            return False
        else:
            for i in range(num_defined + 1, n + 1):
                if scenario & (1 << (i - 1)):
                    return False
            return True
    else:
        for i in range(num_defined + 1, n + 1):
            if scenario & (1 << (i - 1)):
                return False
        return True

def scenarioPossible_2segm(scenario, policy, num_defined,n):
    if (~policy & scenario):
        return False
    else:
        for i in range(num_defined + 1, n + 1):
            if scenario & (1 << (i - 1)):
                return False
        return True


def scenarioProb_2segm(scenario, policy, num_defined, n, p_pup_delta):
    #scenarioProb = reduce(lambda x, y: x*y, p_home[num_defined + 1:], 1)
    scenarioProb = 1
    for i in range(1, num_defined + 1):
        if scenario & (1 << (i - 1)):
            scenarioProb *= p_pup_delta[i]
        elif policy & (1 << (i - 1)):
            scenarioProb *= (1-p_pup_delta[i])

    for i in range(num_defined + 1, n + 1):
        scenarioProb *= 1-p_pup_delta[i]
    return scenarioProb
#
# def scenarioProb(scenario, policy, num_defined, n, p_home, p_pup):
#     if not scenarioPossible(scenario, policy, num_defined, n, p_pup):
#         return 0
#     scenarioProb = 1
#     for i in range(1, num_defined + 1):
#         if scenario & (1 << (i - 1)):
#             if policy & (1 << (i - 1)):
#                 scenarioProb *= (1 - p_home[i])
#             else:
#                 scenarioProb *= p_pup[i]
#         else:
#             if policy & (1 << (i - 1)):
#                 scenarioProb *= p_home[i]
#             else:
#                 scenarioProb *= (1-p_pup[i])
#
#     for i in range(num_defined + 1, n + 1):
#         if scenario & (1 << (i - 1)):
#             scenarioProb *= p_pup[i]
#         else:
#             scenarioProb *= p_home[i]
#     return scenarioProb


def scenarioProb_PROB_PLACE_ORDER(scenario, policy, num_defined, n, p_home, p_pup):
    if not scenarioPossible(scenario, policy,num_defined,n, p_pup):
        return 0
    scenarioProb = 1


    for i in range(1, num_defined + 1):
        if scenario & (1 << (i - 1)):
            if policy & (1 << (i - 1)):
                scenarioProb *= (1 - p_home[i])*constants.PROB_PLACE_ORDER + (1-constants.PROB_PLACE_ORDER)
            else:
                scenarioProb *= p_pup[i]*constants.PROB_PLACE_ORDER+ (1-constants.PROB_PLACE_ORDER)
        else:
            if policy & (1 << (i - 1)):
                scenarioProb *= p_home[i]*constants.PROB_PLACE_ORDER
            else:
                scenarioProb *= (1-p_pup[i])*constants.PROB_PLACE_ORDER

    for i in range(num_defined + 1, n + 1):
        if scenario & (1 << (i - 1)):
            scenarioProb *= p_pup[i]*constants.PROB_PLACE_ORDER + (1-constants.PROB_PLACE_ORDER)
        else:
            scenarioProb *= p_home[i]*constants.PROB_PLACE_ORDER
    return scenarioProb

# defined_policy = 1 << diff_customer - 1
# not_defined_policy = 1 << self.NR_CUST - 1 - defined_policy
# ones = np.ones * (self.NR_CUST + 1)
# anti_p_dev = np.subtract(ones, p_dev)
# anti_scenario = np.subtract(ones, np.unpackbits(scenario))
# scenarioProb = np.multiply(p_dev, np.unpackbits(not_defined_policy))+\
#               np.multiply(p_dev, np.unpackbits(defined_policy),withDiscountIDRight, anti_scenario)+\
#               np.multiply(anti_p_dev, np.unpackbits(defined_policy),withDiscountIDRight,  scenario)
def set_customer_probability(given_discount, segment_probability):

    probability = segment_probability[0]
    if given_discount >= 1:
        probability += segment_probability[1]
    return probability


# calculate probability of scenario given the offered disocunt policy
# INPUT: scenario, policy, p_dev - probability that customer's choice deviate from the discount policy
# OUTPUT: probability
# for convenience in other functions scneario has n+1 elements, while policy has n elements
def setScenarioProbForPolicy(scenario, policy, p_dev):
    probability = 1
    for customer, customerChoice in enumerate(scenario):
        if customerChoice > policy[customer]:
            return 0

        elif policy[customer] == 1 :
            if customerChoice == policy[customer]:
                probability *= (1 - p_dev)
            else:
                probability *= p_dev
    return probability


# for convenience in other functions scneario has n+1 elements, while policy has n elements
# returns upper and lower bound of probability for the scenrio given policy
# bounds due to not fully defined policy
def setScenarioProbForPolicy2(scenarioID, noDiscountID, withDiscountID, p_dev, n):
    probabilityLb = 1
    noDiscount = bin(noDiscountID)[2:].zfill(n)
    withDiscount = bin(withDiscountID)[2:].zfill(n)
    scenario = bin(scenarioID)[2:].zfill(n)
    for customerIndex, customerChoice in enumerate(scenario):
        #print(customerIndex, customerChoice, withDiscount[customerIndex], noDiscount[customerIndex])
        # if policy for customer not defined yet

        if customerChoice == '1':
            if withDiscount[customerIndex]=='0':
                return 0
            else:
                probabilityLb *= (1 - p_dev)

        else:
            if noDiscount[customerIndex]=='0':
                    probabilityLb *= p_dev
    return probabilityLb



def set_scenario_probability(discount_strategy, scenario_choice, input_data, set_N):
    # 4 types of customers:
    disc_pup = 0
    disc_home = 0
    nodisc_pup = 0
    nodisc_home = 0

    probability = 1
    if sum(scenario_choice[i] for i in set_N) <= constants.PUP_CAPACITY:
        for i in set_N:
            if discount_strategy[i] > constants.EPS:
                if scenario_choice[i] == 1:
                    disc_pup += 1
                else:
                    disc_home += 1
            else:
                if scenario_choice[i] == 1:
                    nodisc_pup += 1
                else:
                    nodisc_home += 1

        segment_probability = input_data['segment_probability']

        probability *= (set_customer_probability(1, segment_probability)) ** disc_pup
        probability *= (1 - set_customer_probability(1, segment_probability)) ** disc_home
        probability *= (set_customer_probability(0, segment_probability)) ** nodisc_pup
        probability *= (1 - set_customer_probability(0, segment_probability)) ** nodisc_home

    else:
        probability = 0
    return probability





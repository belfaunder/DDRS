from src.main.discount_strategy.util import constants

#only for 2 segment model
def scenarioPossible(scenario, policy, num_defined,n, p_pup):
    #return True
    if sum(p_pup)<n*constants.EPS:
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


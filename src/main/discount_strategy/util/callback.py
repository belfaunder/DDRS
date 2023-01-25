import gurobipy as grb
import constants

def callback_neg_obj(model,where):
    '''
        MIP_OBJBST MIP double   Current best objective.
        MIP_OBJBND MIP double   Current best objective bound.
    '''
    if where == grb.GRB.Callback.MIP:
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        best_obj = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        best_bnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        num_slns = model.cbGet(grb.GRB.Callback.MIP_SOLCNT)
        # /bnd or /obj?
        gap = abs((best_bnd - best_obj) / (best_obj + 1e-10))
        if gap < constants.GAP:
            #if best_obj < 0 - EPS :
            model.terminate()
        # elif time > 400 and num_slns>0:
        #     model.terminate()
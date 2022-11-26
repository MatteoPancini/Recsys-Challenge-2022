import scipy.sparse as sps

def combine(URM, ICM):
    URMICMCombined = sps.vstack([URM, ICM.T])
    return URMICMCombined

def combineKFold(URM_train_list, ICM, alpha=10):

    URMs_stack_list = []
    ICM_new = ICM * alpha

    for k in range(len(URM_train_list)):
        URMs_stack_list.append(sps.vstack([URM_train_list[k], ICM_new.T]))

    return URMs_stack_list
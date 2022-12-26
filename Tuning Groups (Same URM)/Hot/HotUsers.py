if __name__ == "__main__":

    import numpy as np
    import scipy.sparse as sp
    import pandas as pd
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Utils.recsys2022DataReader import *
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderNoVariables
    import matplotlib.pyplot as plt
    from Evaluation.Evaluator import EvaluatorHoldout
    from datetime import datetime
    import json



    # ---------------------------------------------------------------------------------------------------------
    # Loading URM + ICM

    URM_train_init = load_URMTrainInit()
    URM_test = load_URMTest()
    #ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Fitting of recommenders

    recommender_object_dict = {}

    # --------------------------
    # OLD best


    # RP3 beta MIGLIORATO
    '''
    oldRP3 = RP3betaRecommender(URM_train_init)
    oldRP3.fit(alpha=0.7136052911660057, beta=0.44828831909194655, topK=54)
    recommender_object_dict['oldRP3'] = oldRP3
    '''

    # SLIM MIGLIORATO
    '''
    oldSlim = SLIMElasticNetRecommender(URM_train_init)
    oldSlim.fit(topK=429, alpha=0.0047217460142242595, l1_ratio=0.501517968826842)
    recommender_object_dict['oldSlim'] = oldSlim
    '''

    # --------------------------
    # New Tuning

    # BEST SLIM -> from MULTI
    '''
    bestSlim = SLIMElasticNetRecommender(URM_train_init)
    bestSlim.fit(topK=298, alpha=0.041853285688557666, l1_ratio=0.01653973129200162)
    recommender_object_dict['bestSlim'] = bestSlim
    '''
    # BEST RP3 beta
    '''
    bestRP3 = RP3betaRecommender(URM_train_init)
    bestRP3.fit(alpha=0.6078606485515248, beta=0.32571505237450094, topK=52)
    recommender_object_dict['bestRP3'] = bestRP3
    '''



    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 2

    cutoff = 10

    profile_length = np.ediff1d(URM_train_init.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM_train_init[i, :].nonzero()[0]))

    list_group_interactions = [[0, 20], [21, 49], [50, max(interactions)]]

    lower_bound = list_group_interactions[group_id][0]
    higher_bound = list_group_interactions[group_id][1]

    users_in_group = [user_id for user_id in range(len(interactions))
                      if (lower_bound <= interactions[user_id] <= higher_bound)]
    users_in_group_p_len = profile_length[users_in_group]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    for label, recommender in recommender_object_dict.items():
        result_df, _ = evaluator_test.evaluateRecommender(recommender)
        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
        else:
            MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]

    # ---------------------------------------------------------------------------------------------------------
    # Plot and save

    finalResults = {}
    _ = plt.figure(figsize=(16, 9))
    for label, recommender in recommender_object_dict.items():
        results = MAP_recommender_per_group[label]
        finalResults[label] = results
        plt.scatter(x=label, y=results, label=label)
    plt.title('Hot Group')
    plt.ylabel('MAP')
    plt.legend()
    plt.show()

    with open("logs/ColdUsers_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


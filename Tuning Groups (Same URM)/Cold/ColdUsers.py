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
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Fitting of recommenders

    recommender_object_dict = {}

    # --------------------------
    # OLD best

    oldRP3 = RP3betaRecommender(URM_train_init)
    oldRP3.fit(alpha=0.6627101454340679, beta=0.2350020032542621, topK=250)
    recommender_object_dict['oldRP3'] = oldRP3

    oldItemKNN = ItemKNNCFRecommender(URM_train_init)
    oldItemKNN.fit(ICM=ICM, topK=5893, shrink=50, similarity='rp3beta', normalization='tfidf')
    recommender_object_dict['oldItemKNN'] = oldItemKNN

    oldSLIM = SLIMElasticNetRecommender(URM_train_init)
    oldSLIM.fit(alpha=0.22747568631546267, l1_ratio=0.007954654152433904, topK=214)
    recommender_object_dict['oldSLIM'] = oldSLIM


    # --------------------------
    # New Tuning

    newRP3 = RP3betaRecommender(URM_train_init)
    newRP3.fit(alpha=0.7012109713464896, beta=0.2643507726000572, topK=274)
    recommender_object_dict['newRP3'] = newRP3

    slim = SLIMElasticNetRecommender(URM_train_init)
    slim.fit(topK=214, alpha=0.22747568631546267, l1_ratio=0.007954654152433904)
    recommender_object_dict['slim'] = slim

    bestItemKNNCFG0 = ItemKNNCFRecommender(URM_train_init)
    bestItemKNNCFG0.fit(ICM, shrink=176, topK=1353, similarity='rp3beta',
                        normalization='tfidf')
    recommender_object_dict['bestItemKNNCFG0'] = bestItemKNNCFG0

    newslim = SLIMElasticNetRecommender(URM_train_init)
    newslim.fit(topK=84, alpha=0.08684266980720609, l1_ratio=0.054436419366615744)
    recommender_object_dict['newslim'] = newslim

    slimmulti = SLIMElasticNetRecommender(URM_train_init)
    slimmulti.fit(topK=299, alpha=0.057940560184114316, l1_ratio=0.06563962491123715)
    recommender_object_dict['slimmulti'] = slimmulti


    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 0

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
    plt.title('Cold Group Binary')
    plt.ylabel('MAP')
    plt.legend()
    plt.show()

    with open("logs/ColdUsers_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


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
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    import matplotlib.pyplot as plt
    from Evaluation.Evaluator import EvaluatorHoldout
    from datetime import datetime
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM + ICM

    URM = createURM()
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting of recommenders

    recommender_object_dict = {}
    """
    # SLIM Elastic Net
    SlimElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNet.fit(topK=359, alpha=0.04183472018614359, l1_ratio=0.03260349571135893)
    recommender_object_dict['SLIM Elastic Net'] = SlimElasticNet"""

    # ItemKNNCF
    ItemKNNCFHot = ItemKNNCFRecommender(URM_train)
    ItemKNNCFHot.fit(ICM, shrink=62.12102837804762, topK=479, similarity='cosine', normalization='tfidf')
    recommender_object_dict['CombinedItemKNNCFHot'] = ItemKNNCFHot

    # RP3beta
    RP3betaHot = RP3betaRecommender(URM_train)
    RP3betaHot.fit(alpha=0.2617149087459967, beta=0.38766862684444214, topK=109)
    recommender_object_dict['RP3betaHot'] = RP3betaHot

    # RP3beta_ItemKNN Hybrid
    recommender1 = ItemKNNCFRecommender(URM_train)
    recommender1.fit(ICM, shrink=57.6924228938274, topK=408, similarity='dice', normalization='bm25')

    recommender2 = RP3betaRecommender(URM_train)
    recommender2.fit(alpha=0.5674554399991163, beta=0.38051048617892586, topK=100)

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender1, recommender2)
    hybrid.fit(alpha=0.00793735238105765, beta=0.24158612307881616)
    recommender_object_dict['RP3beta_ItemKNNCF_Hybrid'] = hybrid

    # RP3beta
    RP3betaHot = RP3betaRecommender(URM_train)
    RP3betaHot.fit(alpha=0.7136052911660057, beta=0.44828831909194655, topK=54)
    recommender_object_dict['RP3betaHot'] = RP3betaHot

    # RP3beta
    IALSHot = ImplicitALSRecommender(URM_train)
    IALSHot.fit(factors=145, alpha=2, iterations=84, regularization=0.0068126415129997255)
    recommender_object_dict['IALSHot'] = IALSHot

    # SLIM Elastic Net
    SlimElasticNetHot = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNetHot.fit(topK=429, alpha=0.0047217460142242595, l1_ratio=0.501517968826842)
    recommender_object_dict['SlimElasticNetHot'] = SlimElasticNetHot



    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 2
    cutoff = 10

    profile_length = np.ediff1d(URM.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM[i, :].nonzero()[0]))

    list_group_interactions = [[0, 20], [21, 49], [50, max(interactions)]]

    lower_bound = list_group_interactions[group_id][0]
    higher_bound = list_group_interactions[group_id][1]

    users_in_group = [user_id for user_id in range(len(interactions)) if (lower_bound <= interactions[user_id] <= higher_bound)]
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

    with open("logs/HotUsers_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


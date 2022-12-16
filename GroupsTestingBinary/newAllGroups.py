if __name__ == "__main__":

    import numpy as np
    import scipy.sparse as sp
    import pandas as pd
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Utils.recsys2022DataReader import *
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.NonPersonalizedRecommender import TopPop
    from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    import matplotlib.pyplot as plt
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime



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


    # ItemKNNCF
    ItemKNNCF = ItemKNNCFRecommender(URM_train)
    ItemKNNCF.fit(ICM, shrink=1665.2431108249625, topK=3228, similarity='dice',
                  normalization='bm25')
    recommender_object_dict['ItemKNNCF'] = ItemKNNCF

    # RP3beta
    RP3beta = RP3betaRecommender(URM_train)
    RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)
    recommender_object_dict['RP3Beta'] = RP3beta

    # P3alpha
    P3alpha = P3alphaRecommender(URM_train)
    P3alpha.fit(topK=218, alpha=0.8561168568686058)
    recommender_object_dict['P3Alpha'] = P3alpha

    # SLIM Elastic Net
    SlimElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNet.fit(topK=359, alpha=0.04183472018614359, l1_ratio=0.03260349571135893)
    recommender_object_dict['SLIM Elastic Net'] = SlimElasticNet

    # P3alpha + RP3beta
    recommender_P3alpha = P3alphaRecommender(URM_train)
    recommender_P3alpha.fit(topK=218, alpha=0.8561168568686058)

    recommender_RP3beta = RP3betaRecommender(URM_train)
    recommender_RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender_P3alpha,
                                                                recommender_RP3beta)
    recommender_hybrid.fit(alpha=0.26672657848316894, beta=1.8325046917533472)
    recommender_object_dict['P3alpha+RP3beta'] = recommender_hybrid

    # ------------------------
    # Group 0

    # ItemKNNCF
    ItemKNNCFG0 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG0.fit(ICM, shrink=505.8939180154946, topK=3556, similarity='rp3beta',
                    normalization='bm25plus')
    recommender_object_dict['ItemKNNCFG0'] = ItemKNNCFG0

    # RP3beta
    RP3betaG0 = RP3betaRecommender(URM_train)
    RP3betaG0.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)
    recommender_object_dict['RP3betaG0'] = RP3betaG0



    # ------------------------
    # Group 1

    # SLIM Elastic Net G1
    SlimElasticNetG1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNetG1.fit(topK=216, alpha=0.09376418450176816, l1_ratio=0.03954091993785014)
    recommender_object_dict['SlimElasticNetG1'] = SlimElasticNetG1

    # RP3beta G1
    RP3betaG1 = RP3betaRecommender(URM_train)
    RP3betaG1.fit(alpha=0.4770536011269113, beta=0.36946801560978637, topK=190)
    recommender_object_dict['RP3betaG1'] = RP3betaG1

    # ------------------------
    # Group 2

    # ------------------------
    # Group 3


    # ---------------------------------------------------------------------------------------------------------
    # Profiling & Evaluation

    profile_length = np.ediff1d(URM_train.indptr)

    sorted_users = np.argsort(profile_length)

    MAP_recommender_per_group = {}

    cutoff = 10

    interactions = []
    for i in range(41629):
        interactions.append(len(URM[i, :].nonzero()[0]))

    list_group_interactions = [[0, 219], [220, 289], [290, max(interactions)]]
    MAP_recommender_per_group_int = {}

    for group_id in range(0, 3):
        lower_bound = list_group_interactions[group_id][0]
        higher_bound = list_group_interactions[group_id][1]

        users_in_group = [user_id for user_id in range(len(interactions))
                          if (lower_bound <= interactions[user_id] <= higher_bound)]
        users_in_group_p_len = profile_length[users_in_group]

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
            group_id,
            len(users_in_group),
            users_in_group_p_len.mean(),
            np.median(users_in_group_p_len),
            users_in_group_p_len.min(),
            users_in_group_p_len.max()))

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
        plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
    plt.ylabel('MAP')
    plt.xlabel('User Group')
    plt.title('New Groups Interactions')
    plt.legend()
    plt.show()

    with open("logs/AllGroups_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)



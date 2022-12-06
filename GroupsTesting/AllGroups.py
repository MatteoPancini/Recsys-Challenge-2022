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
    """
    # ItemKNNCF
    ItemKNNCF = ItemKNNCFRecommender(URM_train)
    ItemKNNCF.fit(ICM, shrink=1665.2431108249625, topK=3228, similarity='dice',
                  normalization='bm25')
    recommender_object_dict['ItemKNNCF'] = ItemKNNCF

    # RP3beta
    RP3beta = RP3betaRecommender(URM_train)
    RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)
    recommender_object_dict['RP3Beta'] = RP3beta
    
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
    recommender_object_dict['P3alpha+RP3beta'] = recommender_hybrid"""

    # ------------------------
    # Group 0

    # ItemKNNCF
    ItemKNNCFG0 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG0.fit(ICM, shrink=505.8939180154946, topK=3556, similarity='rp3beta',
                    normalization='bm25plus')
    recommender_object_dict['CombinedItemKNNCFG0'] = ItemKNNCFG0

    # RP3beta
    RP3betaG0 = RP3betaRecommender(URM_train)
    RP3betaG0.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)
    recommender_object_dict['RP3betaG0'] = RP3betaG0
    """
    # SLIM BPR
    SlimBPRG0 = SLIM_BPR_Python(URM_train)
    SlimBPRG0.fit(topK=4439, epochs=85, lambda_j=0.002175177631903779, lambda_i=0.004642005196062006)
    recommender_object_dict['SlimBPRG0'] = SlimBPRG0"""

    # ------------------------
    # Group 1
    """
    # SLIM Elastic Net G1
    SlimElasticNetG1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNetG1.fit(topK=216, alpha=0.09376418450176816, l1_ratio=0.03954091993785014)
    recommender_object_dict['SlimElasticNetG1'] = SlimElasticNetG1"""

    # RP3beta G1
    RP3betaG1 = RP3betaRecommender(URM_train)
    RP3betaG1.fit(alpha=0.4770536011269113, beta=0.36946801560978637, topK=190)
    recommender_object_dict['RP3betaG1'] = RP3betaG1

    # ------------------------
    # Group 2

    # ItemKNNCF
    ItemKNNCFG2 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG2.fit(ICM, shrink=10.544403292046802, topK=309, similarity='rp3beta', normalization='tfidf')
    recommender_object_dict['CombinedItemKNNCFG2'] = ItemKNNCFG2

    # RP3beta
    RP3betaG2 = RP3betaRecommender(URM_train)
    RP3betaG2.fit(alpha=0.6687877652632948, beta=0.3841332145259308, topK=103)
    recommender_object_dict['RP3betaG2'] = RP3betaG2

    # RP3beta_ItemKNN Hybrid
    recommender1 = ItemKNNCFRecommender(URM_train)
    recommender1.fit(ICM, shrink=10.544403292046802, topK=309, similarity='rp3beta', normalization='tfidf')

    recommender2 = RP3betaRecommender(URM_train)
    recommender2.fit(alpha=0.6687877652632948, beta=0.3841332145259308, topK=103)

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender1, recommender2)
    hybrid.fit(alpha=0.13108190815550153, beta=0.4807361601575698)
    recommender_object_dict['RP3beta_ItemKNNCF_Hybrid'] = hybrid

    # ------------------------
    # Group 3

    # ItemKNNCF
    ItemKNNCFG3 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG3.fit(ICM, shrink=57.6924228938274, topK=408, similarity='dice', normalization='bm25')
    recommender_object_dict['CombinedItemKNNCFG3'] = ItemKNNCFG3

    # RP3beta
    RP3betaG3 = RP3betaRecommender(URM_train)
    RP3betaG3.fit(alpha=0.5674554399991163, beta=0.38051048617892586, topK=100)
    recommender_object_dict['RP3betaG3'] = RP3betaG3

    # RP3beta_ItemKNN Hybrid
    recommender1 = ItemKNNCFRecommender(URM_train)
    recommender1.fit(ICM, shrink=57.6924228938274, topK=408, similarity='dice', normalization='bm25')

    recommender2 = RP3betaRecommender(URM_train)
    recommender2.fit(alpha=0.5674554399991163, beta=0.38051048617892586, topK=100)

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender1, recommender2)
    hybrid.fit(alpha=0.00793735238105765, beta=0.24158612307881616)
    recommender_object_dict['RP3beta_ItemKNNCF_Hybrid'] = hybrid


    # ---------------------------------------------------------------------------------------------------------
    # Profiling

    profile_length = np.ediff1d(URM_train.indptr)

    block_size = int(len(profile_length) * 0.25)

    sorted_users = np.argsort(profile_length)

    for group_id in range(0, 4):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
            group_id,
            users_in_group.shape[0],
            users_in_group_p_len.mean(),
            np.median(users_in_group_p_len),
            users_in_group_p_len.min(),
            users_in_group_p_len.max()))

    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    cutoff = 10

    for group_id in range(0, 4):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

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
        plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
    plt.ylabel('MAP')
    plt.xlabel('User Group')
    plt.legend()
    plt.show()

    with open("logs/AllGroups_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)



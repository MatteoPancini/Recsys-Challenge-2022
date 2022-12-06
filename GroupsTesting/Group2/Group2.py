if __name__ == "__main__":

    import scipy.sparse as sp
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Utils.recsys2022DataReader import *
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
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
    SlimElasticNet.fit(topK=100, alpha=0.010975001075569505, l1_ratio=0.48011657798419954)
    recommender_object_dict['SLIM Elastic Net'] = SlimElasticNet"""

    # ItemKNNCF
    ItemKNNCFG2 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG2.fit(ICM, shrink=199.9030743355846, topK=286, similarity='cosine', normalization='bm25')
    recommender_object_dict['CombinedItemKNNCFG2'] = ItemKNNCFG2

    # RP3beta
    RP3betaG2 = RP3betaRecommender(URM_train)
    RP3betaG2.fit(alpha=0.6687877652632948, beta=0.3841332145259308, topK=103)
    recommender_object_dict['RP3betaG2'] = RP3betaG2

    #RP3beta_ItemKNN Hybrid
    recommender1 = ItemKNNCFRecommender(URM_train)
    recommender1.fit(ICM, shrink=10.544403292046802, topK=309, similarity='rp3beta', normalization='tfidf')

    recommender2 = RP3betaRecommender(URM_train)
    recommender2.fit(alpha=0.6687877652632948, beta=0.3841332145259308, topK=103)

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender1, recommender2)
    hybrid.fit(alpha=0.13108190815550153, beta=0.4807361601575698)
    recommender_object_dict['RP3beta_ItemKNNCF_Hybrid'] = hybrid

    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 2

    cutoff = 10

    profile_length = np.ediff1d(URM.indptr)

    block_size = int(len(profile_length) * 0.25)

    sorted_users = np.argsort(profile_length)

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
    # Plot


    finalResults = {}
    _ = plt.figure(figsize=(16, 9))
    for label, recommender in recommender_object_dict.items():
        results = MAP_recommender_per_group[label]
        finalResults[label] = results
        plt.scatter(x=label, y=results, label=label)
    plt.title('User Group 2')
    plt.ylabel('MAP')
    plt.xlabel('Recommenders')
    plt.legend()
    plt.show()


    with open("logs/Group2_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


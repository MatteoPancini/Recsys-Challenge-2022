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
    SlimElasticNet.fit(topK=359, alpha=0.04183472018614359, l1_ratio=0.03260349571135893)
    recommender_object_dict['SLIM Elastic Net'] = SlimElasticNet"""

    # ItemKNNCF
    ItemKNNCFG3 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG3.fit(ICM, shrink=57.6924228938274, topK=408, similarity='dice', normalization='bm25')
    recommender_object_dict['CombinedItemKNNCFG0'] = ItemKNNCFG3

    # RP3beta
    RP3betaG3 = RP3betaRecommender(URM_train)
    RP3betaG3.fit(alpha=0.6446628838942595, beta=0.21126459383188553, topK=363)
    recommender_object_dict['RP3betaG0'] = RP3betaG3

    # RP3beta_ItemKNN Hybrid
    recommender1 = ItemKNNCFRecommender(URM_train)
    recommender1.fit(ICM, shrink=57.6924228938274, topK=408, similarity='dice', normalization='bm25')

    recommender2 = RP3betaRecommender(URM_train)
    recommender2.fit(alpha=0.5674554399991163, beta=0.38051048617892586, topK=100)

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender1, recommender2)
    hybrid.fit(alpha=0.00793735238105765, beta=0.24158612307881616)
    recommender_object_dict['RP3beta_ItemKNNCF_Hybrid'] = hybrid

    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    cutoff = 10

    profile_length = np.ediff1d(URM.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM[i, :].nonzero()[0]))

    lower_bound = 0
    higher_bound = 22

    users_in_group = [user_id for user_id in range(len(interactions)) if
                      (lower_bound <= interactions[user_id] <= higher_bound)]
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
    plt.title('User Group 0')
    plt.ylabel('MAP')
    plt.xlabel('Recommenders')
    plt.legend()
    plt.show()


    with open("logs/Group0_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


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

    # ItemKNNCF
    ItemKNNCFG0 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG0.fit(ICM, shrink=108.99759968449757, topK=5251, similarity='rp3beta',
                       normalization='tfidf')
    recommender_object_dict['ItemKNNCFG0'] = ItemKNNCFG0


    # RP3beta
    RP3betaG0 = RP3betaRecommender(URM_train)
    RP3betaG0.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)
    recommender_object_dict['RP3betaG0'] = RP3betaG0

    # RP3beta
    newRP3betaG0 = RP3betaRecommender(URM_train)
    newRP3betaG0.fit(alpha=0.6649656555023034, beta=0.23957286370333863, topK=254)
    recommender_object_dict['newRP3betaG0'] = newRP3betaG0

    '''
    # RP3beta + ItemKNNCF
    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train)
    recommender_ItemKNNCF.fit(ICM, shrink=108.99759968449757, topK=5251, similarity='rp3beta',
                                          normalization='tfidf')


    recommender_RP3beta = RP3betaRecommender(URM_train)
    recommender_RP3beta.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)

    oldhybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train, Recommender_1=recommender_RP3beta,
                                                    Recommender_2=recommender_ItemKNNCF)
    oldhybrid.fit(alpha=0.36914252072676557, beta=0.37856318068441236)
    recommender_object_dict['old RP3beta + ItemKNNCF'] = oldhybrid

    # Slim Elastic Net
    SlimElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNet.fit(alpha=0.30855661490772857, l1_ratio=0.0034223693582097203, topK=405)
    recommender_object_dict['SlimElasticNet'] = SlimElasticNet
    '''

    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 0

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
    # Plot and save

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


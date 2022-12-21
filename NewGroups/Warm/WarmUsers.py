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
    '''
    # RP3beta G1
    RP3betaG1 = RP3betaRecommender(URM_train)
    RP3betaG1.fit(alpha=0.6190367265325001, beta=0.4018626515197256, topK=206)
    recommender_object_dict['RP3betaG1'] = RP3betaG1

    # RP3beta G1
    newRP3betaG1 = RP3betaRecommender(URM_train)
    newRP3betaG1.fit(alpha=0.612531391112378, beta=0.30067354757914466, topK=188)
    recommender_object_dict['newRP3betaG1'] = newRP3betaG1

    # P3alpha + RP3beta
    recommender_P3alpha = P3alphaRecommender(URM_train)
    recommender_P3alpha.fit(topK=218, alpha=0.8561168568686058)

    recommender_RP3beta = RP3betaRecommender(URM_train)
    recommender_RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender_P3alpha,
                                                                recommender_RP3beta)
    recommender_hybrid.fit(alpha=0.26672657848316894, beta=1.8325046917533472)
    recommender_object_dict['P3alpha+RP3beta'] = recommender_hybrid

    # ItemKNN + RP3beta
    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train)
    recommender_ItemKNNCF.fit(ICM, shrink=976.8108064049092, topK=5300, similarity='cosine',
                              normalization='bm25')

    recommender_RP3beta = RP3betaRecommender(URM_train)
    recommender_RP3beta.fit(alpha=0.6190367265325001, beta=0.4018626515197256, topK=206)

    recommender_hybrid2 = LinearHybridTwoRecommenderTwoVariables(URM_train, recommender_ItemKNNCF,
                                                                 recommender_RP3beta)
    recommender_hybrid2.fit(alpha=0.07806573588790788, beta=0.8465619360796353)
    recommender_object_dict['ItemKNNCF+RP3beta'] = recommender_hybrid2

    # ItemKNNCF Group 1
    ItemKNNCFG1 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG1.fit(ICM, shrink=1106, topK=4108, similarity='cosine',
                    normalization='bm25')
    recommender_object_dict['ItemKNNCFG1'] = ItemKNNCFG1

    # SLIM Elastic Net G1
    SlimElasticNetG1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNetG1.fit(topK=216, alpha=0.09376418450176816, l1_ratio=0.03954091993785014)
    recommender_object_dict['SlimElasticNetG1'] = SlimElasticNetG1

    # SLIM Elastic Net
    SlimElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    SlimElasticNet.fit(topK=100, alpha=0.010975001075569505, l1_ratio=0.48011657798419954)
    recommender_object_dict['SLIM Elastic Net'] = SlimElasticNet

    # ItemKNNCF
    ItemKNNCFG2 = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG2.fit(ICM, shrink=199.9030743355846, topK=286, similarity='cosine', normalization='bm25')
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
    '''

    # RP3beta
    highRP3beta = RP3betaRecommender(URM_train)
    highRP3beta.fit(alpha=0.6589242531275835, beta=0.33600386450849645, topK=81)
    recommender_object_dict['highRP3beta'] = highRP3beta

    # RP3beta
    lowRP3beta = RP3betaRecommender(URM_train)
    lowRP3beta.fit(alpha=0.572121247163269, beta=0.3107107930844788, topK=92)
    recommender_object_dict['Partial'] = lowRP3beta

    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 1

    cutoff = 10

    profile_length = np.ediff1d(URM.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM[i, :].nonzero()[0]))

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
    plt.title('Warm Group')
    plt.ylabel('MAP')
    plt.legend()
    plt.show()

    with open("logs/WarmUsers_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


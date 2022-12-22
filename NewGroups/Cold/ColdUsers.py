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

    URMBin = createURMBinary()
    URM = createURM()

    #ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)
    URMBin_train, URMBin_test = split_train_in_two_percentage_global_sample(URMBin, train_percentage=0.85)



    # ---------------------------------------------------------------------------------------------------------
    # Fitting of recommenders

    recommender_object_dict = {}

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


    # ItemKNNCF
    bestItemKNNCFG0 = ItemKNNCFRecommender(URM_train)
    bestItemKNNCFG0.fit(ICM, shrink=50, topK=5893, similarity='rp3beta',
                    normalization='tfidf')
    recommender_object_dict['bestItemKNNCFG0'] = bestItemKNNCFG0


    
    # RP3beta + ItemKNNCF
    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train)
    recommender_ItemKNNCF.fit(ICM, shrink=84, topK=3738, similarity='rp3beta',
                    normalization='tfidf')

    recommender_RP3beta = RP3betaRecommender(URM_train)
    recommender_RP3beta.fit(alpha=0.6419696179241512, beta=0.17548429620374373, topK=279)

    newhybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train, Recommender_1=recommender_RP3beta,
                                                       Recommender_2=recommender_ItemKNNCF)
    newhybrid.fit(alpha=0.36914252072676557, beta=0.37856318068441236)
    recommender_object_dict['new RP3beta + ItemKNNCF'] = newhybrid
    
    # P3alpha
    lastP3alpha = P3alphaRecommender(URM_train)
    lastP3alpha.fit(alpha=0.7094999549046719, topK=108)
    recommender_object_dict['lastP3alpha'] = lastP3alpha

    # RP3beta
    RP3betaG0 = RP3betaRecommender(URM_train)
    RP3betaG0.fit(alpha=0.6419696179241512, beta=0.17548429620374373, topK=279)
    recommender_object_dict['RP3betaG0'] = RP3betaG0


    newhybrid2 = LinearHybridTwoRecommenderNoVariables(URM_train=URM_train, Recommender_1=lastP3alpha,
                                                        Recommender_2=RP3betaG0)
    newhybrid2.fit()
    recommender_object_dict['new2 RP3beta + P3alpha'] = newhybrid2

    # P3alpha
    P3alpha5k = P3alphaRecommender(URM_train)
    P3alpha5k.fit(alpha=0.7548498182179307, topK=140)
    recommender_object_dict['P3alpha5k'] = P3alpha5k

    # ItemKNNCF
    ItemKNNCFG05k = ItemKNNCFRecommender(URM_train)
    ItemKNNCFG05k.fit(ICM, shrink=82, topK=4160, similarity='rp3beta',
                        normalization='tfidf')
    recommender_object_dict['ItemKNNCFG05k'] = ItemKNNCFG05k


    newhybrid4 = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train, Recommender_1=RP3betaG0,
                                                       Recommender_2=bestItemKNNCFG0)
    newhybrid4.fit(alpha=0.01, beta=0.99)
    recommender_object_dict['rp3beta + ItemKNN'] = newhybrid4
    '''


    slimMulti = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)
    slimMulti.fit(topK=214, alpha=0.22747568631546267, l1_ratio=0.007954654152433904)
    recommender_object_dict['slimMulti'] = slimMulti

    slim = SLIMElasticNetRecommender(URM_train)
    slim.fit(topK=214, alpha=0.22747568631546267, l1_ratio=0.007954654152433904)
    recommender_object_dict['slim'] = slim

    slimMultiBin = MultiThreadSLIM_SLIMElasticNetRecommender(URMBin_train)
    slimMultiBin.fit(topK=214, alpha=0.22747568631546267, l1_ratio=0.007954654152433904)
    recommender_object_dict['slimMultiBin'] = slimMultiBin

    slimBin = SLIMElasticNetRecommender(URMBin_train)
    slimBin.fit(topK=214, alpha=0.22747568631546267, l1_ratio=0.007954654152433904)
    recommender_object_dict['slimBin'] = slimBin


    # ---------------------------------------------------------------------------------------------------------
    # Evaluation of recommenders based on group

    MAP_recommender_per_group = {}

    group_id = 0

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
    plt.title('Cold Group Binary')
    plt.ylabel('MAP')
    plt.legend()
    plt.show()

    with open("logs/ColdUsers_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(finalResults, json_file, indent=4)


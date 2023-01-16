if __name__ == "__main__":
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    import csv
    from datetime import datetime
    import optuna as op
    import json
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_1K_BinURMTrain()
    URM_validation_list = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Profiling

    group_id = 0

    profile_length = np.ediff1d(URM_train_init.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM_train_init[i, :].nonzero()[0]))

    list_group_interactions = [[0, 15], [21, 49], [50, max(interactions)]]

    lower_bound = list_group_interactions[group_id][0]
    higher_bound = list_group_interactions[group_id][1]

    users_in_group = [user_id for user_id in range(len(interactions))
                      if (lower_bound <= interactions[user_id] <= higher_bound)]
    users_in_group_p_len = profile_length[users_in_group]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    users_not_in_group_list = []

    for k in range(1):
        profile_length = np.ediff1d(URM_train_list[k].indptr)
        sorted_users = np.argsort(profile_length)

        users_in_group = [user_id for user_id in range(len(interactions))
                          if (lower_bound <= interactions[user_id] <= higher_bound)]
        users_in_group_p_len = profile_length[users_in_group]

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group_list.append(sorted_users[users_not_in_group_flag])

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False,
                                                ignore_users_list=users_not_in_group_list)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_RP3beta_list = []
    recommender_ItemKNN_list = []
    recommender_Slim_list = []
    recommender_hybrid1_list = []
    MAP_results_list = []

    for index in range(len(URM_train_list)):

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index], verbose=False))
        recommender_RP3beta_list[index].fit(alpha=0.8933157693622866, beta=0.14484583688745223, topK=344)

        recommender_ItemKNN_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
        recommender_ItemKNN_list[index].fit(topK=321, shrink=164)

        recommender_hybrid1_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[index], Recommender_1=recommender_RP3beta_list[index], Recommender_2=recommender_ItemKNN_list[index]))
        recommender_hybrid1_list[index].fit(alpha=0.9196982707381445, beta=0.10860954592979288)

        recommender_Slim_list.append(SLIMElasticNetRecommender(URM_train_list[index], verbose=False))
        recommender_Slim_list[index].fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    def objective(trial):

        recommender_hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 1)

        for index in range(len(URM_train_list)):

            recommender_hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[index], Recommender_1=recommender_hybrid1_list[index], Recommender_2=recommender_Slim_list[index]))
            recommender_hybrid_list[index].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)
        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=250)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    recommender_RP3beta = RP3betaRecommender(URM_train_init, verbose=False)
    recommender_RP3beta.fit(alpha=0.8933157693622866, beta=0.14484583688745223, topK=344)

    recommender_ItemKNN = ItemKNNCFRecommender(URM_train_init, verbose=False)
    recommender_ItemKNN.fit(topK=321, shrink=164)

    rec1 = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=recommender_RP3beta, Recommender_2=recommender_ItemKNN)
    rec1.fit(alpha=0.9196982707381445, beta=0.10860954592979288)

    recommender_Slim = SLIMElasticNetRecommender(URM_train_init)
    recommender_Slim.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec1, Recommender_2=recommender_Slim)
    recommender_hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=users_not_in_group)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "RP3beta_ItemKNNCF15" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)


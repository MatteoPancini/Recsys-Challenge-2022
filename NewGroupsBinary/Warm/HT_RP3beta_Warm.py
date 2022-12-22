if __name__ == "__main__":

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op
    import numpy as np
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM_train_init = load_URMTrainInit()
    URM_train_list = load_K_URMTrain()
    URM_validation_list = load_K_URMValid()
    URM_test = load_URMTest()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'l1_ratio', 'TopK', 'MAP']

    partialsFile = 'SlimElasticNet_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Profiling + K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    group_id = 1

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

    users_not_in_group_list = []

    for k in range(3):
        profile_length = np.ediff1d(URM_train_list[k].indptr)
        sorted_users = np.argsort(profile_length)

        users_in_group = [user_id for user_id in range(len(interactions))
                          if (lower_bound <= interactions[user_id] <= higher_bound)]
        users_in_group_p_len = profile_length[users_in_group]

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group_list.append(sorted_users[users_not_in_group_flag])

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False,
                                                ignore_users_list=users_not_in_group_list)
    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_RP3beta_list = []

        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        beta = trial.suggest_float("beta", 0.1, 0.9)
        topK = trial.suggest_int("topK", 5, 1000)

        for index in range(len(URM_train_list)):
            recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index], verbose=False))
            recommender_RP3beta_list[index].fit(alpha=alpha, topK=topK, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_RP3beta_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_RP3beta_list[0].RECOMMENDER_NAME, alpha, beta, topK,
                          sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)

    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    recommender_RP3beta = RP3betaRecommender(URM_train_init, verbose=False)
    recommender_RP3beta.fit(alpha=alpha, beta=beta, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=users_not_in_group)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_RP3beta)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_RP3beta.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
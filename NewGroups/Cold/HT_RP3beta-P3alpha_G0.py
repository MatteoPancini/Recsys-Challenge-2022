if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    import optuna as op
    import json
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    import csv
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM & ICM

    URM = createURM()

    ICM = createSmallICM()

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)


    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'MAP']

    partialsFile = 'RP3Beta-P3alpha' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)


    # ---------------------------------------------------------------------------------------------------------
    # Profiling

    group_id = 0

    cutoff = 10

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

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator


    URM_train_list = []
    URM_validation_list = []
    users_not_in_group_list = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

        profile_length = np.ediff1d(URM_train.indptr)
        sorted_users = np.argsort(profile_length)

        users_in_group = [user_id for user_id in range(len(interactions))
                          if (lower_bound <= interactions[user_id] <= higher_bound)]
        users_in_group_p_len = profile_length[users_in_group]

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group_list.append(sorted_users[users_not_in_group_flag])

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False, ignore_users_list=users_not_in_group_list)

    MAP_results_list = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_P3alpha_list = []
    recommender_RP3beta_list = []

    for index in range(len(URM_train_list)):
        recommender_P3alpha_list.append(P3alphaRecommender(URM_train_list[index]))
        recommender_P3alpha_list[index].fit(alpha=0.7094999549046719, topK=108)

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index]))
        recommender_RP3beta_list[index].fit(alpha=0.6419696179241512, beta=0.17548429620374373, topK=279)

    def objective(trial):

        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = 1-alpha

        for index in range(len(URM_train_list)):

            recommender_Hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_list[index],
                                                                                  Recommender_1=
                                                                                  recommender_P3alpha_list[index],
                                                                                  Recommender_2=
                                                                                  recommender_RP3beta_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_Hybrid_list[0].RECOMMENDER_NAME, alpha, beta, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize', sampler=RandomSampler())
    study.optimize(objective, n_trials=200)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    recommender_P3alpha = P3alphaRecommender(URM_train_init)
    recommender_P3alpha.fit(alpha=0.7094999549046719, topK=108)

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_init,
                                                                Recommender_1=recommender_P3alpha,
                                                                Recommender_2=recommender_RP3beta)
    recommender_Hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=users_not_in_group)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_Hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/RP3Beta-P3alpha" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
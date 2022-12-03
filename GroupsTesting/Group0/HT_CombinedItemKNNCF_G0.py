if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    import optuna as op
    import json
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM & ICM

    URM = createURM()

    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'shrink', 'topk', 'similarity', 'normalization',  'MAP']

    partialsFile = 'CombinedItemKNNCF_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)


    # ---------------------------------------------------------------------------------------------------------
    # Profiling

    group_id = 0

    profile_length = np.ediff1d(URM.indptr)

    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM[i, :].nonzero()[0]))

    list_group_interactions = [[0, 19], [20, 39], [40, 69], [70, max(interactions)]]

    lower_bound = list_group_interactions[group_id][0]
    higher_bound = list_group_interactions[group_id][1]

    users_in_group = [user_id for user_id in range(len(interactions)) if
                      (lower_bound <= interactions[user_id] <= higher_bound)]
    users_in_group_p_len = profile_length[users_in_group]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False, ignore_users_list=users_not_in_group)

    MAP_results_list = []
    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_ItemKNNCF_list = []

        topK = trial.suggest_int("topK", 500, 6000)
        shrink = trial.suggest_float("shrink", 500, 2000)
        similarity = 'rp3beta'
        normalization = trial.suggest_categorical("normalization", ["bm25", "tfidf", "bm25plus"])

        for index in range(len(URM_train_list)):

            recommender_ItemKNNCF_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
            recommender_ItemKNNCF_list[index].fit(ICM=ICM, shrink=shrink, topK=topK, similarity=similarity,
                                                  normalization=normalization)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_ItemKNNCF_list[0].RECOMMENDER_NAME, shrink, topK, similarity, normalization, sum(MAP_result) / len(MAP_result)]

        with open('partials/Combined' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']
    similarity = 'rp3beta'
    normalization = study.best_params['normalization']

    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train_init, verbose=False)
    recommender_ItemKNNCF.fit(ICM=ICM, shrink=shrink, topK=topK, similarity=similarity, normalization=normalization)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], ignore_users=users_not_in_group)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_ItemKNNCF)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_ItemKNNCF.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
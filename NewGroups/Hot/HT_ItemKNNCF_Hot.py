if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from sklearn.model_selection import ParameterSampler
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

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

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

    profile_length = np.ediff1d(URM.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM[i, :].nonzero()[0]))

    lower_bound = 36
    higher_bound = max(interactions)

    users_in_group = [user_id for user_id in range(len(interactions)) if
                      (lower_bound <= interactions[user_id] <= higher_bound)]
    users_in_group_p_len = profile_length[users_in_group]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_list = []
    URM_validation_list = []
    users_not_in_group_list = []

    for k in range(5):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

        profile_length = np.ediff1d(URM.indptr)
        sorted_users = np.argsort(profile_length)

        lower_bound = 50
        higher_bound = max(interactions)

        users_in_group = [user_id for user_id in range(len(interactions)) if
                          (lower_bound <= interactions[user_id] <= higher_bound)]
        users_in_group_p_len = profile_length[users_in_group]

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group_list.append(sorted_users[users_not_in_group_flag])

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False, ignore_users_list=users_not_in_group_list)

    MAP_results_list = []
    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model
    """
    def objective(trial):

        recommender_ItemKNNCF_list = []

        topK = trial.suggest_int("topK", 100, 500)
        shrink = trial.suggest_float("shrink", 10, 200)
        similarity = trial.suggest_categorical("similarity", ["cosine", "rp3beta"])
        normalization = trial.suggest_categorical("normalization", ["bm25", "tfidf", "bm25plus"])

        for index in range(len(URM_train_list)):
            recommender_ItemKNNCF_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
            recommender_ItemKNNCF_list[index].fit(ICM=ICM, shrink=shrink, topK=topK, similarity=similarity,
                                                  normalization=normalization)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_ItemKNNCF_list[0].RECOMMENDER_NAME, shrink, topK, similarity, normalization, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)"""

    # ---------------------------------------------------------------------------------------------------------
    # Coarse-to-fine hyperparameter model

    grid_size = 100
    TUNE_ITER = 100
    num_epochs = 5
    worse_score = 0

    # Hyperparameter tuning interval
    init_param_grid = {'shrink': [i for i in np.arange(10, 100, 0.0001)],
                       'topK': [i for i in range(400, 1200)],
                       }

    new_param_grid = init_param_grid.copy()
    best_params_dict = {'score': worse_score, 'params': []}
    tried_params_list = []

    for epoch in range(num_epochs):

        # List of sampled hyperparameter combinations will be used for random search
        param_list = list(ParameterSampler(new_param_grid, n_iter=TUNE_ITER, random_state=0))

        # Searching the Best Parameters with Random Search
        rs_results_dict = {}
        for tune_iter in range(TUNE_ITER):
            # Get the set of parameter for this iteration
            strategy_params = param_list[tune_iter]

            # Create K recommenders
            recommender_ItemKNNCF_list = []

            for index in range(len(URM_train_list)):
                recommender_ItemKNNCF_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
                recommender_ItemKNNCF_list[index].fit(ICM=ICM, shrink=strategy_params['shrink'], topK=strategy_params['topK'], similarity='cosine',
                                                      normalization='tfidf')

            MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF_list)
            MAP_results_list.append(MAP_result)

            resultsToPrint = [recommender_ItemKNNCF_list[0].RECOMMENDER_NAME, strategy_params['shrink'], strategy_params['topK'], 'cosine', 'tidfi',
                              sum(MAP_result) / len(MAP_result)]

            with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(resultsToPrint)

            results = sum(MAP_result) / len(MAP_result)

            rs_results_dict[tuple(strategy_params.values())] = {'score': results}

            if results > best_params_dict['score']:
                best_params_dict['score'] = results
                best_params_dict['params'] = list(strategy_params.values())

        # Save the results in dataframe and sort it based on score
        df_rs_results = pd.DataFrame(rs_results_dict).T.reset_index()
        df_rs_results.columns = list(strategy_params.keys()) + ['score']
        df_rs_results = df_rs_results.sort_values(['score'], ascending=False).head(num_epochs - epoch)

        # If the best score from this epoch is worse than the best score,
        # then append the best hyperaparameters combination to this epoch dataframe
        if df_rs_results['score'].iloc[0] < best_params_dict['score']:
            new_row_dict = {}
            new_row_dict['score'] = best_params_dict['score']
            for idx, key in enumerate(init_param_grid):
                new_row_dict[key] = best_params_dict['params'][idx]

            df_rs_results = df_rs_results.append(new_row_dict, ignore_index=True)
            df_rs_results = df_rs_results.sort_values(['score'], ascending=False).head(num_epochs - epoch)

        # display(df_rs_results)
        print(df_rs_results.head(1).T.to_dict())

        # Get the worse and best hyperparameter combinations
        df_rs_results_min = df_rs_results[df_rs_results['score'] > worse_score].min(axis=0)
        df_rs_results_max = df_rs_results[df_rs_results['score'] > worse_score].max(axis=0)

        # Generate new hyperparameter space based on current worse and best hyperparameter combinations
        for key in init_param_grid:
            if isinstance(init_param_grid[key][0], int):
                new_param_grid[key] = np.unique([i for i in range(int(df_rs_results_min[key]), int(df_rs_results_max[key]) + 1)])
            elif isinstance(init_param_grid[key][0], float):
                new_param_grid[key] = np.unique(np.linspace(df_rs_results_min[key], df_rs_results_max[key], grid_size))
            else:
                new_param_grid[key] = init_param_grid[key]

        # Decrease the tuning iteration for random search
        TUNE_ITER = int(TUNE_ITER - epoch * TUNE_ITER / num_epochs)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP
    """
    topK = study.best_params['topK']
    shrink = study.best_params['shrink']
    similarity = study.best_params['similarity']
    normalization = study.best_params['normalization']"""

    topK = best_params_dict['params'][1]
    shrink = best_params_dict['params'][0]
    similarity = 'cosine'
    normalization = 'tfidf'

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
        #json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
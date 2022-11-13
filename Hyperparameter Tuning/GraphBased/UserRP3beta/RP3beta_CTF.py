if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    import json
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import ParameterSampler

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM

    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    grid_size = 100
    TUNE_ITER = 0
    num_epochs = 0
    worse_score = 0

    init_param_grid = {'topK': [i for i in range(10, 500)],
                       'alpha': [i for i in np.arange(0.1, 0.99, 0.01)],
                       'beta': [i for i in np.arange(0.1, 0.99, 0.01)]
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
            recommender_RP3beta_list = []

            for index in range(len(URM_train_list)):
                recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index], verbose=False))
                recommender_RP3beta_list[index].fit(alpha=strategy_params['alpha'], topK=strategy_params['topK'], beta=strategy_params['beta'])
                recommender_RP3beta_list[index].URM_Train = URM_train_list[index]

            MAP_result = evaluator_validation.evaluateRecommender(recommender_RP3beta_list)
            MAP_results_list.append(MAP_result)
            results = sum(MAP_result) / len(MAP_result)

            print(tune_iter)

            rs_results_dict[tuple(strategy_params.values())] = {'score': results}

            if results > best_params_dict['score']:
                best_params_dict['score'] = results
                best_params_dict['params'] = list(strategy_params.values())

        # Save the results in dataframe and sort it based on score
        df_rs_results = pd.DataFrame(rs_results_dict).T.reset_index()
        df_rs_results.columns = list(strategy_params.keys()) + ['score']
        df_rs_results = df_rs_results.sort_values(['score'], ascending=False).head(num_epochs - epoch)

        # If the best score from this epoch is worse than the best score,
        # then append the best hyperparameters combination to this epoch dataframe
        if df_rs_results['score'].iloc[0] < best_params_dict['score']:
            new_row_dict = {}
            new_row_dict['score'] = best_params_dict['score']
            for idx, key in enumerate(init_param_grid):
                new_row_dict[key] = best_params_dict['params'][idx]

            df_rs_results = df_rs_results.append(new_row_dict, ignore_index=True)
            df_rs_results = df_rs_results.sort_values(['score'], ascending=False).head(num_epochs - epoch)

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

    recommender_RP3beta = RP3betaRecommender(URM_train_init, verbose=False)
    recommender_RP3beta.fit(alpha=best_params_dict['params'][1], topK=best_params_dict['params'][0], beta=best_params_dict['params'][2])

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_RP3beta)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)
    resultString = 'topK = ' + str(best_params_dict['params'][0]) + '    alpha = ' + str(best_params_dict['params'][1]) + '    beta = ' + str(best_params_dict['params'][2])
    with open("logs/" + recommender_RP3beta.RECOMMENDER_NAME + "CTF_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(resultString, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
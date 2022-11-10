if __name__ == '__main__':
    import pandas as pd
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Utils.recsys2022DataReader import createBumpURM
    from Utils.recsys2022DataReader import createSmallICM
    import json
    import numpy as np
    from sklearn.model_selection import ParameterSampler

    URM = createBumpURM()

    ICM = createSmallICM()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = ItemKNNCBFRecommender(URM_train, ICM)

    grid_size = 100
    TUNE_ITER = 20
    num_epochs = 5
    worse_score = 0

    init_param_grid = {'topK': [i for i in range(10, 400)],
                       'shrink': [i for i in np.arange(10, 100)],
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

            recommender = ItemKNNCBFRecommender(URM_train, ICM)
            recommender.fit(shrink=strategy_params['shrink'], topK=strategy_params['topK'])
            results, _ = evaluator_validation.evaluateRecommender(recommender)
            results = results.loc[10]['MAP']

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

        print(df_rs_results.head(1).T.to_dict())

        # Get the worse and best hyperparameter combinations
        df_rs_results_min = df_rs_results[df_rs_results['score'] > worse_score].min(axis=0)
        df_rs_results_max = df_rs_results[df_rs_results['score'] > worse_score].max(axis=0)

        # Generate new hyperparameter space based on current worse and best hyperparameter combinations
        for key in init_param_grid:
            if isinstance(init_param_grid[key][0], int):
                new_param_grid[key] = np.unique(
                    [i for i in range(int(df_rs_results_min[key]), int(df_rs_results_max[key]) + 1)])
            elif isinstance(init_param_grid[key][0], float):
                new_param_grid[key] = np.unique(np.linspace(df_rs_results_min[key], df_rs_results_max[key], grid_size))
            else:
                new_param_grid[key] = init_param_grid[key]

        # Decrease the tuning iteration for random search
        TUNE_ITER = int(TUNE_ITER - epoch * TUNE_ITER / num_epochs)


    topK = best_params_dict['params'][0]
    shrink = best_params_dict['params'][1]
    recommender = ItemKNNCBFRecommender(URM_train, ICM)
    recommender.fit(shrink=shrink, topK=topK)
    result_df, _ = evaluator_test.evaluateRecommender(recommender)

    resultToSave = 'MAP = ' + str(best_params_dict['score']) + '    topK = ' + str(best_params_dict['params'][0]) + '   shrink = ' + str(best_params_dict['params'][1])
    with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(resultToSave,json_file, indent=4)
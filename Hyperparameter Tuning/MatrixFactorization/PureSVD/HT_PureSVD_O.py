if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
    from Utils.recsys2022DataReader import createBumpURM
    import json
    import optuna as op

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator
    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(2):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_PureSVD_list = []

        """ Max Intervals:
        topk: [10, 1000]
        shrink: [10, 1000]
        similarity: ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean']
        feature_weighting: ["BM25", "TF-IDF", "none"]
        """

        topK = trial.suggest_int("topK", 10, 500)
        num_factors = trial.suggest_int("num_factors", 200, 400)
        #similarity = trial.suggest_categorical("similarity", ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean'])
        #feature_weighting = trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"])

        for index in range(len(URM_train_list)):

            recommender_PureSVD_list.append(PureSVDItemRecommender(URM_train_list[index], verbose=False))
            recommender_PureSVD_list[index].fit(num_factors=num_factors, topK=topK)
            recommender_PureSVD_list[index].URM_Train = URM_train_list[index]

        MAP_result = evaluator_validation.evaluateRecommender(recommender_PureSVD_list)
        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)

    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    num_factors = study.best_params['num_factors']
    topK = study.best_params['topK']

    recommender_PureSVD = PureSVDItemRecommender(URM_train_init, verbose=False)
    recommender_PureSVD.fit(num_factors=num_factors, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_PureSVD)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_PureSVD.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
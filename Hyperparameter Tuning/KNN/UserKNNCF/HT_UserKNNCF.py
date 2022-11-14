if __name__ == "__main__":

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    import optuna as op
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM

    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(2):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_UserKNNCF_list = []

        """ Max Intervals:
        topk: [10, 1000]
        shrink: [10, 1000]
        similarity: ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean']
        feature_weighting: ["BM25", "TF-IDF", "none"]
        """

        topK = trial.suggest_int("topK", 10, 5000)
        shrink = trial.suggest_float("shrink", 10, 1000)
        # similarity = trial.suggest_categorical("similarity", ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean'])
        # feature_weighting = trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"])

        for index in range(len(URM_train_list)):
            recommender_UserKNNCF_list.append(UserKNNCFRecommender(URM_train_list[index], verbose=False))

            recommender_UserKNNCF_list[index].fit(shrink=shrink, topK=topK)

            recommender_UserKNNCF_list[index].URM_Train = URM_train_list[index]

        MAP_result = evaluator_validation.evaluateRecommender(recommender_UserKNNCF_list)

        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=2)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']
    # similarity = study.best_params['similarity']
    # feature_weighting = study.best_params['feature_weighting']

    recommender_UserKNNCF = UserKNNCFRecommender(URM_train, verbose=False)

    recommender_UserKNNCF.fit(shrink=shrink, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_UserKNNCF)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_UserKNNCF.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
if __name__ == "__main__":

    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_SlimBPR_list = []

        """Max Intervals:
        topK: [5, 1000]
        epochs: [10, 100]
        lambda_i: [1e-5, 1e-2]
        lambda_j: [1e-5, 1e-2]
        """

        topK = trial.suggest_int("topK", 5, 1000)
        epochs = trial.suggest_int("epochs", 10, 100)
        lambda_i = trial.suggest_float("lambda_i", 1e-5, 1e-2)
        lambda_j = trial.suggest_float("lambda_j", 1e-5, 1e-2)

        for index in range(len(URM_train_list)):

            recommender_SlimBPR_list.append(SLIM_BPR_Cython(URM_train_list[index], verbose=False))
            recommender_SlimBPR_list[index].fit(topK=topK, epochs=epochs, lambda_j=lambda_j, lambda_i=lambda_i)
            recommender_SlimBPR_list[index].URM_Train = URM_train_list[index]

        MAP_result = evaluator_validation.evaluateRecommender(recommender_SlimBPR_list)
        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=2)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params["topK"]
    epochs = study.best_params["epochs"]
    lambda_i = study.best_params["lambda_i"]
    lambda_j = study.best_params["lambda_j"]

    recommender_SlimBPR = SLIM_BPR_Cython(URM_train=URM_train_init, verbose=False)
    recommender_SlimBPR.fit(topK=topK, epochs=epochs, lambda_j=lambda_j, lambda_i=lambda_i)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SlimBPR)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_SlimBPR.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
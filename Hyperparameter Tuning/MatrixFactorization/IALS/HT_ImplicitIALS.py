if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['MKL_NUM_THREADS'] = '1'
    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommenders_IALS_list = []

        n_factors = trial.suggest_int("n_factors", 100, 300)
        regularization = trial.suggest_float("regularization", 1e-6, 1e-1)
        alpha = trial.suggest_float("alpha", 10, 50)
        iterations = trial.suggest_int("iterations", 1, 100)

        for index in range(len(URM_train_list)):
            recommenders_IALS_list.append(ImplicitALSRecommender(URM_train_list[index], verbose=False))

            recommenders_IALS_list[index].fit(factors=n_factors, regularization=regularization, alpha=alpha, iterations=10)

            recommenders_IALS_list[index].URM_Train = URM_train_list[index]


        MAP_result = evaluator_validation.evaluateRecommender(recommenders_IALS_list)

        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    n_factors = study.best_params['n_factors']
    regularization = study.best_params['regularization']
    alpha = study.best_params['alpha']
    iterations = study.best_params['iterations']

    recommender_IALS = ImplicitALSRecommender(URM_train, verbose=False)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender_IALS.fit(num_factors=600, reg=regularization, alpha=alpha, epochs=10)

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_IALS.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)

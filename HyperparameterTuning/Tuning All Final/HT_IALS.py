if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_FinalURMTrainInit()
    URM_train_list = load_1K_FinalURMTrain()
    URM_validation_list = load_1K_FinalURMValid()
    URM_test = load_FinalURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_IALS_list = []

        factors = trial.suggest_int("factors", 97, 97)
        alpha = trial.suggest_int("alpha", 6, 6)
        iterations = trial.suggest_int("iterations", 40, 70)
        regularization = trial.suggest_float("regularization", 0.0039, 0.0041)

        for index in range(len(URM_train_list)):

            recommender_IALS_list.append(ImplicitALSRecommender(URM_train_list[index]))
            recommender_IALS_list[index].fit(alpha=alpha, factors=factors, regularization=regularization, iterations=iterations)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_IALS_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    factors = study.best_params['factors']
    alpha = study.best_params['alpha']
    regularization = study.best_params['regularization']
    iterations = study.best_params['iterations']

    recommender_RP3beta = ImplicitALSRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=alpha, factors=factors, regularization=regularization, iterations=iterations)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_RP3beta)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_RP3beta.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
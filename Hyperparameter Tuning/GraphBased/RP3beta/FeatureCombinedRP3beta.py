if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createURM, createSmallICM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    import optuna as op
    import json
    import csv
    from Utils.crossKValidator import CrossKValidator
    from Utils.combine_matrix import combineKFold, combine

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM

    URM = createURM()
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'topK', 'gamma', 'MAP']
    partialsFile = 'partials_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    cross_validator = CrossKValidator(URM_train_init, k=3)
    evaluator_validation, URM_train_list, URM_validation_list = cross_validator.create_k_evaluators()

    #URMs_stack_list = combineKFold(URM_train_list, ICM)

    MAP_results_list = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_RP3beta_list = []

        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        beta = trial.suggest_float("beta", 0.1, 0.9)
        topK = trial.suggest_int("topK", 100, 500)
        gamma = trial.suggest_int("gamma", 10, 100)

        for index in range(len(URM_train_list)):

            URM_stack = combine(URM_train_list[index], gamma * ICM)

            recommender_RP3beta_list.append(RP3betaRecommender(URM_stack, verbose=False))
            recommender_RP3beta_list[index].fit(alpha=alpha, topK=topK, beta=beta)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_RP3beta_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_RP3beta_list[0].RECOMMENDER_NAME, alpha, beta, topK, gamma, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    alpha = study.best_params['alpha']
    beta = study.best_params['beta']
    gamma = study.best_params['gamma']

    URM_final_test = combine(URM_train_init, ICM * gamma)
    recommender_RP3beta = RP3betaRecommender(URM_final_test, verbose=False)
    recommender_RP3beta.fit(alpha=alpha, topK=topK, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_RP3beta)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_RP3beta.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
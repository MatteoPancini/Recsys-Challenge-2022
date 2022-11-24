if __name__ == '__main__':

    from optuna.samplers import TPESampler
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from Utils.recsys2022DataReader import createURMNEW3
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    import optuna as op
    import json


    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createURMNEW3()

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


    def objective(trial):

        recommender_IALS_list = []
        
        n_factors = trial.suggest_int("n_factors", 100, 300)
        regularization = trial.suggest_float("regularization", 1e-6, 1e-1)
        alpha_val = trial.suggest_float("alpha_val", 10, 50)
        iterations = trial.suggest_int("iterations", 1, 100)
        
        for index in range(len(URM_train_list)):

            recommender_IALS_list.append(IALSRecommender(URM_train_list[index], verbose=False))
            recommender_IALS_list[index].fit(num_factors=n_factors, reg=regularization, alpha=alpha_val, epochs=10, **{
                'epochs_min' : 0,
                'evaluator_object' : evaluator_validation,
                'stop_on_validation' : True,
                'validation_every_n' : 1,
                'validation_metric' : 'MAP',
                'lower_validations_allowed' : 3
            })

        MAP_result = evaluator_validation.evaluateRecommender(recommender_IALS_list)
        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))
    study.optimize(objective, n_trials=3)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    n_factors = study.best_params['n_factors']
    regularization = study.best_params['regularization']
    alpha_val = study.best_params['alpha_val']
    iterations = study.best_params['iterations']

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

    recommender_IALS = IALSRecommender(URM_train, verbose=False)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender_IALS.fit(num_factors=600, reg=regularization, alpha=alpha_val, epochs=10, **{
                'epochs_min' : 0,
                'evaluator_object' : evaluator_test,
                'stop_on_validation' : True,
                'validation_every_n' : 1,
                'validation_metric' : 'MAP',
                'lower_validations_allowed' : 3
            })

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_IALS.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)



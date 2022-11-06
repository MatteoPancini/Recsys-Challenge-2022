
if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
    from Utils.createURM import createURM
    import json
    import optuna as op

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createURM()

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender = PureSVDItemRecommender(URM_train, verbose=False)


    def objective(trial):
        num_factors = trial.suggest_float("num_factors", 10, 800)
        topK = trial.suggest_float("topK", 300, 400)
        recommender.fit(num_factors=int(num_factors), topK=int(topK))
        result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
        return result_dict.loc[10]["MAP"]


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    num_factors = study.best_params['num_factors']
    topK = study.best_params['topK']
    recommender.fit(num_factors=int(num_factors), topK=int(topK))
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)


    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
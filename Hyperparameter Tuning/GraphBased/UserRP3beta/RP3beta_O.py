if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Utils.createURM import createBumpURM
    from Utils.combineURMICM import combine
    from Utils.createICM import createICM
    import pandas as pd
    import optuna as op
    from optuna.samplers import TPESampler
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    dataset = pd.read_csv('../../../Input/interactions_and_impressions.csv')
    URM = createBumpURM(dataset)
    type = pd.read_csv('../../../Input/data_ICM_type.csv')
    length = pd.read_csv('../../../Input/data_ICM_length.csv')
    ICM = createICM(length, type, dataset)


    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

    URMICMCombined = combine(ICM, URM_train)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model
    recommender = RP3betaRecommender(URMICMCombined.T, verbose=False)

    def objective(trial):

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 1)
        topK = trial.suggest_int("topK", 300, 1000)

        recommender.fit(alpha=alpha, beta=beta, topK=topK, implicit=True)
        result, _ = evaluator_validation.evaluateRecommender(recommender)
        result = result.loc[10]['MAP']
        return result

    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))
    study.optimize(objective, n_trials=50)


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']
    topK = study.best_params['topK']

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)

    URMICMCombined = combine(ICM, URM_train)

    recommender = RP3betaRecommender(URMICMCombined.T, verbose=False)
    recommender.fit(alpha=alpha, beta=beta, topK= topK, implicit=True)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
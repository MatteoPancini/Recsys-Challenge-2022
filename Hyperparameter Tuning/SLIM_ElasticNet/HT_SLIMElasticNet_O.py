if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.createURM import tryURM
    import pandas as pd
    import optuna as op
    from optuna.samplers import TPESampler
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    dataset = pd.read_csv('../../Input/interactions_and_impressions.csv')
    URM = tryURM(dataset)

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URMs_train = []
    URMs_validation = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

    results = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train, verbose=False)

    recommenders = []
    def objective(trial):

        recommenders = []

        alpha = trial.suggest_float("alpha", 0.00001, 0.1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.000001, 0.1, log=True)
        topK = trial.suggest_int("topK", 300, 400, log=True)

        for index in range(len(URMs_train)):

            recommenders.append(MultiThreadSLIM_SLIMElasticNetRecommender(URMs_train[index], verbose=False))

            recommenders[index].fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

            recommenders[index].URM_train = URMs_train[index] # utile in caso di combine

        result = evaluator_validation.evaluateRecommender(recommenders)

        results.append(result)

        return sum(result) / len(result)


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))
    study.optimize(objective, n_trials=3)


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    alpha = study.best_params['alpha']
    l1_ratio = study.best_params['l1_ratio']
    topK = study.best_params['topK']

    recommenders.append(MultiThreadSLIM_SLIMElasticNetRecommender(URMs_train[0], verbose=False))

    recommenders[0].fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

    result_dict, _ = evaluator_validation.evaluateRecommender(recommenders[0])

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
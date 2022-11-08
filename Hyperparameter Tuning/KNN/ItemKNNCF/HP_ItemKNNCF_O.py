if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Utils.createURM import createURM
    import optuna as op
    from optuna.samplers import TPESampler
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URMs_train = []
    URMs_validation = []

    URM_train_start, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_start, train_percentage=0.80)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    results = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender = ItemKNNCFRecommender(URM_train)
    recommenders = []

    def objective(trial):

        recommenders = []

        topK = trial.suggest_int("topK", 175, 195)
        shrink = trial.suggest_float("shrink", 50, 56)

        for index in range(len(URMs_train)):
            recommenders.append(ItemKNNCFRecommender(URMs_train[index]))

            recommenders[index].fit(topK=topK, shrink=shrink)

            recommenders[index].URM_train = URMs_train[index]  # utile in caso di combine

        result = evaluator_validation.evaluateRecommender(recommenders)
        results.append(result)

        return sum(result) / len(result)


    study = op.create_study(direction='maximize', sampler=TPESampler())

    study.optimize(objective, n_trials=1)


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']

    recommenders.append(ItemKNNCFRecommender(URMs_train[0]))

    recommenders[0].fit(shrink=shrink, topK=topK)

    result_dict, _ = evaluator_test.evaluateRecommender(recommenders[0])

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
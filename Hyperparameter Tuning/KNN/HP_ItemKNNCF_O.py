if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Utils.createURM import createURM
    import optuna as op
    from optuna.samplers import TPESampler
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createURM()

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender = ItemKNNCFRecommender(URM_train)

    def objective(trial):
        topK = trial.suggest_int("topK", 175, 195)
        shrink = trial.suggest_float("shrink", 50, 56)
        recommender.fit(topK=topK, shrink=shrink)
        result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

        return result_dict.loc[10]["MAP"]


    study = op.create_study(direction='maximize', pruner=op.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    ), sampler=TPESampler())

    study.optimize(objective, n_trials=10)

    # ---------------------------------------------------------------------------------------------------------
    # Visualization of hyperparameters

    op.visualization.matplotlib.plot_param_importances(study)
    op.visualization.matplotlib.plot_optimization_history(study)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']

    recommender.fit(shrink=shrink, topK=topK)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
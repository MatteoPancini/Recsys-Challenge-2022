if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import ParameterSampler
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.createURM import createURM
    import optuna as op
    from optuna.samplers import TPESampler
    import json

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

    recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train, verbose=False)

    #TODO: provare il partial_fit con il pruning

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.00001, 0.1)
        l1_ratio = trial.suggest_float("l1_ratio", 0.000001, 0.1)
        topK = trial.suggest_float("topK", 300, 400)
        recommender.fit(alpha=alpha, l1_ratio=l1_ratio, topK=int(topK))

        result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

        return result_dict.loc[10]["MAP"]


    study = op.create_study(direction='maximize', pruner=op.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    ), sampler=TPESampler())
    study.optimize(objective, n_trials=5)


    # ---------------------------------------------------------------------------------------------------------
    # Visualization of hyperparameters

    op.visualization.matplotlib.plot_param_importances(study)
    op.visualization.matplotlib.plot_optimization_history(study)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    alpha = study.best_params['alpha']
    l1_ratio = study.best_params['l1_ratio']
    topK = study.best_params['topK']

    recommender.fit(alpha=alpha, l1_ratio=l1_ratio, topK=int(topK))
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
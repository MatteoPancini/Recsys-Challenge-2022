if __name__ == "__main__":

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import *
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op
    import numpy as np
    import csv
    from optuna.samplers import RandomSampler

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

        recommender_SlimElasticnet_list = []

        topK = trial.suggest_int("topK", 2600, 2700)
        alpha = trial.suggest_float("alpha", 0.002, 0.003)
        l1_ratio = trial.suggest_float("l1_ratio", 0.02, 0.03)

        for index in range(len(URM_train_list)):
            recommender_SlimElasticnet_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_list[index]))
            recommender_SlimElasticnet_list[index].fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_SlimElasticnet_list)

        return sum(MAP_result) / len(MAP_result)

    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    alpha = study.best_params['alpha']
    l1_ratio = study.best_params['l1_ratio']

    recommender_SlimElasticNet = SLIMElasticNetRecommender(URM_train_init)
    recommender_SlimElasticNet.fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SlimElasticNet)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/Multi" + recommender_SlimElasticNet.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
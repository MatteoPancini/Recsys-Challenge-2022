if __name__ == '__main__':

    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommender
    from optuna.samplers import TPESampler
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    import optuna as op
    import json
    from datetime import datetime

    URM = createBumpURM()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, 0.85)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, 0.85)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    ItemKNNCF_recommender = ItemKNNCFRecommender(URM_train)
    SLIM_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train)

    """
    def objective(trial):
        topKCF = trial.suggest_int("topKCF", 300, 400)
        shrinkCF = trial.suggest_float("shrinkCF", 10, 100)
        alphaSLM = trial.suggest_float("alphaSLM", 0.00001, 0.1)
        l1_ratioSLM = trial.suggest_float("l1_ratioSLM", 0.000001, 0.1)
        topKSLM = trial.suggest_int("topKSLM", 300, 400)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)

        ItemKNNCF_recommender.fit(topK=topKCF, shrink=shrinkCF)
        SLIM_recommender.fit(topK=topKSLM, alpha=alphaSLM, l1_ratio=l1_ratioSLM)

        recommender = LinearHybridTwoRecommender(URM_train, Recommender_1=ItemKNNCF_recommender, Recommender_2=SLIM_recommender)
        recommender.fit(alpha=alpha)

        result, _ = evaluator_validation.evaluateRecommender(recommender)

        return result.loc[10]['MAP']


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))
    study.optimize(objective, n_trials=1)"""

    ItemKNNCF_recommender.fit(topK=329, shrink=97)
    SLIM_recommender.fit(topK=362, alpha=0.06747829810332745, l1_ratio=0.0005493724398243842)

    recommender = LinearHybridTwoRecommender(URM_train, Recommender_1=ItemKNNCF_recommender, Recommender_2=SLIM_recommender)
    recommender.fit(alpha=0.2)

    result, _ = evaluator_test.evaluateRecommender(recommender)

    resultParameters = result.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        #json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
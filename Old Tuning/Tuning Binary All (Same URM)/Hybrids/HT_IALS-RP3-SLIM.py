if __name__ == "__main__":

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    import csv
    from datetime import datetime
    import optuna as op
    import json
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_1K_BinURMTrain()
    URM_validation_list = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_IALS_list = []
    recommender_RP3beta_list = []
    recommender_SLIM_List = []

    for i in range(len(URM_train_list)):

        recommender_IALS_list.append(ImplicitALSRecommender(URM_train=URM_train_list[i]))
        recommender_IALS_list[i].fit(factors=110, alpha=7, iterations=57, regularization=0.0008866558623568822)

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[i], verbose=False))
        recommender_RP3beta_list[i].fit(topK=77, alpha=0.8401946814961014, beta=0.3073181471251768)

        recommender_SLIM_List.append(SLIMElasticNetRecommender(URM_train_list[i]))
        recommender_SLIM_List[i].fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    hybrid1 = LinearHybridTwoRecommenderTwoVariables(URM_train_list[0], Recommender_1=recommender_RP3beta_list[0], Recommender_2=recommender_SLIM_List[0])
    hybrid1.fit(alpha=0.40726736669265445, beta=0.7317482903276693)

    def objective(trial):

        recommender_hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 0.1)
        beta = trial.suggest_float("beta", 0.6, 0.9)

        for i in range(len(URM_train_list)):
            recommender_hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[i], Recommender_1=recommender_IALS_list[i], Recommender_2=hybrid1))
            recommender_hybrid_list[i].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    rec1 = ImplicitALSRecommender(URM_train_init)
    rec1.fit(factors=110, alpha=7, iterations=57, regularization=0.0008866558623568822)

    rec2 = RP3betaRecommender(URM_train_init)
    rec2.fit(topK=77, alpha=0.8401946814961014, beta=0.3073181471251768)

    rec3 = SLIMElasticNetRecommender(URM_train_init)
    rec3.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    hybrid1 = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec2,
                                                     Recommender_2=rec3)
    hybrid1.fit(alpha=0.40726736669265445, beta=0.7317482903276693)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec1, Recommender_2=hybrid1)
    recommender_hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "IALS-RP3beta-SLIM" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
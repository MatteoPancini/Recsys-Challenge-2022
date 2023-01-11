if __name__ == "__main__":

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    from datetime import datetime
    import optuna as op
    import json
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_BinURMTrainInit()
    URM_train = load_1K_BinURMTrain()[0]
    URM_validation = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    IALS_train = ImplicitALSRecommender(URM_train=URM_train)
    IALS_train.fit(factors=110, alpha=7, iterations=57, regularization=0.0008866558623568822)

    ItemKNN_train = ItemKNNCFRecommender(URM_train)
    ItemKNN_train.fit(topK=123, shrink=510, similarity="cosine", feature_weighting="TF-IDF")

    RP3beta_train = RP3betaRecommender(URM_train)
    RP3beta_train.fit(topK=77, alpha=0.8401946814961014, beta=0.3073181471251768)

    SLIM_train = SLIMElasticNetRecommender(URM_train)
    SLIM_train.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    hybrid1_train = LinearHybridTwoRecommenderTwoVariables(URM_train, Recommender_1=RP3beta_train, Recommender_2=SLIM_train)
    hybrid1_train.fit(alpha=0.40726736669265445, beta=0.7317482903276693)

    hybrid2_train = LinearHybridTwoRecommenderTwoVariables(URM_train, Recommender_1=IALS_train, Recommender_2=hybrid1_train)
    hybrid2_train.fit(alpha=0.019318928403041356, beta=0.8537494424674974)

    hybrid3_train = LinearHybridTwoRecommenderTwoVariables(URM_train, Recommender_1=hybrid2_train, Recommender_2=ItemKNN_train)
    hybrid3_train.fit(alpha=0.5182132379810547, beta=4.19321787406275e-06)

    UserKNN_train = UserKNNCFRecommender(URM_train)
    UserKNN_train.fit(topK=500, shrink=10)


    def objective(trial):

        alpha = trial.suggest_float("alpha", 0.5, 1)
        beta = trial.suggest_float("beta", 0, 0.1)
        recommender_hybrid = []

        recommender_hybrid.append(LinearHybridTwoRecommenderTwoVariables(URM_train, Recommender_1=hybrid3_train, Recommender_2=UserKNN_train))
        recommender_hybrid[0].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    IALS_init = ImplicitALSRecommender(URM_train_init)
    IALS_init.fit(factors=110, alpha=7, iterations=57, regularization=0.0008866558623568822)

    RP3beta_init = RP3betaRecommender(URM_train_init)
    RP3beta_init.fit(topK=77, alpha=0.8401946814961014, beta=0.3073181471251768)

    SLIM_init = SLIMElasticNetRecommender(URM_train_init)
    SLIM_init.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    ItemKNN_init = ItemKNNCFRecommender(URM_train_init)
    ItemKNN_init.fit(topK=123, shrink=510, similarity="cosine", feature_weighting="TF-IDF")

    hybrid1_init = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=RP3beta_init,
                                                     Recommender_2=SLIM_init)
    hybrid1_init.fit(alpha=0.40726736669265445, beta=0.7317482903276693)

    hybrid2_init = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=IALS_init, Recommender_2=hybrid1_init)
    hybrid2_init.fit(alpha=0.019318928403041356, beta=0.8537494424674974)

    hybrid3_init = LinearHybridTwoRecommenderTwoVariables(URM_train, Recommender_1=hybrid2_init, Recommender_2=ItemKNN_init)
    hybrid3_init.fit(alpha=0.5182132379810547, beta=4.19321787406275e-06)

    UserKNN_init = UserKNNCFRecommender(URM_train_init)
    UserKNN_init.fit(topK=500, shrink=10)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=hybrid3_init, Recommender_2=UserKNN_init)
    recommender_hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "IALS-RP3beta-SLIM-ItemKNN" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
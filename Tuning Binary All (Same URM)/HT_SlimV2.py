if __name__ == "__main__":

    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import *
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    from bayes_opt import BayesianOptimization
    import numpy as np

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_1K_BinURMTrain()
    URM_validation_list = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    tuning_params = {
        "l1_ratio": (0.00001, 0.001),
        "topK": (100, 800),
        "alpha": (0.00001, 0.001)
    }

    def BO_func(
            l1_ratio,
            topK,
            alpha
    ):
        recommender_SlimElasticnet_list = []

        for index in range(len(URM_train_list)):
            recommender_SlimElasticnet_list.append(SLIMElasticNetRecommender(URM_train_list[index]))
            recommender_SlimElasticnet_list[index].fit(alpha=alpha, l1_ratio=l1_ratio, topK=int(topK))

        MAP_result = evaluator_validation.evaluateRecommender(recommender_SlimElasticnet_list)

        return sum(MAP_result) / len(MAP_result)


    optimizer = BayesianOptimization(
        f=BO_func,
        pbounds=tuning_params,
        verbose=5,
        random_state=5,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=15
    )

    topK = optimizer.max["params"]["topK"]
    l1_ratio = optimizer.max["params"]["l1_ratio"]
    alpha = optimizer.max["params"]["alpha"]

    recommender_SlimElasticNet = SLIMElasticNetRecommender(URM_train_init)
    recommender_SlimElasticNet.fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SlimElasticNet)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_SlimElasticNet.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(optimizer.max, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
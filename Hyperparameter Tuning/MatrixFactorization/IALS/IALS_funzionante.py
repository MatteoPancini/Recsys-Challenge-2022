if __name__ == '__main__':
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    import json
    from bayes_opt import BayesianOptimization
    from Utils.recsys2022DataReader import createURMNEW3
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URM_all = createURMNEW3()

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85)

    URMs_train = []
    URMs_validation = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

    recommenders = []

    for index in range(len(URMs_train)):
        recommenders.append(
            ImplicitALSRecommender(URM_train=URMs_train[index])
        )


    tuning_params = {
        "alpha": (10, 50),
        "factors": (200, 300),
        "iterations": (10, 100)
    }

    results = []

    def BO_func(
            factors,
            iterations,
            alpha
    ):
        for index in range(len(recommenders)):
            recommenders[index].fit(
                factors=int(factors),
                regularization=0.01,
                use_gpu=False,
                iterations=int(iterations),
                num_threads=4,
                **{"alpha": alpha}
            )

        result = evaluator_validation.evaluateRecommender(recommenders)
        results.append(result)
        return sum(result) / len(result)



    optimizer = BayesianOptimization(
        f=BO_func,
        pbounds=tuning_params,
        verbose=5,
        random_state=5,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=20
    )

    alpha = optimizer.max["params"]["alpha"]
    factors = optimizer.max["params"]["factors"]
    iterations = optimizer.max["params"]["iterations"]

    recommender_IALS = ImplicitALSRecommender(URM_train_init)
    recommender_IALS.fit(factors=factors, alpha=alpha, regularization=0.01, iterations=iterations)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)
    print(result_dict)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    #with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
     #   json.dump(parsed, json_file)
     #   json.dump(optimizer.max, json_file)
if __name__ == '__main__':
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Recommenders.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
    from Utils.combine_matrix import combine
    import json
    from bayes_opt import BayesianOptimization
    from Utils.confidence_scaling import linear_scaling_confidence
    from Utils.recsys2022DataReader import createURM, createSmallICM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URM_all = createURM()
    ICM = createSmallICM()

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
            FeatureCombinedImplicitALSRecommender(
                URM_train=URMs_train[index],
                ICM_train=ICM,
                verbose=True
            )
        )

    tuning_params = {
        "urm_alpha": (40, 100),
        "icm_alpha": (40, 100),
        "factors": (200, 300),
        "epochs": (10, 100)
    }

    results = []


    def BO_func(
            factors,
            epochs,
            urm_alpha,
            icm_alpha
    ):
        for index in range(len(recommenders)):
            recommenders[index].fit(
                factors=int(factors),
                regularization=0.01,
                use_gpu=False,
                iterations=int(epochs),
                num_threads=4,
                confidence_scaling = linear_scaling_confidence,
                **{
                    'URM': {"alpha": urm_alpha},
                    'ICM': {"alpha": icm_alpha}
                }
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

    urm_alpha = optimizer.max["params"]["urm_alpha"]
    icm_alpha = optimizer.max["params"]["icm_alpha"]
    factors = optimizer.max["params"]["factors"]
    epochs = optimizer.max["params"]["epochs"]


    recommender_IALS = FeatureCombinedImplicitALSRecommender(URM_train=URM_train_init,ICM_train=ICM,verbose=True)
    recommender_IALS.fit(
                factors=int(factors),
                regularization=0.01,
                use_gpu=False,
                iterations=int(epochs),
                num_threads=4,
                confidence_scaling=linear_scaling_confidence,
                **{
                    'URM': {"alpha": urm_alpha},
                    'ICM': {"alpha": icm_alpha}
                })

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)

    with open("logs/" + recommender_IALS.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(optimizer.max, json_file)
if __name__ == '__main__':
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    import json
    import numpy as np
    from bayes_opt import BayesianOptimization
    from Utils.recsys2022DataReader import createURM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    URM_all = createURM()

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85)

    # ---------------------------------------------------------------------------------------------------------
    # Profiling

    group_id = 0

    profile_length = np.ediff1d(URM_train_init.indptr)
    sorted_users = np.argsort(profile_length)

    interactions = []
    for i in range(41629):
        interactions.append(len(URM_train_init[i, :].nonzero()[0]))

    list_group_interactions = [[0, 20], [21, 49], [50, max(interactions)]]

    lower_bound = list_group_interactions[group_id][0]
    higher_bound = list_group_interactions[group_id][1]

    users_in_group = [user_id for user_id in range(len(interactions))
                      if (lower_bound <= interactions[user_id] <= higher_bound)]
    users_in_group_p_len = profile_length[users_in_group]

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_list = []
    URM_validation_list = []
    users_not_in_group_list = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

        profile_length = np.ediff1d(URM_train_init.indptr)
        sorted_users = np.argsort(profile_length)

        users_in_group = [user_id for user_id in range(len(interactions))
                          if (lower_bound <= interactions[user_id] <= higher_bound)]
        users_in_group_p_len = profile_length[users_in_group]

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group_list.append(sorted_users[users_not_in_group_flag])

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False,
                                                ignore_users_list=users_not_in_group_list)
    MAP_results_list = []

    recommenders = []

    for index in range(len(URM_train_list)):
        recommenders.append(
            ImplicitALSRecommender(URM_train=URM_train_list[index])
        )


    tuning_params = {
        "alpha": (10, 50),
        "factors": (200, 250),
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
        n_iter=40
    )

    alpha = optimizer.max["params"]["alpha"]
    factors = optimizer.max["params"]["factors"]
    iterations = optimizer.max["params"]["iterations"]

    recommender_IALS = ImplicitALSRecommender(URM_train_init)
    recommender_IALS.fit(factors=int(factors), alpha=alpha, regularization=0.01, iterations=int(iterations))

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)
    print(result_dict)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(parsed, json_file)
        json.dump(optimizer.max, json_file)
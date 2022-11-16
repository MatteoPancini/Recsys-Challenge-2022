if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createBumpURM, createSmallICM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridThreeRecommender
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM

    URM = createBumpURM()
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []
    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_SlimBPR_list = []
        recommender_RP3beta_list = []
        recommender_ItemKNNCBF_list = []
        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        beta = trial.suggest_float("beta", 0.1, 0.9)
        gamma = trial.suggest_float("gamma", 0.1, 0.9)

        for index in range(len(URM_train_list)):

            recommender_SlimBPR_list.append(SLIM_BPR_Cython(URM_train_list[index]))
            recommender_SlimBPR_list[index].fit(lambda_j=0.00427112631147574, lambda_i=0.004864871170753198, topK=622, epochs=59)

            recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index]))
            recommender_RP3beta_list[index].fit(alpha=0.2723304259820941, beta=0.34952850616150266, topK=78)

            recommender_ItemKNNCBF_list.append(ItemKNNCBFRecommender(URM_train_list[index], ICM_train=ICM))
            recommender_ItemKNNCBF_list[index].fit(shrink=20.842705935575843, topK=498)

            recommender_Hybrid_list.append(LinearHybridThreeRecommender(URM_train=URM_train_list[index], Recommender_1=recommender_SlimBPR_list[index], Recommender_2=recommender_RP3beta_list[index], Recommender_3=recommender_ItemKNNCBF_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha, beta=beta, gamma=gamma)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']
    gamma = study.best_params['gamma']

    recommender_SlimBPR = SLIM_BPR_Cython(URM_train_init)
    recommender_SlimBPR.fit(lambda_j=0.00427112631147574, lambda_i=0.004864871170753198, topK=622, epochs=59)

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=0.2723304259820941, beta=0.34952850616150266, topK=78)

    recommender_ItemKNNCBF = ItemKNNCBFRecommender(URM_train_init, ICM)
    recommender_ItemKNNCBF.fit(shrink=20.842705935575843, topK=498)

    recommender_Hybrid = LinearHybridThreeRecommender(URM_train= URM_train_init, Recommender_1=recommender_SlimBPR,Recommender_2=recommender_RP3beta, Recommender_3=recommender_ItemKNNCBF)
    recommender_Hybrid.fit(alpha=alpha, beta=beta, gamma=gamma)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_Hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_Hybrid.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
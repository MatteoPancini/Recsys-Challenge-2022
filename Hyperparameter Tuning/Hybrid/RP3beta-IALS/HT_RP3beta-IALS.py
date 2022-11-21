if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommender
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

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

        recommender_IALS_list = []
        recommender_RP3beta_list = []
        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0.1, 0.9)

        for index in range(len(URM_train_list)):
            recommender_IALS_list.append(ImplicitALSRecommender(URM_train_list[index]))
            recommender_IALS_list[index].fit(factors=250, alpha=10, regularization=0.01, iterations=89)

            recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index]))
            recommender_RP3beta_list[index].fit(alpha=0.2723304259820941, beta=0.34952850616150266, topK=78)

            recommender_Hybrid_list.append(LinearHybridTwoRecommender(URM_train=URM_train_list[index], Recommender_1=recommender_IALS_list[index], Recommender_2=recommender_RP3beta_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']

    recommender_IALS = ImplicitALSRecommender(URM_train_init)
    recommender_IALS.fit(factors=250, alpha=10, regularization=0.01, iterations=89)

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=0.2723304259820941, beta=0.34952850616150266, topK=78)

    recommender_Hybrid = LinearHybridTwoRecommender(URM_train=URM_train_init, Recommender_1=recommender_IALS, Recommender_2=recommender_RP3beta)
    recommender_Hybrid.fit(alpha=alpha)

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
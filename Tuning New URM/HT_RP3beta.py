if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    import optuna as op
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    import json
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
    URM = createRP3betaURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_RpP3beta_list = []

        topK = trial.suggest_int("topK", 10, 500)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        beta = trial.suggest_float("beta", 0.1, 0.9)

        for index in range(len(URM_train_list)):

            recommender_RpP3beta_list.append(RP3betaRecommender(URM_train_list[index]))
            recommender_RpP3beta_list[index].fit(alpha=alpha, beta=beta, topK=topK)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_RpP3beta_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=alpha, beta=beta, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_RP3beta)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_RP3beta.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
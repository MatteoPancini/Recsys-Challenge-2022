if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_FinalURMTrainInit()
    URM_train_list = load_1K_FinalURMTrain()
    URM_validation_list = load_1K_FinalURMValid()
    URM_test = load_FinalURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_IALS_list = []

        topK = trial.suggest_int("topK", 1000, 5000)
        l2_norm = trial.suggest_float("l2_norm", 1e-5, 1e-1)

        for index in range(len(URM_train_list)):

            recommender_IALS_list.append(EASE_R_Recommender(URM_train_list[index]))
            recommender_IALS_list[index].fit(topK=topK, l2_norm=l2_norm)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_IALS_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=80)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    l2_norm = study.best_params['l2_norm']

    recommender_RP3beta = EASE_R_Recommender(URM_train_init)
    recommender_RP3beta.fit(topK=topK, l2_norm=l2_norm)

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
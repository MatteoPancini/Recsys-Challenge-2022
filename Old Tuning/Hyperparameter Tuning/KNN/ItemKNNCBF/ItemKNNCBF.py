if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createBigURM, createBigICM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM

    URM = createBigURM()
    ICM = createBigICM()

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

        recommender_ItemKNNCBF_list = []

        topK = trial.suggest_int("topK", 10, 500)
        shrink = trial.suggest_float("shrink", 10, 100)

        for index in range(len(URM_train_list)):

            recommender_ItemKNNCBF_list.append(ItemKNNCBFRecommender(URM_train_list[index], ICM, verbose=False))
            recommender_ItemKNNCBF_list[index].fit(shrink=shrink, topK=topK)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCBF_list)
        MAP_results_list.append(MAP_result)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']

    recommender_ItemKNNCBF = ItemKNNCBFRecommender(URM_train_init, ICM, verbose=False)
    recommender_ItemKNNCBF.fit(shrink=shrink, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_ItemKNNCBF)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_ItemKNNCBF.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
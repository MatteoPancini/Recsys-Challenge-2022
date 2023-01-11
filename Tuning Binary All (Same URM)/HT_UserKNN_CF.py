if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_1K_BinURMTrain()
    URM_validation_list = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)
    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_ItemKNNCF_list = []

        topK = trial.suggest_int("topK", 10, 500)
        shrink = trial.suggest_int("shrink", 10, 200)

        for index in range(len(URM_train_list)):
            recommender_ItemKNNCF_list.append(UserKNNCFRecommender(URM_train=URM_train_list[index]))
            recommender_ItemKNNCF_list[index].fit(shrink=shrink, topK=topK)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=250)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']

    recommender_ItemKNNCF = UserKNNCFRecommender(URM_train_init, verbose=False)
    recommender_ItemKNNCF.fit(shrink=shrink, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_ItemKNNCF)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_ItemKNNCF.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)

if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    import optuna as op
    import json
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'topk', 'shrink', 'MAP']

    partialsFile = 'ItemKNNCBF_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
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
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init,train_percentage=0.85)
            URM_train_list.append(URM_train)
            URM_validation_list.append(URM_validation)

        evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_ItemKNNCBF_list = []

        topK = trial.suggest_int("topK", 10, 500)
        shrink = trial.suggest_float("shrink", 10, 200)

        for index in range(len(URM_train_list)):

            recommender_ItemKNNCBF_list.append(ItemKNNCBFRecommender(URM_train=URM_train_list[index], ICM_train=ICM))
            recommender_ItemKNNCBF_list[index].fit(topK=topK, shrink=shrink)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCBF_list)

        resultsToPrint = [recommender_ItemKNNCBF_list[0].RECOMMENDER_NAME, topK, shrink, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']

    recommender_RP3beta = ItemKNNCBFRecommender(URM_train_init, ICM)
    recommender_RP3beta.fit(shrink=shrink, topK=topK)

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
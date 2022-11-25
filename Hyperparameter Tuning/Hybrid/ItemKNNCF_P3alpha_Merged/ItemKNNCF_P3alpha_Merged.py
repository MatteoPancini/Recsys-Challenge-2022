if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import createURM
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
    import optuna as op
    import json
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createURM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header
    """
    header = ['recommender', 'alpha', 'MAP']

    partialsFile = 'partials_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
    """

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)
    """
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

        recommender_P3alpha_list = []
        recommender_ItemKNNCF_list = []
        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)

        for index in range(len(URM_train_list)):
            recommender_P3alpha_list.append(P3alphaRecommender(URM_train_list[index], verbose=False))
            recommender_P3alpha_list[index].fit(alpha=0.8561168568686058, topK=218)

            recommender_ItemKNNCF_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
            recommender_ItemKNNCF_list[index].fit(shrink=186.40047767299245, topK=157)

            new_similarity = (1 - alpha) * recommender_ItemKNNCF_list[index].W_sparse + alpha * recommender_P3alpha_list[index].W_sparse

            recommender_Hybrid_list.append(ItemKNNCustomSimilarityRecommender(URM_train_list[index]))
            recommender_Hybrid_list[index].fit(new_similarity)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_Hybrid_list[0].RECOMMENDER_NAME, alpha, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    """
    recommender_P3alpha = P3alphaRecommender(URM_train_init, verbose=False)
    recommender_P3alpha.fit(alpha=0.8561168568686058, topK=218)

    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train_init, verbose=False)
    recommender_ItemKNNCF.fit(shrink=186.40047767299245, topK=157)

    new_similarity = (1 - 0.7172534192416513) * recommender_ItemKNNCF.W_sparse + 0.7172534192416513 * recommender_P3alpha.W_sparse

    recommender_Hybrid = ItemKNNCustomSimilarityRecommender(URM_train_init)
    recommender_Hybrid.fit(new_similarity)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_Hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_Hybrid.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        #json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
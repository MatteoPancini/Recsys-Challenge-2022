if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    import optuna as op
    import json
    import csv
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'shrink', 'topk', 'similarity', 'normalization', 'MAP']

    partialsFile = 'ItemKNNCBF_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_1K_BinURMTrain()
    URM_validation_list = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    ICM = createSmallICM()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)
    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_ItemKNNCF_list = []

        topK = trial.suggest_int("topK", 100, 2000)
        shrink = trial.suggest_int("shrink", 10, 1000)
        similarity = trial.suggest_categorical('similarity', ['cosine', 'dice', 'jaccard', 'asym', 'rp3beta'])
        normalization = trial.suggest_categorical("normalization", ["tfidf", "bm25plus", "bm25"])

        for index in range(len(URM_train_list)):
            recommender_ItemKNNCF_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
            recommender_ItemKNNCF_list[index].fit(ICM=ICM, shrink=shrink, topK=topK, similarity=similarity,
                                                  normalization=normalization)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF_list)

        resultsToPrint = [recommender_ItemKNNCF_list[0].RECOMMENDER_NAME, shrink, topK, similarity, normalization,  sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize', sampler=RandomSampler())
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    shrink = study.best_params['shrink']
    similarity = study.best_params['similarity']
    normalization = study.best_params['normalization']

    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train_init, verbose=False)
    recommender_ItemKNNCF.fit(ICM=ICM, shrink=shrink, topK=topK, similarity=similarity, normalization=normalization)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_ItemKNNCF)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/ItemKNNCBF_" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)

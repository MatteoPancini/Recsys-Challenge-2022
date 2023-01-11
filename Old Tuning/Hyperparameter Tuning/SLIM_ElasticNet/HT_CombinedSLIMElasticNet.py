if __name__ == "__main__":

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op
    import scipy.sparse as sps
    import similaripy
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM & ICM
    URM = createURM()

    ICM = createSmallICM()

    ICM = similaripy.normalization.bm25plus(ICM.copy())

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'topK', 'alpha', 'l1_ration', 'ICMweight', 'MAP']

    partialsFile = 'Combinedpartials_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)


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

    CombinedURMs = []
    for URM in URM_train_list:
        CombinedURMs.append(sps.vstack([URM, ICM.T]))

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_SlimElasticnet_list = []

        topK = trial.suggest_int("topK", 100, 500)
        alpha = trial.suggest_float("alpha", 1e-5, 0.1)
        l1_ratio = trial.suggest_float("l1_ratio", 1e-5, 0.1)
        ICMweight = trial.suggest_int("icm_weight", 1, 100)

        for index in range(len(URM_train_list)):

            CombinedURM = sps.vstack([URM_train_list[index], (ICM * ICMweight).T])
            recommender_SlimElasticnet_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(CombinedURM))
            recommender_SlimElasticnet_list[index].fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_SlimElasticnet_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_SlimElasticnet_list[0].RECOMMENDER_NAME, topK, alpha, l1_ratio, ICMweight,
                          sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params["topK"]
    alpha = study.best_params["alpha"]
    l1_ratio = study.best_params["l1_ratio"]
    ICMweight = study.best_params("icm_weight")

    CombinedURM = sps.vstack([URM_train_init, (ICM * ICMweight).T])

    recommender_SlimElasticnet = MultiThreadSLIM_SLIMElasticNetRecommender(CombinedURM)
    recommender_SlimElasticnet.fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SlimElasticnet)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/Combined" + recommender_SlimElasticnet.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
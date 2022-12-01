if __name__ == "__main__":

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.crossKValidator import CrossKValidator
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

    cross_validator = CrossKValidator(URM_train_init, k=3)
    evaluator_validation, URM_train_list, URM_validation_list = cross_validator.create_combined_k_evaluators(ICM)

    MAP_results_list = []

    # ---------------------------------------------------------------------------------------------------------
    # URM normalization

    for URM in URM_train_list:
        URM = similaripy.normalization.bm25plus(URM)


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_SlimElasticnet_list = []

        topK = trial.suggest_int("topK", 100, 500)
        alpha = trial.suggest_float("alpha", 1e-5, 0.1)
        l1_ratio = trial.suggest_float("l1_ratio", 1e-5, 0.1)

        for index in range(len(URM_train_list)):

            recommender_SlimElasticnet_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_list[index]))
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
if __name__ == "__main__":

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import *
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op
    import csv
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'l1_ratio', 'TopK', 'MAP']

    partialsFile = 'SlimElasticNet' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_URMTrainInit()
    URM_train_list = load_K_URMTrain()
    URM_validation_list = load_K_URMValid()
    URM_test = load_URMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_SlimElasticnet_list = []

        topK = trial.suggest_int("topK", 10, 400)
        alpha = trial.suggest_float("alpha", 0, 1)
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)

        for index in range(len(URM_train_list)):

            recommender_SlimElasticnet_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_list[index]))
            recommender_SlimElasticnet_list[index].fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_SlimElasticnet_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_SlimElasticnet_list[0].RECOMMENDER_NAME, alpha, l1_ratio, topK, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params["topK"]
    alpha = study.best_params["alpha"]
    l1_ratio = study.best_params["l1_ratio"]

    recommender_SlimElasticnet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_init)
    recommender_SlimElasticnet.fit(alpha=alpha, l1_ratio=l1_ratio, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SlimElasticnet)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_SlimElasticnet.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
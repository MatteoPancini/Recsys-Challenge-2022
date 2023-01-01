if __name__ == "__main__":

    from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import *
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op
    import numpy as np
    import csv
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'topK', 'epochs', 'lambda_i', 'lambda_j', 'MAP']

    partialsFile = 'SlimBPR_' + datetime.now().strftime('%b%d_%H-%M-%S')

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

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_SlimElasticnet_list = []

        topK = trial.suggest_int("topK", 5, 1000)
        epochs = trial.suggest_int("epochs", 10, 100)
        lambda_i = trial.suggest_float("lambda_i", 1e-5, 1e-2)
        lambda_j = trial.suggest_float("lambda_j", 1e-5, 1e-2)

        for index in range(len(URM_train_list)):
            recommender_SlimElasticnet_list.append(SLIM_BPR_Python(URM_train_list[index]))
            recommender_SlimElasticnet_list[index].fit(epochs=epochs, lambda_i=lambda_i, lambda_j=lambda_j, topK=topK)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_SlimElasticnet_list)

        resultsToPrint = [recommender_SlimElasticnet_list[0].RECOMMENDER_NAME, topK, epochs, lambda_i, lambda_j,
                          sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)

    study = op.create_study(direction='maximize', sampler=RandomSampler())
    study.optimize(objective, n_trials=50)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    topK = study.best_params['topK']
    epochs = study.best_params["epochs"]
    lambda_i = study.best_params["lambda_i"]
    lambda_j = study.best_params["lambda_j"]

    recommender_SlimBPR = SLIM_BPR_Python(URM_train_init)
    recommender_SlimBPR.fit(epochs=epochs, lambda_i=lambda_i, lambda_j=lambda_j, topK=topK)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SlimBPR)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_SlimBPR.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
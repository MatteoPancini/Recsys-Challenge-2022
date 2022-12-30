if __name__ == "__main__":

    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    import csv
    from datetime import datetime
    import optuna as op
    import json
    from optuna.samplers import RandomSampler

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'MAP']

    partialsFile = 'Slim-Ials_' + datetime.now().strftime('%b%d_%H-%M-%S')

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

    recSlim = SLIMElasticNetRecommender(URM_train_list[0])
    recSlim.load_model(folder_path='../../Models/', file_name='1kSLIMElasticNetRecommender.zip')

    recIals = ImplicitALSRecommender(URM_train_list[0])
    recIals.load_model(folder_path='../../Models/', file_name='1kImplicitALSRecommender.zip')


    def objective(trial):

        recommender_hybrid_list = []

        alpha = trial.suggest_float("alpha", 0.3, 1)
        beta = trial.suggest_float("beta", 0, 0.1)

        recommender_hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[0], Recommender_1=recSlim, Recommender_2=recIals))
        recommender_hybrid_list[0].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)

        resultsToPrint = ["Slim-Ials", alpha, beta, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    rec1 = SLIMElasticNetRecommender(URM_train_init)
    rec1.load_model(folder_path='../../Models/', file_name='initSLIMElasticNetRecommender.zip')

    rec2 = ImplicitALSRecommender(URM_train_init)
    rec2.load_model(folder_path='../../Models/', file_name='initImplicitALSRecommender.zip')

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec1, Recommender_2=rec2)
    recommender_hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "Slim-Ials" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
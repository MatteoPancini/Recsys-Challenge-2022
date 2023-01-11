if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    import optuna as op
    import json
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'MAP']

    partialsFile = 'RP3beta_ItemKNNCF_Binary_' + datetime.now().strftime('%b%d_%H-%M-%S')

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
    ICM = createSmallICM()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Fit recommenders

    recommender_ItemKNNCF_list = []
    recommender_RP3beta_list = []

    for index in range(len(URM_train_list)):
        recommender_ItemKNNCF_list.append(ItemKNNCFRecommender(URM_train_list[index]))
        recommender_ItemKNNCF_list[index].fit(ICM=ICM, shrink=189, topK=352, similarity='asym', normalization='bm25')

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index]))
        recommender_RP3beta_list[index].fit(alpha=0.7374740262490058, beta=0.31114169352550514, topK=100)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 1)

        for index in range(len(URM_train_list)):

            recommender_Hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_list[index], Recommender_1=recommender_RP3beta_list[index], Recommender_2=recommender_ItemKNNCF_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)

        resultsToPrint = [recommender_Hybrid_list[0].RECOMMENDER_NAME, alpha, beta, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']

    recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train_init)
    recommender_ItemKNNCF.fit(ICM=ICM, shrink=189, topK=352, similarity='asym', normalization='bm25')

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=0.7374740262490058, beta=0.31114169352550514, topK=100)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_init, Recommender_1=recommender_RP3beta, Recommender_2=recommender_ItemKNNCF)
    recommender_Hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_Hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "RP3beta_ItemKNNCF_Binary_Recommender" + "_logs_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
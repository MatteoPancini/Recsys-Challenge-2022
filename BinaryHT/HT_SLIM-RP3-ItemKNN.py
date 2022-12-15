if __name__ == "__main__":

    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Utils.recsys2022DataReader import createURMBinary
    from Utils.recsys2022DataReader import createSmallICM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    import json
    from datetime import datetime
    import optuna as op
    import csv
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridThreeRecommenderThreeVariables


    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createURMBinary()
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'gamma',  'TopK', 'MAP']

    partialsFile = '3Hybrid' + datetime.now().strftime('%b%d_%H-%M-%S')

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

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_SlimElasticnet_list = []
    recommender_RP3beta_list = []
    recommender_ItemKNN_list = []


    for index in range(len(URM_train_list)):
        recommender_SlimElasticnet_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_list[index]))
        recommender_SlimElasticnet_list[index].fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train_list[index]))
        recommender_RP3beta_list[index].fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

        recommender_ItemKNN_list.append(ItemKNNCFRecommender(URM_train_list[index]))
        recommender_ItemKNN_list[index].fit(ICM, shrink=1665.2431108249625, topK=3228, similarity='dice', normalization='bm25')

    def objective(trial):

        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 1)
        gamma = trial.suggest_float("gamma", 0, 1)

        for index in range(len(URM_train_list)):
            recommender_Hybrid_list.append(LinearHybridThreeRecommenderThreeVariables(URM_train=URM_train_list[index],
                                                                                  Recommender_1=
                                                                                  recommender_SlimElasticnet_list[
                                                                                      index], Recommender_2=
                                                                                  recommender_RP3beta_list[index],
                                                                                      Recommender_3=recommender_ItemKNN_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha, beta=beta, gamma=gamma)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = ['SLIM+RP3+ItemKNN', alpha, beta, gamma,
                          sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)



    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params["alpha"]
    beta = study.best_params["beta"]
    gamma = study.best_params["gamma"]

    Slim = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_init)
    Slim.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

    RP3 = RP3betaRecommender(URM_train_init)
    RP3.fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

    ItemKNN = ItemKNNCFRecommender(URM_train_init)
    ItemKNN.fit(ICM, shrink=1665.2431108249625, topK=3228, similarity='dice', normalization='bm25')

    hybrid = LinearHybridThreeRecommenderThreeVariables(URM_train_init, Slim, RP3, ItemKNN)
    hybrid.fit(alpha=alpha, beta=beta, gamma=gamma)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/Hybrid_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
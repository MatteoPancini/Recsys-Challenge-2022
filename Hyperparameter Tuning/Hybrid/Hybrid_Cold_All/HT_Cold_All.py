if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables, LinearHybridTwoRecommenderOneVariable
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderOneVariableForCold
    import optuna as op
    import json
    import csv

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM

    URM = createURM()
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'MAP']

    partialsFile = 'RP3beta_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    URM_train_list = []
    URM_validation_list = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_init, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    MAP_results_list = []
    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    #Create hybrid Cold Users

    recommender_ItemKNN_Cold_list = []
    recommender_RP3beta_Cold_list = []
    recommender_hybrid_Cold_list = []

    for index in range(len(URM_train_list)):
        recommender_RP3beta_Cold_list.append(RP3betaRecommender(URM_train_list[index],verbose=False))
        recommender_RP3beta_Cold_list[index].fit(alpha=0.6627101454340679, beta=0.2350020032542621, topK=250)

        recommender_ItemKNN_Cold_list.append(ItemKNNCFRecommender(URM_train_list[index], verbose=False))
        recommender_ItemKNN_Cold_list[index].fit(ICM=ICM, topK=5893, shrink=50, similarity='rp3beta', normalization='tfidf')

        recommender_hybrid_Cold_list.append(LinearHybridTwoRecommenderOneVariable(URM_train=URM_train_list[index], Recommender_1=recommender_RP3beta_Cold_list[index],
                                                           Recommender_2=recommender_ItemKNN_Cold_list[index]))
        recommender_hybrid_Cold_list[index].fit(alpha=0.2584478495159924)


    #Create hybrid All Users

    recommender_RP3beta_All_list = []
    recommender_SlimElasticnet_All_list = []
    recommender_hybrid_All_list = []

    for index in range(len(URM_train_list)):
        recommender_SlimElasticnet_All_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_list[index]))
        recommender_SlimElasticnet_All_list[index].fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

        recommender_RP3beta_All_list.append(RP3betaRecommender(URM_train_list[index]))
        recommender_RP3beta_All_list[index].fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

        recommender_hybrid_All_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_list[index], Recommender_1= recommender_SlimElasticnet_All_list[index], Recommender_2=recommender_RP3beta_All_list[index]))
        recommender_hybrid_All_list[index].fit(alpha=0.18228980979705656, beta=0.5426630600143958)

    #Hybrid of All with Cold

    recommender_hybrid_list = []

    def objective(trial):

        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)

        for index in range(len(URM_train_list)):

            recommender_Hybrid_list.append(LinearHybridTwoRecommenderOneVariableForCold(URM_train=URM_train_list[index], Recommender_All=recommender_hybrid_All_list[index], Recommender_Cold=recommender_hybrid_Cold_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = ["Hybrid_Cold_All", alpha, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP
    """
    alpha = study.best_params['alpha']

    recommender_Slim_Elasticnet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_init)
    recommender_Slim_Elasticnet.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_init, Recommender_1=recommender_Slim_Elasticnet, Recommender_2=recommender_RP3beta)
    recommender_Hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_Hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_Hybrid.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)"""
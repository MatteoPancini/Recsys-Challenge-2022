# ---------------------------------------------------------------------------------------------------------
    ####### SPECIALIZED COLD + SLIM
# ---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.Hybrid.HandMade.HybridAll import HybridAll
    from Recommenders.Hybrid.LinearHybridRecommender import *
    import optuna as op
    import json
    import csv


    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'MAP']

    partialsFile = 'partials_' + datetime.now().strftime('%b%d_%H-%M-%S')

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

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_Group1_list = []
    recommender_SLIM_list = []

    for index in range(len(URM_train_list)):
        recommender_Group1_list.append(HybridAll(URM_train_list[index], ICM))
        recommender_Group1_list[index].fit()

        recommender_SLIM_list.append(MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_list[index]))
        recommender_SLIM_list[index].fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)


    def objective(trial):

        recommender_Hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 1)

        for index in range(len(URM_train_list)):
            recommender_Hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_list[index], Recommender_1=recommender_Group1_list[index], Recommender_2=recommender_SLIM_list[index]))
            recommender_Hybrid_list[index].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_Hybrid_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = ['Hybrid1+SLIM', alpha, beta, sum(MAP_result) / len(MAP_result)]

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

    rec1 = HybridAll(URM_train_init, ICM)
    rec1.fit()

    rec2 = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train_init)
    rec2.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_init, Recommender_1=rec1, Recommender_2=rec2)
    hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "Hybrid1+SLIM" + "_logs_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
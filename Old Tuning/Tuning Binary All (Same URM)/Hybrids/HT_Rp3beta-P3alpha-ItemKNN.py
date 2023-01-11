if __name__ == "__main__":

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    import csv
    from datetime import datetime
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'alpha', 'beta', 'MAP']

    partialsFile = 'RP3betaP3alpha-ItemKNN_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_3K_BinURMTrain()
    URM_validation_list = load_3K_BinURMValid()
    URM_test = load_BinURMTest()

    ICM = createSmallICM()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model


    recommender_RP3beta_list = []
    recommender_P3alpha_list = []
    recommender_ItemKNN_list = []

    recommender_hybrid1_list = []

    for i in range(len(URM_train_list)):

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train=URM_train_list[i]))
        recommender_RP3beta_list[i].fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

        recommender_P3alpha_list.append(P3alphaRecommender(URM_train=URM_train_list[i]))
        recommender_P3alpha_list[i].fit(topK=116, alpha=0.8763131065621229)

        recommender_hybrid1_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[i], Recommender_1=recommender_RP3beta_list[i], Recommender_2=recommender_P3alpha_list[i] ))
        recommender_hybrid1_list[i].fit(alpha=0.5042061133754471, beta=0.1229236356527148)


        recommender_ItemKNN_list.append(ItemKNNCFRecommender(URM_train_list[i], verbose=False))
        recommender_ItemKNN_list[i].fit(ICM=ICM, shrink=919, topK=584, similarity='dice', normalization='bm25')


    def objective(trial):

        recommender_hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 1)

        for i in range(len(URM_train_list)):
            recommender_hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[i], Recommender_1=recommender_hybrid1_list[i], Recommender_2=recommender_ItemKNN_list[i]))
            recommender_hybrid_list[i].fit(alpha=alpha, beta=beta)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)

        resultsToPrint = ["RP3betaP3alpha-ItemKNN", alpha, beta, sum(MAP_result) / len(MAP_result)]

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

    rec01 = RP3betaRecommender(URM_train_init)
    rec01.fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

    rec02 = P3alphaRecommender(URM_train_init)
    rec02.fit(topK=116, alpha=0.8763131065621229)

    rec1 = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec01, Recommender_2=rec02)
    rec1.fit(alpha=0.5042061133754471, beta=0.1229236356527148)

    rec2 = ItemKNNCFRecommender(URM_train_init)
    rec2.fit(ICM=ICM, shrink=919, topK=584, similarity='dice', normalization='bm25')

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec1, Recommender_2=rec2)
    recommender_hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "RP3betaP3alpha-ItemKNN" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
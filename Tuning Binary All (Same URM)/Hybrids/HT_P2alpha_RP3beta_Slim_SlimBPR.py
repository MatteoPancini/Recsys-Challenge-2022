if __name__ == "__main__":

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables, LinearHybridTwoRecommenderOneVariable
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
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

    header = ['recommender', 'alpha', 'MAP']

    partialsFile = 'RP3beta-P3alpha-Slim-BPR_' + datetime.now().strftime('%b%d_%H-%M-%S')

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

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model


    recommender_RP3beta_list = []
    recommender_P3alpha_list = []
    recommender_Slim_list = []
    recommender_BPR_list = []

    recommender_hybrid1_list = []
    recommender_hybrid2_list = []

    for i in range(len(URM_train_list)):

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train=URM_train_list[i]))
        recommender_RP3beta_list[i].fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

        recommender_P3alpha_list.append(P3alphaRecommender(URM_train=URM_train_list[i]))
        recommender_P3alpha_list[i].fit(topK=116, alpha=0.8763131065621229)

        recommender_hybrid1_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train_list[i], Recommender_1=recommender_RP3beta_list[i], Recommender_2=recommender_P3alpha_list[i] ))
        recommender_hybrid1_list[i].fit(alpha=0.5042061133754471, beta=0.1229236356527148)

        recommender_Slim_list.append(SLIMElasticNetRecommender(URM_train_list[i]))
        recommender_Slim_list[i].fit(topK=241, alpha=0.0031642653228324906, l1_ratio=0.009828283497311959)

        recommender_hybrid2_list.append(LinearHybridTwoRecommenderOneVariable(URM_train=URM_train_list[i], Recommender_1=recommender_hybrid1_list[i], Recommender_2=recommender_Slim_list[i]))
        recommender_hybrid2_list[i].fit(alpha=0.47677749252644375)

        recommender_BPR_list.append(SLIM_BPR_Python(URM_train=URM_train_list[i]))
        recommender_BPR_list[i].fit(topK=44, epochs=149, lambda_i=0.0023822524202104594, lambda_j=0.004375104840716731)


    def objective(trial):

        recommender_hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 1)

        for i in range(len(URM_train_list)):
            recommender_hybrid_list.append(LinearHybridTwoRecommenderOneVariable(URM_train_list[i], Recommender_1=recommender_hybrid2_list[i], Recommender_2=recommender_BPR_list[i]))
            recommender_hybrid_list[i].fit(alpha=alpha)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)

        resultsToPrint = ["Slim-Ials", alpha, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']

    rec01 = RP3betaRecommender(URM_train_init)
    rec01.fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

    rec02 = P3alphaRecommender(URM_train_init)
    rec02.fit(topK=116, alpha=0.8763131065621229)

    rec1 = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec01, Recommender_2=rec02)
    rec1.fit(alpha=0.5042061133754471, beta=0.1229236356527148)

    rec03 = SLIMElasticNetRecommender(URM_train_init)
    rec03.fit(topK=241, alpha=0.0031642653228324906, l1_ratio=0.009828283497311959)

    recommender_hybrid1 = LinearHybridTwoRecommenderOneVariable(URM_train_init, Recommender_1=rec1, Recommender_2=rec03)
    recommender_hybrid1.fit(alpha=0.47677749252644375)

    rec04 = SLIM_BPR_Python(URM_train=URM_train_init)
    rec04.fit(topK=44, epochs=149, lambda_i=0.0023822524202104594, lambda_j=0.004375104840716731)

    recommender_hybrid2 = LinearHybridTwoRecommenderOneVariable(URM_train=URM_train_init, Recommender_1=recommender_hybrid1, Recommender_2=rec04)
    recommender_hybrid2.fit(alpha=alpha)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid2)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "RP3betaP3alpha-Slim" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
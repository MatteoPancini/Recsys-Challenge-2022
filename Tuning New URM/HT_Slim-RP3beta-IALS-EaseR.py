if __name__ == "__main__":

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.Hybrid.Best_SlimRp3Beta import BestSlimRP3Beta
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridThreeRecommenderThreeVariables
    from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    from datetime import datetime
    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs

    URM_train_init = load_FinalURMTrainInit()
    URM_train_list = load_1K_FinalURMTrain()
    URM_validation_list = load_1K_FinalURMValid()
    URM_test = load_FinalURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_SlimRP3beta_list = []
    recommender_IALS_list = []
    recommender_EaseR_list = []

    for i in range(1):
        recommender_SlimRP3beta_list.append(BestSlimRP3Beta(URM_train_list[i]))
        recommender_SlimRP3beta_list[i].fit()

        recommender_IALS_list.append(ImplicitALSRecommender(URM_train=URM_train_list[i]))
        recommender_IALS_list[i].fit(factors=97, alpha=6, iterations=59, regularization=0.004070427647981844)

        recommender_EaseR_list.append(EASE_R_Recommender(URM_train_list[i]))
        recommender_EaseR_list[i].fit(topK=265, l2_norm=0.09561632250768182)



    def objective(trial):

        recommender_hybrid_list = []
        alpha = trial.suggest_float("alpha", 0, 1)
        beta = trial.suggest_float("beta", 0, 0.1)
        gamma = trial.suggest_float("gamma", 0, 0.1)

        for i in range(1):
            recommender_hybrid_list.append(LinearHybridThreeRecommenderThreeVariables(URM_train_list[i],
                                                                                  Recommender_1=recommender_SlimRP3beta_list[i],
                                                                                  Recommender_2=recommender_IALS_list[i],
                                                                                      Recommender_3=recommender_EaseR_list[i]))
            recommender_hybrid_list[i].fit(alpha=alpha, beta=beta, gamma=gamma)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params['alpha']
    beta = study.best_params['beta']
    gamma = study.best_params['gamma']


    rec1 = BestSlimRP3Beta(URM_train_init)
    rec1.fit()

    rec2 = ImplicitALSRecommender(URM_train_init)
    rec2.fit(factors=97, alpha=6, iterations=59, regularization=0.004070427647981844)

    rec3 = EASE_R_Recommender(URM_train_init)
    rec3.fit(topK=265, l2_norm=0.09561632250768182)

    recommender_hybrid = LinearHybridThreeRecommenderThreeVariables(URM_train_init,
                                                                    Recommender_1=rec1,
                                                                    Recommender_2=rec2,
                                                                    Recommender_3=rec3)
    recommender_hybrid.fit(alpha=alpha, beta=beta, gamma=gamma)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "Slim-RP3beta-IALS-EaseR" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
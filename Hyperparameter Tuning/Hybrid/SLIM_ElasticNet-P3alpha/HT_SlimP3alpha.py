if __name__ == "__main__":
    import json
    from datetime import datetime
    import optuna as op
    from optuna.samplers import TPESampler
    import scipy.sparse as sp
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Utils.recsys2022DataReader import *

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator
    URM_train_list = []
    URM_validation_list = []

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    # ----------------------------------------------------------------------------------------------------------
    # Defining recommenders and fitting them with k-splits
    recommender_P3alpha_list = []
    recommender_SLIMElasticNet_list = []

    for i in range(k):
        # Recommender 1 - P3alpha
        recommender_P3alpha = P3alphaRecommender(URM_train[i])
        recommender_P3alpha.fit(topK=424, alpha=0.8779015738944255)
        recommender_SLIMElasticNet_list.append(recommender_P3alpha)

        # Recommender 2 - SLIM Elastic Net
        recommender_SLIMElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train[i])
        recommender_SLIMElasticNet.fit(alpha=0.06747829810332745, l1_ratio=0.0005493724398243842, topK=362)
        recommender_SLIMElasticNet_list.append(recommender_SLIMElasticNet)


    def objective(trial):

        MAP_List_Fold = []

        topK = trial.suggest_int("topK", 250, 600)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)

        for index in range(len(URM_train_list)):
            recommender_P3alpha = P3alphaRecommender(URM_train_list[index], verbose=False)

            recommender_P3alpha_list.append(recommender_P3alpha)

            recommender_P3alpha.fit(alpha=alpha, topK=topK)

            evaluator_validation = EvaluatorHoldout(URM_validation_list[index], cutoff_list=[10])

            result_dict, _ = evaluator_validation.evaluateRecommender(recommender_P3alpha)

            MAP_List_Fold.append(result_dict.iloc[0]["MAP"])

        return sum(MAP_List_Fold) / len(MAP_List_Fold)


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))
    study.optimize(objective, n_trials=10)
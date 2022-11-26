if __name__ == "__main__":
    import json
    from datetime import datetime
    import optuna as op
    from optuna.samplers import TPESampler
    import scipy.sparse as sp
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.recsys2022DataReader import *

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator
    URM_train_list = []
    URM_validation_list = []

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85)

    """
    kfold = 1

    for k in range(kfold):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.85)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)
    
    """


    # ----------------------------------------------------------------------------------------------------------
    # Defining recommenders and fitting them with k-splits
    """
    recommender_P3alpha_list = []
    recommender_SLIMBPR_list = []

    for i in range(kfold):
        # Recommender 1 - P3alpha
        recommender_P3alpha = P3alphaRecommender(URM_train[i])
        recommender_P3alpha.fit(topK=424, alpha=0.8779015738944255)
        recommender_P3alpha_list.append(recommender_P3alpha)

        # Recommender 2 - SLIM BPR
        recommender_SLIMBPR = SLIM_BPR_Python(URM_train[i])
        recommender_SLIMBPR.fit(topK=45, lambda_j=1e-05, lambda_i=1e-05, epochs=75)
        recommender_SLIMBPR_list.append(recommender_SLIMBPR)

    hybridRecommender_list = []
    """
    recommender_P3alpha = P3alphaRecommender(URM_train)
    recommender_P3alpha.fit(topK=424, alpha=0.8779015738944255)
    recommender_SLIMBPR = SLIM_BPR_Python(URM_train)
    recommender_SLIMBPR.fit(topK=45, lambda_j=1e-05, lambda_i=1e-05, epochs=75)

    hybridRecommender = LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train,
                                                               Recommender_1=recommender_P3alpha,
                                                               Recommender_2=recommender_SLIMBPR)

    def objective(trial):

        #MAP_List_Fold = []

        alpha = trial.suggest_float("alpha", 0, 1)

        hybridRecommender.fit(alpha=alpha)
        """

        for index in range(kfold):
            hybridRecommender = LinearHybridTwoRecommender(URM_train=URM_train_list[index], Recommender_1=recommender_SLIMBPR_list[index],
                                                           Recommender_2=recommender_P3alpha_list[index])

            hybridRecommender.fit(alpha=alpha)

            hybridRecommender_list.append(hybridRecommender)

            evaluator_validation = EvaluatorHoldout(URM_validation_list[index], cutoff_list=[10])

            result_dict, _ = evaluator_validation.evaluateRecommender(hybridRecommender)

            MAP_List_Fold.append(result_dict.iloc[0]["MAP"])
            """
        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

        result_dict, _ = evaluator_validation.evaluateRecommender(hybridRecommender)

        #return sum(MAP_List_Fold) / len(MAP_List_Fold)
        return result_dict.iloc[0]["MAP"]


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))
    study.optimize(objective, n_trials=3)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    alpha = study.best_params['alpha']
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(hybridRecommender)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + hybridRecommender.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)

"""
    alpha = study.best_params['alpha']

    hybridRecommender_list[0].fit(alpha=alpha)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_SLIMBPR_list[0])
    

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + hybridRecommender_list[0].RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
"""
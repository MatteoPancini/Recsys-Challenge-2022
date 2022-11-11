if __name__ == '__main__':

    from optuna.samplers import TPESampler
    from Evaluation.Evaluator import EvaluatorHoldout
    import pandas as pd
    from datetime import datetime
    from Utils.recsys2022DataReader import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

    import optuna as op
    import json

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    URM = createBumpURM()

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_list = []
    URM_validation_list = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    # evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # results = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommender_P3alpha_list = []


    # for index in range(len(URM_train_list)):
    #    recommenders.append(
    #        IALSRecommender_implicit(URM_train=URM_train_list[index])
    #    )

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

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    topK = study.best_params['topK']
    alpha = study.best_params['alpha']

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

    recommender_P3alpha = P3alphaRecommender(URM_train, verbose=False)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender_P3alpha.fit(alpha=alpha, topK=topK)

    result_dict, _ = evaluator_test.evaluateRecommender(recommender_P3alpha)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    log = open("logs/" + recommender_P3alpha.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime('%b%d_%H-%M-%S/') + ".json", 'w')
    log.write(json.dump(study.best_params, indent=4))
    with open(log, 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)
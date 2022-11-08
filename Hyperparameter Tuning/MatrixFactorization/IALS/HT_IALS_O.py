if __name__ == '__main__':

    from optuna.samplers import TPESampler
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator import EvaluatorHoldout
    import pandas as pd
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from Utils.createURM import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    #from Recommenders.MatrixFactorization.IALS_implicit_Recommender import IALSRecommender_implicit
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

    import optuna as op
    import json


    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    dataset = pd.read_csv('/Users/matteopancini/PycharmProjects/recsys-challenge-2022-Pancini-Vitali/Input/interactions_and_impressions.csv')
    URM = createBumpURM(dataset)

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_list = []
    URM_validation_list = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
        URM_train_list.append(URM_train)
        URM_validation_list.append(URM_validation)

    #evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    #results = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommenders_IALS_list = []

    #for index in range(len(URM_train_list)):
    #    recommenders.append(
    #        IALSRecommender_implicit(URM_train=URM_train_list[index])
    #    )


    def objective(trial):

        #recommenders = []

        MAP_List_Fold = []
        
        n_factors = trial.suggest_int("n_factors", 300, 400)
        regularization = trial.suggest_float("regularization", 1e-6, 1e-1)
        alpha_val = trial.suggest_float("alpha_val", 10, 50)
        iterations = trial.suggest_int("iterations", 1, 100)
        
        for index in range(len(URM_train_list)):

            recommender_IALS = IALSRecommender(URM_train_list[index], verbose=False)

            #recommender_IALS.fit(n_factors=n_factors, regularization=regularization, alpha_val=alpha_val, iterations=3)
            recommender_IALS.fit(num_factors=n_factors, reg=regularization, alpha=alpha_val, epochs=3)

            #recommenders.append(IALSRecommender_implicit(URM_train_list[index], verbose=False))

            #recommenders[index].fit(n_factors=n_factors, regularization=regularization, alpha_val=alpha_val,
                                    #iterations=3)

            #recommenders[index].URM_train = URM_train_list[index]  # utile in caso di combine
            print("starting evaluation...")

            evaluator_validation = EvaluatorHoldout(URM_validation_list[index], cutoff_list=[10])
            
            result_dict, _ = evaluator_validation.evaluateRecommender(recommender_IALS)
            
            MAP_List_Fold.append(result_dict.iloc[0]["MAP"])
            

        print("starting evaluation...")
        #result = evaluator_validation.evaluateRecommender(recommenders)
        #print(result)

        #results.append(result)

        return sum(MAP_List_Fold) / len(MAP_List_Fold)


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))

    study.optimize(objective, n_trials=3)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    n_factors = study.best_params['n_factors']
    regularization = study.best_params['regularization']
    alpha_val = study.best_params['alpha_val']
    iterations = study.best_params['iterations']

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

    recommender_IALS = IALSRecommender(URM_train, verbose=False)
    #recommenders.append(IALSRecommender_implicit(URM_train_list[0], verbose=False))

    recommender_IALS.fit(num_factors=n_factors, reg=regularization, alpha=alpha_val, epochs=3)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommender_IALS.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)

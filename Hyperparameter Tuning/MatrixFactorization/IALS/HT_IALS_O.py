if __name__ == '__main__':

    from optuna.samplers import TPESampler
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    import pandas as pd
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    from Utils.createURM import createBumpURM
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Recommenders.MatrixFactorization.IALS_implicit_Recommender import IALSRecommender_implicit
    import optuna as op
    import json


    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    dataset = pd.read_csv('/Users/matteopancini/PycharmProjects/recsys-challenge-2022-Pancini-Vitali/Input/interactions_and_impressions.csv')
    URM = createBumpURM(dataset)

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URMs_train = []
    URMs_validation = []

    for k in range(1):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

    evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

    results = []

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    recommenders = []

    for index in range(len(URMs_train)):
        recommenders.append(
            IALSRecommender_implicit(URM_train=URMs_train[index])
        )


    def objective(trial):

        recommenders = []

        n_factors = trial.suggest_int("n_factors", 300, 400)
        regularization = trial.suggest_float("regularization", 1e-6, 1e-1)
        alpha_val = trial.suggest_float("alpha_val", 10, 50)
        iterations = trial.suggest_int("iterations", 1, 100)

        for index in range(len(URMs_train)):

            recommenders.append(IALSRecommender_implicit(URMs_train[index], verbose=False))

            recommenders[index].fit(n_factors=n_factors, regularization=regularization, alpha_val=alpha_val,
                                    iterations=3)

            recommenders[index].URM_train = URMs_train[index]  # utile in caso di combine
        print("starting evaluation...")
        result = evaluator_validation.evaluateRecommender(recommenders)
        print(result)

        results.append(result)

        return sum(result) / len(result)


    study = op.create_study(direction='maximize', sampler=TPESampler(multivariate=True))

    study.optimize(objective, n_trials=3)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    n_factors = study.best_params['facton_factorsrs']
    regularization = study.best_params['regularization']
    alpha_val = study.best_params['alpha_val']
    iterations = study.best_params['iterations']

    recommenders.append(IALSRecommender_implicit(URMs_train[0], verbose=False))

    recommenders[0].fit(n_factors=n_factors, regularization=regularization, alpha_val=alpha_val,
                                    iterations=30)

    result_dict = evaluator_validation.evaluateRecommender(recommenders[0])

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(parsed, json_file, indent=4)

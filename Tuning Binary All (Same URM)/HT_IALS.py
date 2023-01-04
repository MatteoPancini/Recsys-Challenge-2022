if __name__ == '__main__':
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    import json
    import optuna as op
    from datetime import datetime
    import csv
    from Utils.recsys2022DataReader import *

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'factors', 'alpha', 'regularization', 'iterations', 'MAP']

    partialsFile = 'IALS_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_1K_BinURMTrain()
    URM_validation_list = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation_list, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_IALS_list = []

        factors = trial.suggest_int("factors", 200, 400)
        alpha = trial.suggest_int("alpha", 1, 50)
        iterations = trial.suggest_int("iterations", 10, 100)
        regularization = trial.suggest_float("regularization", 1e-5, 1e-2)

        for index in range(len(URM_train_list)):

            recommender_IALS_list.append(ImplicitALSRecommender(URM_train_list[index]))
            recommender_IALS_list[index].fit(alpha=alpha, factors=factors,
                                             iterations=iterations, regularization=regularization)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_IALS_list)

        resultsToPrint = [recommender_IALS_list[0].RECOMMENDER_NAME, factors, alpha, regularization,
                          iterations, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)


    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    factors = study.best_params['factors']
    alpha = study.best_params["alpha"]
    iterations = study.best_params["iterations"]
    regularization = study.best_params["regularization"]

    recommender_IALS = ImplicitALSRecommender(URM_train_init)
    recommender_IALS.fit(alpha=alpha, factors=factors,
                                             iterations=iterations, regularization=regularization)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    with open("logs/" + recommender_IALS.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(result_dict, json_file, indent=4)
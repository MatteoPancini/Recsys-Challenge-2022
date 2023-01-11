if __name__ == '__main__':

    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    import json
    import optuna as op
    from datetime import datetime
    import csv
    import similaripy
    import scipy.sparse as sp
    from Utils.crossKValidator import CrossKValidator
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM + ICM
    URM = createURM()

    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommender', 'factors', 'alpha', 'regularization', 'iterations', 'MAP']

    partialsFile = 'partials_' + datetime.now().strftime('%b%d_%H-%M-%S')

    with open('partials/' + partialsFile + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # K-Fold Cross Validation + Preparing training, validation, test split and evaluator

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

    cross_validator = CrossKValidator(URM_train_init, k=3)
    evaluator_validation, URM_train_list, URM_validation_list = cross_validator.create_combined_k_evaluators(ICM)

    MAP_results_list = []


    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_IALS_list = []

        factors = trial.suggest_int("factors", 200, 600)
        alpha = trial.suggest_int("alpha", 10, 50)
        iterations = trial.suggest_int("iterations", 10, 100)
        regularization = 0.0001

        for index in range(len(URM_train_list)):

            recommender_IALS_list.append(ImplicitALSRecommender(URM_train_list[index]))
            recommender_IALS_list[index].fit(alpha=alpha, factors=factors,
                                             iterations=iterations, regularization=regularization)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_IALS_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = [recommender_IALS_list[0].RECOMMENDER_NAME, factors, alpha, regularization,
                          iterations, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    factors = study.best_params['factors']
    alpha = study.best_params["alpha"]
    iterations = study.best_params["iterations"]
    regularization = 0.001

    recommender_IALS = ImplicitALSRecommender(URM_train_init, verbose=False)
    recommender_IALS.fit(alpha=alpha, factors=factors,
                                             iterations=iterations, regularization=regularization)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_IALS)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    with open("logs/Combined" + recommender_IALS.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(result_dict, json_file, indent=4)
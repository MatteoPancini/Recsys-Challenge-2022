if __name__ == '__main__':

    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    import json
    import optuna as op
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

    from datetime import datetime
    import csv
    import scipy.sparse as sp
    from Utils.crossKValidator import CrossKValidator
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM + ICM
    URM = createURM()

    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV header

    header = ['recommenders', 'alpha', 'beta', 'MAP']

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
    # Base recommenders creation

    recommender_P3alpha_list = []
    recommender_RP3beta_list = []
    recommender_hybrid_list = []


    for i in range(len(URM_train_list)):

        recommender_P3alpha_list.append(P3alphaRecommender(URM_train=URM_train_list[i]))
        recommender_P3alpha_list[i].fit(topK=218, alpha=0.8561168568686058)

        recommender_RP3beta_list.append(RP3betaRecommender(URM_train=URM_train_list[i]))
        recommender_RP3beta_list[i].fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)



    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_hybrid_list = []

        alpha = trial.suggest_float("alpha", 0, 2)
        beta = trial.suggest_float("beta", 0, 2)


        for index in range(len(URM_train_list)):

            recommender_hybrid_list.append(LinearHybridTwoRecommenderTwoVariables(URM_train=URM_train_list[i],
                                                                                 Recommender_1=recommender_P3alpha_list[i],
                                                                                 Recommender_2=recommender_RP3beta_list[i]))
            recommender_hybrid_list[index].fit(alpha=alpha, beta=beta)


        MAP_result = evaluator_validation.evaluateRecommender(recommender_hybrid_list)
        MAP_results_list.append(MAP_result)

        resultsToPrint = ['P3alpha + RP3beta', alpha, beta, sum(MAP_result) / len(MAP_result)]

        with open('partials/' + partialsFile + '.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(resultsToPrint)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    alpha = study.best_params["alpha"]
    beta = study.best_params["beta"]

    recommender_P3alpha = P3alphaRecommender(URM_train_init)
    recommender_P3alpha.fit(topK=218, alpha=0.8561168568686058)

    recommender_RP3beta = RP3betaRecommender(URM_train_init)
    recommender_RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, recommender_P3alpha, recommender_RP3beta)
    recommender_hybrid.fit(alpha=alpha, beta=beta)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    with open("logs/" + 'P3alpha + RP3beta' + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(result_dict, json_file, indent=4)



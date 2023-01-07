if __name__ == '__main__':
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Recommenders.Neural.MultVAERecommender import MultVAERecommender
    import json
    import optuna as op
    from datetime import datetime
    from Utils.recsys2022DataReader import *


    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
    URM_train_init = load_BinURMTrainInit()
    URM_train = load_1K_BinURMTrain()[0]
    URM_validation = load_1K_BinURMValid()
    URM_test = load_BinURMTest()

    evaluator_validation = K_Fold_Evaluator_MAP(URM_validation, cutoff_list=[10], verbose=False)

    # ---------------------------------------------------------------------------------------------------------
    # Optuna hyperparameter model

    def objective(trial):

        recommender_MultVAE = []

        epochs = trial.suggest_int("epochs", 50, 100)
        learning_rate = trial.suggest_int("learning_rate", 1e-5, 1e-3)
        batch_size = trial.suggest_int("batch_size", 350, 850)
        dropout = trial.suggest_float("dropout", 0, 0.9)

        recommender_MultVAE.append(MultVAERecommender(URM_train, force_gpu=True))
        recommender_MultVAE[0].fit(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, dropout=dropout)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_MultVAE)


        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)


    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    factors = study.best_params['factors']
    alpha = study.best_params["alpha"]
    iterations = study.best_params["iterations"]
    regularization = study.best_params["regularization"]

    recommender_MultVAE = MultVAERecommender(URM_train_init)
    recommender_MultVAE.fit(alpha=alpha, factors=factors,
                                             iterations=iterations, regularization=regularization)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_MultVAE)


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    with open("logs/" + recommender_MultVAE.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(result_dict, json_file, indent=4)
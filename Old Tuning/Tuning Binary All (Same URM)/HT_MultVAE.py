if __name__ == '__main__':
    from Evaluation.Evaluator_IALS import EvaluatorHoldout
    from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
    import json
    import optuna as op
    from datetime import datetime
    from Utils.recsys2022DataReader import *
    from optuna.samplers import RandomSampler

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

        recommender_MultVAE_list = []

        epochs = trial.suggest_int("epochs", 10, 100)
        batch_size = trial.suggest_int("batch_size", 350, 850)
        dropout = trial.suggest_float("dropout", 0, 0.9)
        max_n_hidden_layers = trial.suggest_int("max_n_hidden_layers", 3, 5)
        encoding_size = trial.suggest_int("encoding_size", 50, 250)

        for index in range(len(URM_train_list)):
            recommender_MultVAE_list.append(MultVAERecommender_OptimizerMask(URM_train_list[index], force_gpu=True))
            recommender_MultVAE_list[index].fit(epochs=epochs, dropout=dropout, batch_size=batch_size,
                                                max_n_hidden_layers=max_n_hidden_layers,
                                                encoding_size=encoding_size)

        MAP_result = evaluator_validation.evaluateRecommender(recommender_MultVAE_list)

        return sum(MAP_result) / len(MAP_result)


    study = op.create_study(direction='maximize', sampler=RandomSampler())
    study.optimize(objective, n_trials=50)


    # ---------------------------------------------------------------------------------------------------------
    # Fitting and testing to get local MAP

    epochs = study.best_params['epochs']
    dropout = study.best_params["dropout"]
    batch_size = study.best_params["batch_size"]
    max_n_hidden_layers = study.best_params["max_n_hidden_layers"]
    encoding_size = study.best_params["encoding_size"]

    recommender_MultVAE = MultVAERecommender_OptimizerMask(URM_train_init)
    recommender_MultVAE.fit(epochs=epochs, dropout=dropout, batch_size=batch_size,
                                                max_n_hidden_layers=max_n_hidden_layers,
                                                encoding_size=encoding_size)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_MultVAE)


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    with open("logs/" + recommender_MultVAE.RECOMMENDER_NAME + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)
        json.dump(result_dict, json_file, indent=4)
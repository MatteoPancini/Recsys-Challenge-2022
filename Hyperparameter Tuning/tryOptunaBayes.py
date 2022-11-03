import pandas as pd
import numpy as np
import sklearn as sk
import optuna
from Utils.createURM import createURM
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

URM = createURM()
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

tuning_params = {
    "topK": (10, 500),
    "shrink": (0, 200)
}

recommender = ItemKNNCFRecommender(URM_train)
hyperparameters = {'target' : 0.0}


def objective(trial, topK, shrink):
    recommender.fit(topK=int(topK), shrink=shrink)
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    return result_dict[10]["MAP"]


if __name__ == "__main__":
    df = Utils.createURM()
    optimization_function = sk.partial(optimize, X=x, y=y)

    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_function, n_trials=15)
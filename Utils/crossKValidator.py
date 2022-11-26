from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

class CrossKValidator:
    def __init__(self, URM_train, k=1):
        self.URM_train = URM_train
        self.k = k

    def create_k_evaluators(self):

        URMs_train = []
        URMs_validation = []

        for k in range(3):
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(self.URM_train, train_percentage=0.85)
            URMs_train.append(URM_train)
            URMs_validation.append(URM_validation)

        evaluator = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

        return evaluator, URMs_train, URMs_validation

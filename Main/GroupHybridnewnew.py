if __name__ == "__main__":

    import scipy.sparse as sp
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.Hybrid.LastNewGroupHybridRecommender import GroupHybrid
    from Utils.writeSubmission import write_submission
    import json
    from datetime import datetime

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM + ICM

    URM = createURM()
    ICM = createSmallICM()

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator

    hybrid = GroupHybrid(URM, ICM)
    hybrid.fit()

    # ---------------------------------------------------------------------------------------------------------
    # Write submission

    write_submission(recommender=hybrid,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('newnewGroupHybrid'))
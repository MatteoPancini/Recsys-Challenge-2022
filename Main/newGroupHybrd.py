if __name__ == "__main__":

    import scipy.sparse as sp
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.Hybrid.newNewGroupHybridRecommender import GroupHybrid
    from Utils.writeSubmission import write_submission
    import json
    from datetime import datetime

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM + ICM

    URM = createURM()

    ICM = createSmallICM()

    #URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)



    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator

    hybrid = GroupHybrid(URM, ICM)
    hybrid.fit()


    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log
    '''
    
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(hybrid)

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("Hybrid_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(parsed, json_file, indent=4)
        
    '''

    # ---------------------------------------------------------------------------------------------------------
    # Write submission

    write_submission(recommender=hybrid,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('newGroupHybrid'))

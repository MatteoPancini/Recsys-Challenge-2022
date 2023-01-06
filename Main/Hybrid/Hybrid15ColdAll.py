if __name__ == '__main__':

    from Utils.recsys2022DataReader import *
    from Recommenders.Hybrid.HybridCold15WithAll import HybridCold15WithAll
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.writeSubmission import write_submission

    '''
    # Loading URM
    URM_train = load_BinURMTrainInit()
    URM_test = load_BinURMTest()

    recommender = InteractionsHybridRecommender(URM_train=URM_train)
    recommender.fit()

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)

    print(result_dict)
    '''
    URM = createURMBinary()
    recommender = HybridCold15WithAll(URM_train=URM)
    recommender.fit()

    # Write the submission file
    write_submission(recommender=recommender,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}2712_submission.csv'.format('Hybrid15All'))





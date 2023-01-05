if __name__ == '__main__':

    from Utils.recsys2022DataReader import *
    from Recommenders.Hybrid.HybridCold15WithAll import InteractionsHybridRecommender
    from Evaluation.Evaluator import EvaluatorHoldout


    # Loading URM
    URM_train = load_BinURMTrainInit()
    URM_test = load_BinURMTest()

    recommender = InteractionsHybridRecommender(URM_train=URM_train)
    recommender.fit()

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=False)
    result_dict, _ = evaluator_test.evaluateRecommender(recommender)

    print(result_dict)



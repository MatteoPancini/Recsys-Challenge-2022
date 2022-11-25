if __name__ == '__main__':

    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import createURM, createSmallICM
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Recommenders.Hybrid.ItemKNNCBF_ItemKNNCF_PipelinedRecommender import ItemKNNCBFItemKNNCFPipelinedRecommender

    URM = createURM()
    ICM = createSmallICM()

    URM_train_init, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = ItemKNNCBFItemKNNCFPipelinedRecommender(URM_train_init, ICM, topK_knncbf=236, shrink_knncbf=26.126055105423717)
    recommender.fit(topK_knncf=165, shrink_knncf=21.12990644177724)

    result, _ = evaluator_test.evaluateRecommender(recommender)
    print(result)

if __name__ == "__main__":

    from Utils.recsys2022DataReader import *
    import numpy as np
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

    # ---------------------------------------------------------------------------------------------------------
    # Loading URMs
    URM_train_init = load_BinURMTrainInit()
    URM_train_list = load_3K_BinURMTrain()
    URM = createURMBinary()

    factors = 97
    alpha = 8
    iterations = 24
    regularization = 0.0005620315371961919

    '''
    recTrainInit = ImplicitALSRecommender(URM_train_init)
    recTrainInit.fit(factors=factors, alpha=alpha, iterations=iterations, regularization=regularization)
    recTrainInit.save_model(folder_path='Models/', file_name='init' + recTrainInit.RECOMMENDER_NAME)

    recTrainList = ImplicitALSRecommender(URM_train_list[0])
    recTrainList.fit(factors=factors, alpha=alpha, iterations=iterations, regularization=regularization)
    recTrainInit.save_model(folder_path='Models/', file_name='1k' + recTrainInit.RECOMMENDER_NAME)

    recTrainInit = SLIMElasticNetRecommender(URM_train_init)
    recTrainInit.fit(topK=246, alpha=0.003032696689659577, l1_ratio=0.009813852653622251)
    recTrainInit.save_model(folder_path='Models/', file_name='init' + recTrainInit.RECOMMENDER_NAME)
    '''

    recAll = SLIMElasticNetRecommender(URM_train_list[1])
    recAll.fit(topK=246, alpha=0.003032696689659577, l1_ratio=0.009813852653622251)
    recAll.save_model(folder_path='Models/', file_name='2k' + recAll.RECOMMENDER_NAME)

    recAll = SLIMElasticNetRecommender(URM_train_list[2])
    recAll.fit(topK=246, alpha=0.003032696689659577, l1_ratio=0.009813852653622251)
    recAll.save_model(folder_path='Models/', file_name='3k' + recAll.RECOMMENDER_NAME)



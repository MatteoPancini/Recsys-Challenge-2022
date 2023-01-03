if __name__ == '__main__':

    from Utils.recsys2022DataReader import createURMBinary, createSmallICM
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
    from Utils.writeSubmission import write_submission_List_Combination

    # Loading URM
    URM = createURMBinary()
    ICM = createSmallICM()

    # Create the recommenders
    recommender_RP3beta = RP3betaRecommender(URM)
    recommender_RP3beta.fit(alpha=0.8462944464325309, beta=0.3050885269698352, topK=78)

    recommender_ItemKNN = ItemKNNCFRecommender(URM_train=URM)
    recommender_ItemKNN.fit(ICM=ICM, shrink=919, topK=584, similarity='dice', normalization='bm25')



    # Write the submission file
    write_submission_List_Combination(recommender1=recommender_RP3beta, recommender2=recommender_ItemKNN,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('prova'))
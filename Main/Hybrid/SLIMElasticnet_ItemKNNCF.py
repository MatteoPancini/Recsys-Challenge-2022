if __name__ == '__main__':

    from Utils.recsys2022DataReader import createBumpURM
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.writeSubmission import write_submission

    URM = createBumpURM()

    ItemKNNCF_recommender = ItemKNNCFRecommender(URM)
    SLIM_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM)

    ItemKNNCF_recommender.fit(topK=329, shrink=97)
    SLIM_recommender.fit(topK=362, alpha=0.06747829810332745, l1_ratio=0.0005493724398243842)

    recommender = LinearHybridTwoRecommenderTwoVariables(URM, Recommender_1=ItemKNNCF_recommender, Recommender_2=SLIM_recommender)
    recommender.fit(alpha=0.2)

    write_submission(recommender=recommender,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('SLIMElasticnet_ItemKNNCF'))
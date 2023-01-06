if __name__ == '__main__':

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Utils.recsys2022DataReader import createURMBinary
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURMBinary()

    IALS_init = ImplicitALSRecommender(URM)
    IALS_init.fit(factors=110, alpha=7, iterations=57, regularization=0.0008866558623568822)

    RP3beta_init = RP3betaRecommender(URM)
    RP3beta_init.fit(topK=77, alpha=0.8401946814961014, beta=0.3073181471251768)

    SLIM_init = SLIMElasticNetRecommender(URM)
    SLIM_init.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    ItemKNN_init = ItemKNNCFRecommender(URM)
    ItemKNN_init.fit(topK=123, shrink=510, similarity="cosine", feature_weighting="TF-IDF")

    hybrid1_init = LinearHybridTwoRecommenderTwoVariables(URM, Recommender_1=RP3beta_init,
                                                          Recommender_2=SLIM_init)
    hybrid1_init.fit(alpha=0.40726736669265445, beta=0.7317482903276693)

    hybrid2_init = LinearHybridTwoRecommenderTwoVariables(URM, Recommender_1=IALS_init,
                                                          Recommender_2=hybrid1_init)
    hybrid2_init.fit(alpha=0.019318928403041356, beta=0.8537494424674974)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM, Recommender_1=hybrid2_init,
                                                                Recommender_2=ItemKNN_init)
    recommender_Hybrid.fit(alpha=0.5182132379810547, beta=4.19321787406275e-06)


    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('ItemKNNIALSSlimRP3_0501'))
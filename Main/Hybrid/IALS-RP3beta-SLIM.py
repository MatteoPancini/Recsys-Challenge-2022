if __name__ == '__main__':

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Utils.recsys2022DataReader import createURMBinary
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURMBinary()

    factors = 110
    alpha = 7
    iterations = 57
    regularization = 0.0008866558623568822

    rec3 = ImplicitALSRecommender(URM)
    rec3.fit(factors=factors, alpha=alpha, iterations=iterations, regularization=regularization)


    rec1 = RP3betaRecommender(URM)
    rec1.fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

    rec2 = SLIMElasticNetRecommender(URM)
    rec2.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

    hybrid1 = LinearHybridTwoRecommenderTwoVariables(URM_train=URM, Recommender_1=rec1,
                                                                Recommender_2=rec2)
    hybrid1.fit(alpha=0.4133522121773261, beta=0.7451419993321209)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM, Recommender_1=rec3, Recommender_2=hybrid1)
    recommender_Hybrid.fit(alpha=0.019318928403041356, beta=0.8537494424674974)


    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('IALSSlimRP3_0401'))
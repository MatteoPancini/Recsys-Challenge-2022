if __name__ == '__main__':

    from Utils.recsys2022DataReader import createURMBinary
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables, LinearHybridTwoRecommenderOneVariable
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURMBinary()

    # Create the recommenders
    recommender_RP3beta = RP3betaRecommender(URM)
    recommender_RP3beta.fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

    recommender_P3alpha = P3alphaRecommender(URM)
    recommender_P3alpha.fit(topK=116, alpha=0.8763131065621229)

    hybrid1 = LinearHybridTwoRecommenderTwoVariables(URM_train=URM, Recommender_1=recommender_RP3beta, Recommender_2=recommender_P3alpha)
    hybrid1.fit(alpha=0.5042061133754471, beta=0.1229236356527148)

    recommender_Slim_Elasticnet = SLIMElasticNetRecommender(URM)
    recommender_Slim_Elasticnet.fit(topK=241, alpha=0.0031642653228324906, l1_ratio=0.009828283497311959)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM, Recommender_1=hybrid1, Recommender_2=recommender_Slim_Elasticnet)
    recommender_Hybrid.fit(alpha=0.4988757159276701, beta=0.6213438538130325)

    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('SlimRP3P3A_0201_2'))
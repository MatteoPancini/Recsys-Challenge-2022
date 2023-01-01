if __name__ == '__main__':

    from Utils.recsys2022DataReader import createURMBinary
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURMBinary()

    # Create the recommenders
    recommender_RP3beta = RP3betaRecommender(URM)
    recommender_RP3beta.fit(alpha=0.8462944464325309, beta=0.3050885269698352, topK=78)

    recommender_Slim_Elasticnet = SLIMElasticNetRecommender(URM)
    recommender_Slim_Elasticnet.fit(topK=211, alpha=0.003520668066481557, l1_ratio=0.007825415595326402)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM, Recommender_1=recommender_RP3beta, Recommender_2=recommender_Slim_Elasticnet)
    recommender_Hybrid.fit(alpha=0.26851340374280425, beta=0.584035197302269)

    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('SlimRP3_3112'))
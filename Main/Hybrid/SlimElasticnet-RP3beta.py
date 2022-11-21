if __name__ == '__main__':

    from Utils.recsys2022DataReader import createBumpURM
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommender
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createBumpURM()

    # Create the recommenders
    recommender_Slim_Elasticnet = MultiThreadSLIM_SLIMElasticNetRecommender(URM)
    recommender_Slim_Elasticnet.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

    recommender_RP3beta = RP3betaRecommender(URM)
    recommender_RP3beta.fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

    recommender_Hybrid = LinearHybridTwoRecommender(URM_train=URM, Recommender_1=recommender_Slim_Elasticnet, Recommender_2=recommender_RP3beta)
    recommender_Hybrid.fit(alpha=0.27553159386864057)

    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('Hybrid-SlimElasticnet-RP3Beta'))
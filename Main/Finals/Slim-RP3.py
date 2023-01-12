if __name__ == '__main__':

    from Utils.recsys2022DataReader import createSlimBothAssumptionsURM
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createSlimBothAssumptionsURM()

    rec1 = SLIMElasticNetRecommender(URM)
    rec1.fit(topK=3310, alpha=0.0014579129528836648, l1_ratio=0.04059573169766696)

    rec2 = RP3betaRecommender(URM)
    rec2.fit(alpha=0.8285172350759491, beta=0.292180138700761, topK=54)


    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM, Recommender_1=rec1,
                                                                Recommender_2=rec2)
    recommender_Hybrid.fit(alpha=0.5738329337854908, beta=0.269980536299904)


    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('SlimRP3_New_1201'))

    # "MAP": 0.0284057785,
if __name__ == '__main__':

    from Utils.recsys2022DataReader import createSlimBothAssumptionsURM
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createSlimBothAssumptionsURM()

    rec1 = SLIMElasticNetRecommender(URM)
    rec1.fit(topK=3890, alpha=0.0029228929900398564, l1_ratio=0.009916947930369228)

    rec2 = RP3betaRecommender(URM)
    rec2.fit(alpha=0.8285172350759491, beta=0.292180138700761, topK=54)


    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=URM, Recommender_1=rec1,
                                                                Recommender_2=rec2)
    recommender_Hybrid.fit(alpha=0.79278764830278, beta=0.4475002878522597)

    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('SlimRP3_New_1101'))
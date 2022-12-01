if __name__ == '__main__':

    from Utils.recsys2022DataReader import *
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Utils.writeSubmission import write_submission

    URM = createURM()

    recommender_P3alpha = P3alphaRecommender(URM)
    recommender_P3alpha.fit(topK=218, alpha=0.8561168568686058)

    recommender_RP3beta = RP3betaRecommender(URM)
    recommender_RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM, recommender_P3alpha,
                                                                recommender_RP3beta)
    recommender_hybrid.fit(alpha=0.26672657848316894, beta=1.8325046917533472)

    # Write the submission file
    write_submission(recommender=recommender_hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/Hybrid/{}_submission.csv'.format('P3alpha-RP3Beta'))


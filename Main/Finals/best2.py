if __name__ == '__main__':

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Utils.recsys2022DataReader import createSlimBothAssumptionsURM
    from Recommenders.Hybrid.Best_SlimRp3Beta import BestSlimRP3Beta
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createSlimBothAssumptionsURM()

    rec1 = BestSlimRP3Beta(URM)
    rec1.fit()

    rec2 = ImplicitALSRecommender(URM)
    rec2.fit(factors=97, alpha=6, iterations=59, regularization=0.004070427647981844)

    recommender_Hybrid = LinearHybridTwoRecommenderTwoVariables(URM, Recommender_1=rec1, Recommender_2=rec2)
    recommender_Hybrid.fit(alpha=0.7289755420016903, beta=0.03978984202537053)

    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('Final2'))

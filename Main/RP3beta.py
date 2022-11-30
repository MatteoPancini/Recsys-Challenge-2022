if __name__ == '__main__':

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Utils.recsys2022DataReader import createURM
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURM()

    #Create the recommender
    recommender = RP3betaRecommender(URM, verbose=False)
    recommender.fit()

    #Write the submission file
    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('RP3beta'))
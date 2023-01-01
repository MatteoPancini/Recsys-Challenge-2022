if __name__ == '__main__':

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Utils.recsys2022DataReader import *
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURMBinary()

    #Create the recommender
    recommender = RP3betaRecommender(URM, verbose=False)
    recommender.fit(alpha=0.8462944464325309, beta=0.3050885269698352, topK=78)

    #Write the submission file
    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('RP3beta'))
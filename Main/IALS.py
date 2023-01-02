if __name__ == '__main__':

    from Utils.recsys2022DataReader import *
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Utils.writeSubmission import write_submission

    URM = createURMBinary()
    '''
    factors = 110
    alpha = 7
    iterations = 57
    regularization = 0.0008866558623568822
    '''
    recommender_IALS = ImplicitALSRecommender(URM)
    recommender_IALS.fit(factors=110, alpha=7, regularization=0.0008866558623568822, iterations=57)

    write_submission(recommender=recommender_IALS,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('IALS0101'))

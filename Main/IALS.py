if __name__ == '__main__':

    from Utils.recsys2022DataReader import createBumpURM
    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Utils.writeSubmission import write_submission

    URM = createBumpURM()

    recommender_IALS = ImplicitALSRecommender(URM)
    recommender_IALS.fit(factors=600, alpha=11, regularization=0.01, iterations=97)

    write_submission(recommender=recommender_IALS,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('IALS'))

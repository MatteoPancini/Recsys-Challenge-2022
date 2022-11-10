if __name__ == '__main__':

    from Utils.recsys2022DataReader import createBumpURM
    import pandas as pd
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    from Utils.writeSubmission import write_submission

    URM = createBumpURM()

    recommender_IALS = IALSRecommender(URM)
    recommender_IALS.fit(num_factors=600, reg=0.036268979348825274, alpha=21.411061579022103, epochs=2)

    write_submission(recommender=recommender_IALS,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('IALS'))

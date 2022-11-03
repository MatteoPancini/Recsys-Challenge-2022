if __name__ == '__main__':
    from Utils.createURM import createURM
    from Utils.writeSubmission import write_submission
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

    URM = createURM()

    recommender = UserKNNCFRecommender(URM)
    recommender.fit(shrink=int(16.9579146232028), topK=int(31.195369834225907))

    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('UserKNNCF'))


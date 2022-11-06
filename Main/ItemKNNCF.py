if __name__ == '__main__':

    from Utils.createURM import createURM2
    from Utils.writeSubmission import write_submission
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    URM = createURM2()

    recommender = ItemKNNCFRecommender(URM)
    # BEST recommender.fit(shrink = int(11.360087017080575), topK = int(24.286589663434658))

    recommender.fit(shrink = 58.04637977082229, topK = 146)

    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('ItemKNNCF'))
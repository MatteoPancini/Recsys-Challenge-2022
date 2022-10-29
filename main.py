if __name__ == '__main__':

    from Recommenders.NonPersonalizedRecommender import TopPop
    from Utils.loadURM import load_URM
    from Utils.writeSubmission import write_submission

    URM = load_URM("Input/interactions_and_impressions.csv")

    topPopRecommender = TopPop(URM)
    topPopRecommender.fit()
    topPopRecommender.recommend()



    write_submission(recommender=topPopRecommender, target_users_path="Input/data_target_users_test.csv",
                     out_path='Output/{}_submission.csv'.format(topPopRecommender.RECOMMENDER_NAME))
if __name__ == '__main__':

    from Recommenders.NonPersonalizedRecommender import TopPop
    from Utils.loadURM import load_URM
    from Utils.writeSubmission import write_submission
    import pandas as pd
    import numpy as np

    URM = load_URM("Input/interactions_and_impressions.csv")

    user_list = pd.read_csv('../Input/interactions_and_impressions.csv')['UserID'].unique().tolist()
    user_array = np.array(user_list)

    topPopRecommender = TopPop(URM)
    topPopRecommender.fit()
    topPopRecommender.recommend(user_array)


    write_submission(recommender=topPopRecommender, target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format(topPopRecommender.RECOMMENDER_NAME))
if __name__ == '__main__':

    from Utils.recsys2022DataReader import *
    from Recommenders.Hybrid.InteractionsHybridRecommender import InteractionsHybridRecommender
    from Utils.writeSubmission import write_submission

    # Loading URM
    URM = createURM()
    ICM = createSmallICM()

    # Create the recommenders

    recommender_Hybrid = InteractionsHybridRecommender(URM_train=URM, ICM=ICM)
    recommender_Hybrid.fit()

    # Write the submission file
    write_submission(recommender=recommender_Hybrid,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}_submission.csv'.format('NEWHybrid-Group-Interactions'))
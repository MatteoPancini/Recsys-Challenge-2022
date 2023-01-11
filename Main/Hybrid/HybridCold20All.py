if __name__ == '__main__':

    from Utils.recsys2022DataReader import *
    from Recommenders.Hybrid.HybridCold20WithAll import HybridCold20WithAll
    from Utils.writeSubmission import write_submission

    URM = createURMBinary()
    recommender = HybridCold20WithAll(URM_train=URM)
    recommender.fit()

    # Write the submission file
    write_submission(recommender=recommender,
                     target_users_path="../../Input/data_target_users_test.csv",
                     out_path='../../Output/{}0601_submission.csv'.format('Hybrid20All'))





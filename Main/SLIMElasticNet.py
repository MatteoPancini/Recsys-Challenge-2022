if __name__ == '__main__':

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.recsys2022DataReader import createURM
    from Utils.writeSubmission import write_submission
    from datetime import datetime

    URM = createURM()

    recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM, verbose=False)
    # BEST recommender.fit(alpha=0.06747829810332745, l1_ratio=0.039535749124922186, topK=int(334.5605306683192))

    recommender.fit(alpha=0.009579327319679334, l1_ratio=0.3586486802592749, topK=159)

    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_{}_submission.csv'.format('SLIMElasticNet', datetime.now().strftime(
            '%b%d_%H-%M-%S')))
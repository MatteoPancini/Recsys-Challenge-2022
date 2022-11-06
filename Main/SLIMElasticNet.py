if __name__ == '__main__':

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.createURM import createURM
    from Utils.writeSubmission import write_submission

    URM = createURM()

    recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM, verbose=False)
    recommender.fit(alpha=0.04137068723357783, l1_ratio=0.039535749124922186, topK=int(334.5605306683192))

    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('SLIMElasticNet'))
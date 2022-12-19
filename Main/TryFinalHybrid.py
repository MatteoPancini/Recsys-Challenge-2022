if __name__ == '__main__':

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    from Utils.recsys2022DataReader import *
    from Utils.writeSubmission import write_submission
    from Recommenders.Hybrid.HandMade.HybridAll import HybridAll
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from datetime import datetime

    URM = createURM()
    ICM = createSmallICM()

    rec1 = MultiThreadSLIM_SLIMElasticNetRecommender(URM, verbose=False)
    rec1.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

    rec2 = HybridAll(URM, ICM)
    rec2.fit()

    hybrid = LinearHybridTwoRecommenderTwoVariables(URM, rec1, rec2)
    hybrid.fit(alpha=0.271661709765995, beta=0.7658632769751493)


    write_submission(recommender=hybrid,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_{}_submission.csv'.format('ProvaHybrid', datetime.now().strftime(
            '%b%d_%H-%M-%S')))
if __name__ == '__main__':

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
    from Utils.recsys2022DataReader import *
    from Utils.writeSubmission import write_submission
    from datetime import datetime

    URM = createURMBinary()

    recommender = SLIMElasticNetRecommender(URM)
    #recommender.load_model(folder_path='../Models/', file_name='All' + recommender.RECOMMENDER_NAME)
    # BEST recommender.fit(alpha=0.06747829810332745, l1_ratio=0.039535749124922186, topK=int(334.5605306683192))

    recommender.fit(topK=396, alpha=0.0026732398099801086, l1_ratio=0.00859835955640564)

    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_{}_submission.csv'.format('SLIMElasticNet0901', datetime.now().strftime('%b%d_%H-%M-%S')))
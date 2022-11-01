if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import matplotlib.pyplot as pyplot
    from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
    from Utils.createURM import createURM
    from Utils.writeSubmission import write_submission
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    URM = createURM()

    recommender = ItemKNNCFRecommender(URM)
    recommender.fit(shrink=10, topK=25)

    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('CFItemKNN'))


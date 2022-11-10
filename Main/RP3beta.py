if __name__ == '__main__':

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    from Utils.createURM import createBumpURM
    import pandas as pd
    from Utils.writeSubmission import write_submission

    # Loading URM
    dataset = pd.read_csv('../Input/interactions_and_impressions.csv')
    URM = createBumpURM(dataset)

    #Create the recommender
    recommender = RP3betaRecommender(URM, verbose=False)
    recommender.fit(alpha=0.3596922439611471, beta=0.47779864322375626, topK=309, implicit=True)

    #Write the submission file
    write_submission(recommender=recommender,
                     target_users_path="../Input/data_target_users_test.csv",
                     out_path='../Output/{}_submission.csv'.format('RP3beta'))
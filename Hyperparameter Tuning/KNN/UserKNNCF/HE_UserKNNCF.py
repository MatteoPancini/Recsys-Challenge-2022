if __name__ == '__main__':

    from Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample
    from Evaluation.Evaluator import EvaluatorHoldout
    import csv
    import pandas as pd
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
    from Utils.createURM import *


    # ---------------------------------------------------------------------------------------------------------
    # Creating CSV settings
    header = ['rec_name', 'topK', 'shrink', 'similarity', 'feature_weighting', 'MAP']

    with open('logs/HE_UserKNNCF_NewURM.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # ---------------------------------------------------------------------------------------------------------
    # Loading URM
    dataset = pd.read_csv('/Users/matteopancini/PycharmProjects/recsys-challenge-2022-Pancini-Vitali/Input/interactions_and_impressions.csv')
    URM = createBumpURM(dataset)

    # ---------------------------------------------------------------------------------------------------------
    # Preparing training, validation, test split and evaluator
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # ---------------------------------------------------------------------------------------------------------
    # Exploration of hyperparameters

    for topK in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        for shrink in [10, 50, 100, 500, 1000, 10000]:
            for similarity in ["tanimoto", "cosine", "euclidean", "jaccard"]:
                for feature_weighting in ["none", "BM25", "TF-IDF"]:
                    print(f"tuning: " + str(topK) + " " + str(shrink) + " " + str(similarity) + " " + str(feature_weighting))
                    recommender = UserKNNCFRecommender(URM_train)
                    recommender.fit(topK=topK, shrink=shrink, similarity=similarity, feature_weighting=feature_weighting)

                    results, _ = evaluator_validation.evaluateRecommender(recommender)
                    results = results.loc[10]['MAP']

                    data = [recommender.RECOMMENDER_NAME, topK, shrink, similarity, feature_weighting, results]
                    with open('logs/HE_UserKNNCF_NewURM.csv', 'a+', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        f.write(str(data) +'\n')

    exploration = pd.read_csv("logs/HE_UserKNNCF_OldURM.csv")
    exploration.sort_values(["MAP"], axis=0, ascending=False, inplace=True)
    print(exploration)
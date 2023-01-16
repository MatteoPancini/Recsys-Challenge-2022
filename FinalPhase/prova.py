if __name__ == "__main__":

    from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
    from Recommenders.Hybrid.Best_SlimRp3Beta import BestSlimRP3Beta
    from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
    from Evaluation.Evaluator import EvaluatorHoldout
    from Utils.recsys2022DataReader import *
    import numpy as np
    from datetime import datetime
    import json

    URM_train_init = load_FinalURMTrainInit()
    URM_test = load_FinalURMTest()

    rec1 = BestSlimRP3Beta(URM_train_init)
    rec1.fit()

    rec2 = ImplicitALSRecommender(URM_train_init)
    rec2.fit(factors=97, alpha=6, iterations=59, regularization=0.004070427647981844)

    recommender_hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train_init, Recommender_1=rec1, Recommender_2=rec2)
    recommender_hybrid.fit(alpha=0.9845938449285698, beta=0.056648148734841475)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    result_dict, _ = evaluator_test.evaluateRecommender(recommender_hybrid)

    # ---------------------------------------------------------------------------------------------------------
    # Writing hyperparameter into a log

    resultParameters = result_dict.to_json(orient="records")
    parsed = json.loads(resultParameters)

    with open("logs/" + "best1" + "_logs_" + datetime.now().strftime(
            '%b%d_%H-%M-%S') + ".json", 'w') as json_file:
        json.dump(parsed, json_file, indent=4)
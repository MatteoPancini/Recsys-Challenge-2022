from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridWarm(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train):
        super(HybridWarm, self).__init__(URM_train)

    def fit(self):

        self.rec1 = RP3betaRecommender(self.URM_train)
        self.rec1.fit(alpha=0.572121247163269, beta=0.3107107930844788, topK=92)

        self.rec2 = SLIMElasticNetRecommender(self.URM_train)
        self.rec2.fit(topK=185, alpha=0.06551072224428456, l1_ratio=0.032574129361384)

        self.Hybrid = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1= self.rec2, Recommender_2=self.rec1)
        self.Hybrid.fit(alpha=0.26787679343978904, beta=0.4557292556511457)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights

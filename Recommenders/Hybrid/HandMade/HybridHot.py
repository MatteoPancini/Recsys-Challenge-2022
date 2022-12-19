from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridHot(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train):
        super(HybridHot, self).__init__(URM_train)

    def fit(self):

        self.rec2 = RP3betaRecommender(self.URM_train)
        self.rec2.fit(alpha=0.7136052911660057, beta=0.44828831909194655, topK=54)

        self.rec1 = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.rec1.fit(topK=429, alpha=0.0047217460142242595, l1_ratio=0.501517968826842)

        self.Hybrid = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1= self.rec1, Recommender_2=self.rec2)
        self.Hybrid.fit(alpha=0.9812218605848964, beta=0.9233919307046949)

    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
from Recommenders.Hybrid.HandMade.HybridCold import HybridCold
import numpy as np


class HybridAll(BaseRecommender):

    RECOMMENDER_NAME = "Cold1Hybrid"

    def __init__(self, URM_train):
        super(HybridAll, self).__init__(URM_train)



    def fit(self):
        self.rec1 = SLIMElasticNetRecommender(self.URM_train)
        self.rec1.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893,
                                                       topK=359)
        self.rec2 = RP3betaRecommender(self.URM_train)
        self.rec2.fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

        self.Hybrid = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.rec1, self.rec2)
        self.Hybrid.fit(alpha=0.18228980979705656, beta=0.5426630600143958)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights

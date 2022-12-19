from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderOneVariable
import numpy as np


class HybridCold(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train, ICM):
        super(HybridCold, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):

        self.rec1 = ItemKNNCFRecommender(self.URM_train)
        self.rec1.fit(ICM=self.ICM, topK=5893, shrink=50, similarity='rp3beta', normalization='tfidf')

        self.rec2 = RP3betaRecommender(self.URM_train)
        self.rec2.fit(alpha=0.6627101454340679, beta=0.2350020032542621, topK=250)

        self.Hybrid = LinearHybridTwoRecommenderOneVariable(self.URM_train, self.rec1, self.rec2)
        self.Hybrid.fit(alpha=0.2584478495159924)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights






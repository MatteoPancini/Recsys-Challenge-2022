from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
from Recommenders.Hybrid.HandMade.Hybrid0COLDItemKNNRP3beta import Hybrid0
import numpy as np


class Hybrid1(BaseRecommender):

    RECOMMENDER_NAME = "Cold1Hybrid"

    def __init__(self, URM_train, ICM):
        super(Hybrid1, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):

        self.rec1 = Hybrid0(self.URM_train, self.ICM)
        self.rec1.fit()

        self.rec2 = P3alphaRecommender(self.URM_train)
        self.rec2.fit(topK=218, alpha=0.8561168568686058)

        self.Hybrid = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.rec1, self.rec2)
        self.Hybrid.fit(alpha=0.884548752019934, beta=0.9136029911355894)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights

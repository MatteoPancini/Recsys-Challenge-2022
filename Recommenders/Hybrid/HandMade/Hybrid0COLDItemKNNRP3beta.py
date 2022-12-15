from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
import numpy as np


class Hybrid0(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train, ICM):
        super(Hybrid0, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):

        self.rec1 = ItemKNNCFRecommender(self.URM_train)
        self.rec1.fit(self.ICM, shrink=1665.2431108249625, topK=3228, similarity='dice',
                                            normalization='bm25')

        self.rec2 = RP3betaRecommender(self.URM_train)
        self.rec2.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)

        self.Hybrid = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.rec1, self.rec2)
        self.Hybrid.fit(alpha=0.20725587449876504, beta=0.6358928843508406)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights






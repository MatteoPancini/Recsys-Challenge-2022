from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
from tqdm import tqdm
import numpy as np

class BestSlimRP3Beta(BaseRecommender):

    RECOMMENDER_NAME = "Hybrid_Slim-RP3beta"

    def __init__(self, URM_train):
        super(BestSlimRP3Beta, self).__init__(URM_train)

    def fit(self):
        rec1 = SLIMElasticNetRecommender(URM_train=self.URM_train)
        rec1.fit(topK=3310, alpha=0.0014579129528836648, l1_ratio=0.04059573169766696)

        rec2 = RP3betaRecommender(URM_train=self.URM_train)
        rec2.fit(alpha=0.8285172350759491, beta=0.292180138700761, topK=54)

        self.hybrid = LinearHybridTwoRecommenderTwoVariables(URM_train=self.URM_train, Recommender_1=rec1, Recommender_2=rec2)
        self.hybrid.fit(alpha=0.5738329337854908, beta=0.269980536299904)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights = self.hybrid._compute_item_score(user_id_array, items_to_compute)

        return items_weights

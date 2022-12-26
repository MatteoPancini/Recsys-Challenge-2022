from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridHot(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train, ICM):
        super(HybridHot, self).__init__(URM_train)
        self.ICM = ICM

    def fit(self):

        self.rec1 = RP3betaRecommender(self.URM_train)
        self.rec1.fit(alpha=0.6078606485515248, beta=0.32571505237450094, topK=52)

        self.rec2 = ItemKNNCFRecommender(self.URM_train)
        self.rec2.fit(ICM=self.ICM, topK=461, shrink=10, similarity='rp3beta', normalization='tfidf')

        #self.rec1 = SLIMElasticNetRecommender(self.URM_train)
        #self.rec1.fit(topK=429, alpha=0.0047217460142242595, l1_ratio=0.501517968826842)

        self.Hybrid = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1= self.rec1, Recommender_2=self.rec2)
        self.Hybrid.fit(alpha=0.8476487776384586, beta=0.1505476499435428)

    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.Hybrid._compute_item_score(user_id_array, items_to_compute)

        return item_weights

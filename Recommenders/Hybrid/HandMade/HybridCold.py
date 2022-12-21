from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridCold(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train, ICM):
        super(HybridCold, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):
        self.rec1 = RP3betaRecommender(self.URM_train)
        self.rec1.fit(alpha=0.6627101454340679, beta=0.2350020032542621, topK=250)

        self.rec2 = ItemKNNCFRecommender(self.URM_train)
        self.rec2.fit(ICM=self.ICM, topK=5893, shrink=50, similarity='rp3beta', normalization='tfidf')

        self.rec3 = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.rec3.fit(alpha=0.22747568631546267, l1_ratio=0.007954654152433904, topK=214)

        self.hybrid1 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.rec1, self.rec2)
        self.hybrid1.fit(alpha=0.3281013138044576, beta=0.7779050787709695)

        self.hybrid2 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.hybrid1, self.rec3)
        self.hybrid2.fit(alpha=0.5973236168542123, beta=0.057935940935008756)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.hybrid2._compute_item_score(user_id_array, items_to_compute)

        return item_weights






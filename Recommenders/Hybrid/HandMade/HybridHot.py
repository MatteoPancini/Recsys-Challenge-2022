from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridHot(BaseRecommender):

    RECOMMENDER_NAME = "HotHybrid"

    def __init__(self, URM_train, ICM):
        super(HybridHot, self).__init__(URM_train)
        self.ICM = ICM

    def fit(self):

        self.rec1 = RP3betaRecommender(self.URM_train)
        self.rec1.fit(alpha=0.6078606485515248, beta=0.32571505237450094, topK=52)

        self.rec2 = ItemKNNCFRecommender(self.URM_train)
        self.rec2.fit(ICM=self.ICM, topK=461, shrink=10, similarity='rp3beta', normalization='tfidf')

        self.hybrid1 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1=self.rec1, Recommender_2=self.rec2)
        self.hybrid1.fit(alpha=0.8476487776384586, beta=0.1505476499435428)

        self.rec3= SLIMElasticNetRecommender(self.URM_train)
        self.rec3.fit(topK=298, alpha=0.041853285688557666, l1_ratio=0.0165397312920016)

        self.hybrid2 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1=self.hybrid1, Recommender_2=self.rec3)
        self.hybrid2.fit(alpha=0.5432232314277623, beta=0.7787958198874051)

    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.hybrid2._compute_item_score(user_id_array, items_to_compute)

        return item_weights
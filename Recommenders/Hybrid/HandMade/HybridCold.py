from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender,SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridCold(BaseRecommender):

    RECOMMENDER_NAME = "ColdHybrid"

    def __init__(self, URM_train, ICM):
        super(HybridCold, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):
        self.rec1 = RP3betaRecommender(self.URM_train)
        self.rec1.fit(alpha=0.8815611011233834, beta=0.23472570066237713, topK=225)

        self.rec2 = ItemKNNCFRecommender(self.URM_train)
        self.rec2.fit(ICM=self.ICM, topK=1296, shrink=51, similarity='rp3beta', normalization='tfidf')

        #self.rec3 = SLIMElasticNetRecommender(self.URM_train)
        #self.rec3.fit(alpha=0.22747568631546267, l1_ratio=0.007954654152433904, topK=214)

        self.hybrid1 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.rec1, self.rec2)
        self.hybrid1.fit(alpha=0.8190677327782062, beta=0.686509249107007)

        #self.hybrid2 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.hybrid1, self.rec3)
        #self.hybrid2.fit(alpha=0.5973236168542123, beta=0.057935940935008756)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.hybrid1._compute_item_score(user_id_array, items_to_compute)

        return item_weights






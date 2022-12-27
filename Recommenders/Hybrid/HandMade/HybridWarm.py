from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables

class HybridWarm(BaseRecommender):

    RECOMMENDER_NAME = "WarmHybrid"

    def __init__(self, URM_train, ICM):
        super(HybridWarm, self).__init__(URM_train)
        self.ICM = ICM

    def fit(self):

        self.rec1 = RP3betaRecommender(self.URM_train)
        self.rec1.fit(alpha=0.7849910963981444, beta=0.3219406144420833, topK=64)

        self.rec2 = ItemKNNCFRecommender(self.URM_train)
        self.rec2.fit(ICM=self.ICM, topK=377, shrink=10, similarity='rp3beta', normalization='tfidf')

        self.hybrid1 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1=self.rec1, Recommender_2=self.rec2)
        self.hybrid1.fit(alpha=0.6745409371119246, beta=0.19918230600969603)

        self.rec3 = SLIMElasticNetRecommender(self.URM_train)
        self.rec3.fit(topK=258, alpha=0.035237980092119314, l1_ratio=0.05512644878845981)

        self.hybrid2 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1=self.hybrid1, Recommender_2=self.rec3)
        self.hybrid2.fit(alpha=0.7764977714573097, beta=0.655210940917789)


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        item_weights = self.hybrid2._compute_item_score(user_id_array, items_to_compute)

        return item_weights
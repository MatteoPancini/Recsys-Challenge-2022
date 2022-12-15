from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
import numpy as np


class GroupHybrid(BaseRecommender):

    RECOMMENDER_NAME = "GroupHybrid"

    def __init__(self, URM_train, ICM):
        super(GroupHybrid, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):

        # ------------------
        # COLD

        #self.RP3betaG1 = RP3betaRecommender(self.URM_train)
        #self.RP3betaG1.fit(alpha=0.4770536011269113, beta=0.36946801560978637, topK=190)


        # ------------------
        # WARM

        #self.RP3betaG2 = RP3betaRecommender(self.URM_train)
        #self.RP3betaG2.fit(alpha=0.6687877652632948, beta=0.3841332145259308, topK=103)

        self.ItemKNNCF = ItemKNNCFRecommender(self.URM_train)
        self.ItemKNNCF.fit(self.ICM, shrink=1665.2431108249625, topK=3228, similarity='dice',
                      normalization='bm25')
        # ------------------
        # HOT

        self.recommender_P3alpha = P3alphaRecommender(self.URM_train)
        self.recommender_P3alpha.fit(topK=218, alpha=0.8561168568686058)

        self.recommender_RP3beta = RP3betaRecommender(self.URM_train)
        self.recommender_RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)

        self.recommender_hybridHOT = LinearHybridTwoRecommenderTwoVariables(self.URM_train, self.recommender_P3alpha,
                                                                    self.recommender_RP3beta)
        self.recommender_hybridHOT.fit(alpha=0.26672657848316894, beta=1.8325046917533472)

        '''
        self.RP3betaG3 = RP3betaRecommender(self.URM_train)
        self.RP3betaG3.fit(alpha=0.5674554399991163, beta=0.38051048617892586, topK=100)
        

        self.ItemKNNCFG0 = ItemKNNCFRecommender(self.URM_train)
        self.ItemKNNCFG0.fit(self.ICM, shrink=50, topK=5893, similarity='rp3beta',
                            normalization='tfidf')

        self.SlimElasticNetG2 = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.SlimElasticNetG2.fit(topK=359, alpha=0.04183472018614359, l1_ratio=0.03260349571135893)

        self.SlimElasticNetG1 = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.SlimElasticNetG1.fit(topK=216, alpha=0.09376418450176816, l1_ratio=0.03954091993785014)
        
        
        
        self.ItemKNNCFG0 = ItemKNNCFRecommender(self.URM_train)
        self.ItemKNNCFG0.fit(self.ICM, shrink=200.37965608072673, topK=4985, similarity='rp3beta',
                           normalization='tfidf')
        '''


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        for i in range(len(user_id_array)):
            interactions = len(self.URM_train[user_id_array[i],:].indices)

            if interactions < 21:
                items_weights1 = self.ItemKNNCF._compute_item_score(user_id_array, items_to_compute)

            elif interactions > 20 and interactions < 50:
                items_weights1 = self.recommender_hybridHOT._compute_item_score(user_id_array, items_to_compute)

            elif interactions > 49:
                items_weights1 = self.recommender_hybridHOT._compute_item_score(user_id_array, items_to_compute)


        return items_weights1






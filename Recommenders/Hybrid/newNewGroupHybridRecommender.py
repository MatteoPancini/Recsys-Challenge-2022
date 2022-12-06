from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
import numpy as np


class GroupHybrid(BaseRecommender):

    RECOMMENDER_NAME = "GroupHybrid"

    def __init__(self, URM_train, ICM):
        super(GroupHybrid, self).__init__(URM_train)
        self.ICM = ICM



    def fit(self):

        #------------------
        # Group 0
        '''
        self.ItemKNNCFG0 = ItemKNNCFRecommender(self.URM_train)
        self.ItemKNNCFG0.fit(self.ICM, shrink=200.37965608072673, topK=4985, similarity='rp3beta',
                           normalization='tfidf')


        # ------------------
        # Group 1

        self.RP3betaG1 = RP3betaRecommender(self.URM_train)
        self.RP3betaG1.fit(alpha=0.4770536011269113, beta=0.36946801560978637, topK=190)


        # ------------------
        # Group 2

        self.RP3betaG2 = RP3betaRecommender(self.URM_train)
        self.RP3betaG2.fit(alpha=0.6687877652632948, beta=0.3841332145259308, topK=103)


        # ------------------
        # Group 3

        self.RP3betaG3 = RP3betaRecommender(self.URM_train)
        self.RP3betaG3.fit(alpha=0.5674554399991163, beta=0.38051048617892586, topK=100)
        '''


        self.RP3beta = RP3betaRecommender(self.URM_train)
        self.RP3beta.fit(alpha=0.5126756776495514, beta=0.396119587486951, topK=100)




    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        for i in range(len(user_id_array)):
            interactions = len(self.URM_train[user_id_array[i],:].indices)

            if interactions < 110:
                items_weights1 = self.RP3beta._compute_item_score(user_id_array, items_to_compute)

            elif interactions > 109 and interactions < 150:
                items_weights1 = self.RP3beta._compute_item_score(user_id_array, items_to_compute)

            elif interactions > 149:
                items_weights1 = self.RP3beta._compute_item_score(user_id_array, items_to_compute)


        return items_weights1






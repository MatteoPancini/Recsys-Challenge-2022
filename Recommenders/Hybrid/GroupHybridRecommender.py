from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
import numpy as np


class GroupHybrid(BaseRecommender):

    RECOMMENDER_NAME = "GroupHybrid"

    def __init__(self, URM_train, ICM):
        super(GroupHybrid, self).__init__(URM_train)
        self.ICM = ICM

        self.group0 = []
        self.group1 = []
        self.group2 = []
        self.group3 = []

        for group_id in range (4):

            profile_length = np.ediff1d(self.URM_train.indptr)

            block_size = int(len(profile_length) * 0.25)

            sorted_users = np.argsort(profile_length)

            start_pos = group_id * block_size
            end_pos = min((group_id + 1) * block_size, len(profile_length))

            users_in_group = sorted_users[start_pos:end_pos]

            if(group_id == 0):
                self.group0 = users_in_group
            elif(group_id == 1):
                self.group1 = users_in_group
            if (group_id == 2):
                self.group2 = users_in_group
            elif (group_id == 3):
                self.group3 = users_in_group



    def fit(self):

        #------------------
        # Group 0

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






    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        if user_id_array in self.group0:
            print('group0')
            items_weights1 = self.ItemKNNCFG0._compute_item_score(user_id_array, items_to_compute)

        elif user_id_array in self.group1:

            items_weights1 = self.RP3betaG1._compute_item_score(user_id_array, items_to_compute)

        elif user_id_array in self.group2:
            items_weights1 = self.RP3betaG2._compute_item_score(user_id_array, items_to_compute)

        elif user_id_array in self.group3:

            items_weights1 = self.RP3betaG3._compute_item_score(user_id_array, items_to_compute)

        return items_weights1






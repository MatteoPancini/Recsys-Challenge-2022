from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
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

        recommender_ItemKNNCFG0 = ItemKNNCFRecommender(self.URM_train)
        recommender_ItemKNNCFG0.fit(self.ICM, shrink=108.99759968449757, topK=5251, similarity='rp3beta',
                                  normalization='tfidf')

        recommender_RP3betaG0 = RP3betaRecommender(self.URM_train)
        recommender_RP3betaG0.fit(alpha=0.748706443270007, beta=0.16081149387492433, topK=370)

        self.hybridG0 = LinearHybridTwoRecommenderTwoVariables(URM_train=self.URM_train, Recommender_1=recommender_RP3betaG0,
                                                           Recommender_2=recommender_ItemKNNCFG0)
        self.hybridG0.fit(alpha=0.36914252072676557, beta=0.37856318068441236)

        # ------------------
        # Group 1

        recommender_ItemKNNCFG1 = ItemKNNCFRecommender(self.URM_train)
        recommender_ItemKNNCFG1.fit(self.ICM, shrink=976.8108064049092, topK=5300, similarity='cosine',
                                  normalization='bm25')

        recommender_RP3beta = RP3betaRecommender(self.URM_train)
        recommender_RP3beta.fit(alpha=0.6190367265325001, beta=0.4018626515197256, topK=206)

        self.hybridG1 = LinearHybridTwoRecommenderTwoVariables(self.URM_train, recommender_ItemKNNCFG1,
                                                                     recommender_RP3beta)
        self.hybridG1.fit(alpha=0.07806573588790788, beta=0.8465619360796353)

        # ------------------
        # Group 2

        self.SlimElasticNetG2 = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.SlimElasticNetG2.fit(topK=100, alpha=0.010975001075569505, l1_ratio=0.48011657798419954)

        # ------------------
        # Group 3

        self.SlimElasticNetG3 = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM_train)
        self.SlimElasticNetG3.fit(topK=359, alpha=0.04183472018614359, l1_ratio=0.03260349571135893)

    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        if user_id_array in self.group0:
            items_weights1 = self.hybridG0._compute_item_score(user_id_array, items_to_compute)

        elif user_id_array in self.group1:

            items_weights1 = self.hybridG1._compute_item_score(user_id_array, items_to_compute)

        elif user_id_array in self.group2:
            items_weights1 = self.hybridG2._compute_item_score(user_id_array, items_to_compute)

        elif user_id_array in self.group3:

            items_weights1 = self.SlimElasticNetG3._compute_item_score(user_id_array, items_to_compute)

        return items_weights1






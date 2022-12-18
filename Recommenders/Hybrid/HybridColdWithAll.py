from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCFRecommenderPLUS import ItemKNNCFRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables, LinearHybridTwoRecommenderOneVariable
import numpy as np


class GroupHybrid(BaseRecommender):

    RECOMMENDER_NAME = "HybridColdAll"

    def __init__(self, URM_train, ICM):
        super(GroupHybrid, self).__init__(URM_train)
        self.ICM = ICM
        self.URM = URM_train

        group_id = 0

        interactions = []
        for i in range(41629):
            interactions.append(len(URM_train[i, :].nonzero()[0]))

        list_group_interactions = [[0, 20], [21, 49], [50, max(interactions)]]

        lower_bound = list_group_interactions[group_id][0]
        higher_bound = list_group_interactions[group_id][1]

        self.users_cold = [user_id for user_id in range(len(interactions))
                          if (lower_bound <= interactions[user_id] <= higher_bound)]

    def fit(self, alpha=0.5):

        self.alpha = alpha

        #------------------
        # Cold

        recommender_RP3beta = RP3betaRecommender(self.URM, verbose=False)
        recommender_RP3beta.fit(alpha=0.6627101454340679, beta=0.2350020032542621, topK=250)

        recommender_ItemKNN = ItemKNNCFRecommender(self.URM, verbose=False)
        recommender_ItemKNN.fit(ICM=self.ICM, topK=5893, shrink=50, similarity='rp3beta', normalization='tfidf')

        self.hybridG0 = LinearHybridTwoRecommenderOneVariable(URM_train=self.URM_train, Recommender_1=recommender_RP3beta,
                                                           Recommender_2=recommender_ItemKNN)
        self.hybridG0.fit(alpha=0.2584478495159924)

        # ------------------
        # All

        recommender_SlimElasticnet = MultiThreadSLIM_SLIMElasticNetRecommender(self.URM)
        recommender_SlimElasticnet.fit(alpha=0.04183472018614359, l1_ratio=0.03260349571135893, topK=359)

        recommender_RP3beta = RP3betaRecommender(self.URM_train)
        recommender_RP3beta.fit(alpha=0.5586539802603512, beta=0.49634087886207484, topK=322)

        self.hybridAll = LinearHybridTwoRecommenderTwoVariables(URM_train=self.URM,
                                                                              Recommender_1=recommender_SlimElasticnet,
                                                                              Recommender_2=recommender_RP3beta)
        self.hybridAll.fit(alpha=0.18228980979705656, beta=0.5426630600143958)



    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        if user_id_array in self.users_cold:
            items_weightsCold = self.hybridG0._compute_item_score(user_id_array, items_to_compute)
            items_weightAll = self.hybridAll._compute_item_score(user_id_array, items_to_compute)

            items_weights1 = items_weightAll * self.alpha + items_weightsCold * (1-self.alpha)
        else:
            items_weights1 = self.hybridAll._compute_item_score(user_id_array, items_to_compute)

        return items_weights1






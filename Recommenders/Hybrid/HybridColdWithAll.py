from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Hybrid.HandMade.HybridAll import HybridAll
from Recommenders.Hybrid.HandMade.HybridCold import HybridCold
import numpy as np

class HybridColdAllRecommender(BaseRecommender):

    RECOMMENDER_NAME = "HybridColdAll"

    def __init__(self, URM_train, ICM):
        super(HybridColdAllRecommender, self).__init__(URM_train)
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

        self.hybridCold = HybridCold(URM_train=self.URM, ICM=self.ICM)
        self.hybridCold.fit()

        # ------------------
        # All

        self.hybridAll = HybridAll(URM_train=self.URM)
        self.hybridAll.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), 24507])

        for i in range(len(user_id_array)):
            interactions = len(self.URM[i, :].nonzero()[0])

            if interactions <= 20:
                items_weightsCold = self.hybridCold._compute_item_score(user_id_array[i], items_to_compute)
                items_weightAll = self.hybridAll._compute_item_score(user_id_array[i], items_to_compute)

                item_weight_return = items_weightsCold * self.alpha + items_weightAll * (1 - self.alpha)
                item_weights[i, :] = item_weight_return
            else:
                item_weights = self.hybridAll._compute_item_score(user_id_array[i], items_to_compute)

        return item_weights

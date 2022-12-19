from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Hybrid.HandMade.HybridCold import HybridCold
from Recommenders.Hybrid.HandMade.HybridWarm import HybridWarm
from Recommenders.Hybrid.HandMade.HybridHot import HybridHot
import numpy as np

class InteractionsHybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "InteractionsHybridRecommender"

    def __init__(self, URM_train, ICM):
        super(InteractionsHybridRecommender, self).__init__(URM_train)
        self.ICM = ICM

    def fit(self):

        # ------------------
        # COLD

        self.hybrid_cold = HybridCold(URM_train=self.URM_train, ICM=self.ICM)
        self.hybrid_cold.fit()

        # ------------------
        # WARM

        self.hybrid_warm = HybridWarm(URM_train=self.URM_train)
        self.hybrid_warm.fit()

        # ------------------
        # HOT

        self.hybrid_hot = HybridHot(URM_train=self.URM_train)
        self.hybrid_hot.fit()


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        for i in range(len(user_id_array)):
            interactions = len(self.URM_train[user_id_array[i],:].indices)

            if interactions <= 20:
                items_weights1 = self.hybrid_cold._compute_item_score(user_id_array, items_to_compute)
            elif interactions < 50:
                items_weights1 = self.hybrid_warm._compute_item_score(user_id_array, items_to_compute)
            else:
                items_weights1 = self.hybrid_hot._compute_item_score(user_id_array, items_to_compute)

        return items_weights1

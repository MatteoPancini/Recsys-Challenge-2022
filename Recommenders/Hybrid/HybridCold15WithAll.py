from Recommenders.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.Hybrid.LinearHybridRecommender import LinearHybridTwoRecommenderTwoVariables
from tqdm import tqdm
import numpy as np

class InteractionsHybridRecommender(BaseRecommender):

    RECOMMENDER_NAME = "InteractionsHybridRecommender"

    def __init__(self, URM_train):
        super(InteractionsHybridRecommender, self).__init__(URM_train)

    def fit(self):

        # ------------------
        # COLD

        recommender_rp3beta = RP3betaRecommender(URM_train=self.URM_train)
        recommender_rp3beta.fit(alpha=0.8933157693622866, beta=0.14484583688745223, topK=344)

        recommender_itemKNN = ItemKNNCFRecommender(URM_train=self.URM_train)
        recommender_itemKNN.fit(topK=321, shrink=164)

        self.recommender_cold = LinearHybridTwoRecommenderTwoVariables(URM_train=self.URM_train, Recommender_1=recommender_rp3beta, Recommender_2=recommender_itemKNN)
        self.recommender_cold.fit(alpha=0.9196982707381445, beta=0.10860954592979288)


        # ------------------
        # ALL

        factors = 110
        alpha = 7
        iterations = 57
        regularization = 0.0008866558623568822

        rec3 = ImplicitALSRecommender(self.URM_train)
        rec3.fit(factors=factors, alpha=alpha, iterations=iterations, regularization=regularization)

        rec1 = RP3betaRecommender(self.URM_train)
        rec1.fit(alpha=0.8401946814961014, beta=0.3073181471251768, topK=77)

        rec2 = SLIMElasticNetRecommender(self.URM_train)
        rec2.fit(topK=250, alpha=0.00312082198837027, l1_ratio=0.009899185175306373)

        hybrid1 = LinearHybridTwoRecommenderTwoVariables(URM_train=self.URM_train, Recommender_1=rec1,
                                                         Recommender_2=rec2)
        hybrid1.fit(alpha=0.4133522121773261, beta=0.7451419993321209)

        self.recommender_all = LinearHybridTwoRecommenderTwoVariables(self.URM_train, Recommender_1=rec1, Recommender_2=hybrid1)
        self.recommender_all.fit(alpha=0.019318928403041356, beta=0.8537494424674974)

        group_id = 0

        interactions = []
        for i in range(41629):
            interactions.append(len(self.URM_train[i, :].nonzero()[0]))

        list_group_interactions = [[0, 15], [21, 49], [50, max(interactions)]]

        lower_bound = list_group_interactions[group_id][0]
        higher_bound = list_group_interactions[group_id][1]

        self.users_in_group = [user_id for user_id in range(len(interactions)) if (lower_bound <= interactions[user_id] <= higher_bound)]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        items_weights1 = np.empty([len(user_id_array), 24507])

        for i in tqdm(range(len(user_id_array))):

            if i in self.users_in_group:
                items_weights1 = self.recommender_cold._compute_item_score(user_id_array, items_to_compute)
            else:
                items_weights1 = self.recommender_all._compute_item_score(user_id_array, items_to_compute)

        return items_weights1

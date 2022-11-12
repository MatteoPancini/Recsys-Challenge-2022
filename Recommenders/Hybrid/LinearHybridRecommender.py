from ..Recommender_utils import check_matrix
from ..BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class LinearHybridTwoRecommender(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridTwoRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(LinearHybridTwoRecommender, self).__init__(URM_train)
        self.W_sparse = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)


        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights


class LinearHybridThreeRecommender(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridThreeRecommender
    Hybrid of three prediction scores R = R1*alpha + R2*beta + R3*gamma
    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridThreeRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3):
        super(LinearHybridThreeRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.W_sparse = None

    def fit(self, alpha=0.5, beta=0.5, gamma = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array, items_to_compute)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * self.gamma

        return item_weights
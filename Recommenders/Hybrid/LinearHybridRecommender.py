from ..Recommender_utils import check_matrix
from ..BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from numpy import linalg as LA




class LinearHybridTwoRecommenderNoVariables(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridTwoRecommender
    Hybrid of two prediction scores R = R1 + R2
    """

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(LinearHybridTwoRecommenderNoVariables, self).__init__(URM_train, verbose=False)
        self.W_sparse = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self):
        self.weight = 1

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        item_weights_1 /= LA.norm(item_weights_1, 2)
        item_weights_2 /= LA.norm(item_weights_2, 2)



        item_weights = item_weights_1 + item_weights_2

        return item_weights


class LinearHybridTwoRecommenderTwoVariables(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridTwoRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*beta
    """

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(LinearHybridTwoRecommenderTwoVariables, self).__init__(URM_train, verbose=False)
        self.W_sparse = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        item_weights_1 /= LA.norm(item_weights_1, 2)
        item_weights_2 /= LA.norm(item_weights_2, 2)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta

        return item_weights

class LinearHybridTwoRecommenderOneVariable(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridTwoRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(LinearHybridTwoRecommenderOneVariable, self).__init__(URM_train, verbose=False)
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


class LinearHybridThreeRecommenderThreeVariables(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridThreeRecommender
    Hybrid of three prediction scores R = R1*alpha + R2*beta + R3*gamma
    """

    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3):
        super(LinearHybridThreeRecommenderThreeVariables, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.Recommender_3 = Recommender_3
        self.W_sparse = None

    def fit(self, alpha=0.5, beta=0.5, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)
        item_weights_3 = self.Recommender_3._compute_item_score(user_id_array, items_to_compute)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * self.gamma
        return item_weights
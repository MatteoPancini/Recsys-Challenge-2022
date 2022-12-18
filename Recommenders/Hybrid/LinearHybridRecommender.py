from ..Recommender_utils import check_matrix
from ..BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from numpy import linalg as LA
import numpy as np




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

    def fit(self, norm_scores=False):
        self.weight = 1
        self.norm_scores = norm_scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        if self.norm_scores:
            mean1 = np.mean(item_weights_1)
            mean2 = np.mean(item_weights_2)
            std1 = np.std(item_weights_1)
            std2 = np.std(item_weights_2)
            if std1 != 0 and std2 != 0:
                item_weights_1 = (item_weights_1 - mean1) / std1
                item_weights_2 = (item_weights_2 - mean2) / std2

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

    def fit(self, alpha=0.5, beta=0.5, norm_scores=False):
        self.alpha = alpha
        self.beta = beta
        self.norm_scores = norm_scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        if self.norm_scores:
            mean1 = np.mean(item_weights_1)
            mean2 = np.mean(item_weights_2)
            std1 = np.std(item_weights_1)
            std2 = np.std(item_weights_2)
            if std1 != 0 and std2 != 0:
                item_weights_1 = (item_weights_1 - mean1) / std1
                item_weights_2 = (item_weights_2 - mean2) / std2
        #item_weights_1 /= LA.norm(item_weights_1, 2)
        #item_weights_2 /= LA.norm(item_weights_2, 2)

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

    def fit(self, alpha=0.5, norm_scores=False):
        self.alpha = alpha
        self.norm_scores = norm_scores


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)

        if self.norm_scores:
            mean1 = np.mean(item_weights_1)
            mean2 = np.mean(item_weights_2)
            std1 = np.std(item_weights_1)
            std2 = np.std(item_weights_2)
            if std1 != 0 and std2 != 0:
                item_weights_1 = (item_weights_1 - mean1) / std1
                item_weights_2 = (item_weights_2 - mean2) / std2

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights

class LinearHybridTwoRecommenderOneVariableForCold(BaseItemSimilarityMatrixRecommender):
    """ LinearHybridTwoRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha) only for cold users
    """

    def __init__(self, URM_train, Recommender_Cold, Recommender_All):
        super(LinearHybridTwoRecommenderOneVariableForCold, self).__init__(URM_train, verbose=False)
        self.W_sparse = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_Cold = Recommender_Cold
        self.Recommender_All = Recommender_All

    def fit(self, alpha=0.5, norm_scores=False):
        self.alpha = alpha

        group_id = 0

        interactions = []
        for i in range(41629):
            interactions.append(len(self.URM_train[i, :].nonzero()[0]))

        list_group_interactions = [[0, 20], [21, 49], [50, max(interactions)]]

        lower_bound = list_group_interactions[group_id][0]
        higher_bound = list_group_interactions[group_id][1]

        self.users_cold = [user_id for user_id in range(len(interactions)) if (lower_bound <= interactions[user_id] <= higher_bound)]

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), 25975])

        for i in range(len(user_id_array)):
            if user_id_array[i] in self.users_cold:

                item_weights_1 = self.Recommender_Cold._compute_item_score(user_id_array[i], items_to_compute)
                item_weights_2 = self.Recommender_All._compute_item_score(user_id_array[i], items_to_compute)

                item_weight_return = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)
                item_weights[i, :] = item_weight_return
            else:
                item_weight_return = self.Recommender_All._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = item_weight_return

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



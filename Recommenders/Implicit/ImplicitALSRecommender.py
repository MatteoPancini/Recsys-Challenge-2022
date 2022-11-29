import os

import implicit
from ..BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from ..Recommender_utils import check_matrix
import numpy as np

class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            confidence_scaling=None,
            alpha=24
            ):

        self.rec = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=iterations,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads)

        C = check_matrix(self.URM_train, format="csr", dtype=np.float32)
        C.data = 1.0 + alpha * C.data

        self.rec.fit(C, show_progress=self.verbose)

        self.USER_factors = self.rec.user_factors
        self.ITEM_factors = self.rec.item_factors




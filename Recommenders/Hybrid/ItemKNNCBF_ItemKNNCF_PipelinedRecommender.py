#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/11/2022
@author: Michael Vitali
"""

from ..BaseRecommender import BaseRecommender
from ..KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from ..KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class ItemKNNCBFItemKNNCFPipelinedRecommender(BaseRecommender):

    RECOMMENDER_NAME = "ItemKNNCBFItemKNNCFPipelinedRecommender"

    def __init__(
            self,
            URM_train,
            ICM_train,
            topK_knncbf=50,
            shrink_knncbf=100,
    ):
        super().__init__(URM_train)

        recommender_ItemKNNCBF = ItemKNNCBFRecommender(URM_train, ICM_train, verbose=False)
        recommender_ItemKNNCBF.fit(topK_knncbf, shrink_knncbf)

        self.recommender = ItemKNNCFRecommender(recommender_ItemKNNCBF.URM_train.dot(recommender_ItemKNNCBF.W_sparse), verbose=False)

    def fit(
            self,
            shrink_knncf=10,
            topK_knncf=100
    ):
        print("Sono dentro")
        self.recommender.fit(shrink=shrink_knncf, topK=topK_knncf)
        print("Finito")
        self.recommender.URM_train = self.URM_train

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        self.recommender._compute_item_score(user_id_array=user_id_array, items_to_compute=items_to_compute)
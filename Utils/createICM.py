import numpy as np
import pandas as pd
import scipy.sparse as sp

def createICM():
    types = pd.read_csv('../Input/data_ICM_type.csv')
    types = types.drop(columns=['data'], axis=1)
    types = types.rename(columns={'feature_id': 'type'})

    typesFiltered = types[types['item_id'] <= 24506]
    typesArray = typesFiltered['type'].to_numpy()
    itemsID = typesFiltered['item_id'].to_numpy()

    URM = pd.read_csv('../Input/interactions_and_impressions.csv')
    itemsURM = URM['ItemID'].to_numpy()
    itemsURM = np.unique(itemsURM, axis=0)

    ICM = np.zeros((len(itemsURM), 2), dtype=int)

    for x in range(len(itemsID)):
        ICM[itemsID[x]][0] = typesArray[x]

    lenght = pd.read_csv('../Input/data_ICM_length.csv')
    lenght = lenght.drop(columns=['feature_id'], axis=1)
    lenght = lenght.rename(columns={'data': 'numberOfEpisodes'})

    lenghtFiltered = lenght[lenght['item_id'] <= 24506]
    lenghtArray = lenghtFiltered['numberOfEpisodes'].to_numpy()
    itemsID = typesFiltered['item_id'].to_numpy()

    for x in range(len(itemsID)):
        ICM[itemsID[x]][1] = lenghtArray[x]

    ICM = sp.csr_matrix(ICM)

    return ICM
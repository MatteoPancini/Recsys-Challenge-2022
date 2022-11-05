import numpy as np
import pandas as pd
import scipy.sparse as sp
def createURM():

    dataset = pd.read_csv('../Input/interactions_and_impressions.csv')
    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    itemIDS = dataset['ItemID'].unique()

    URM = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(1)

    URM = sp.csr_matrix(URM)

    return URM

def newCreateURM():

    dataset = pd.read_csv('../Input/interactions_and_impressions.csv')
    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    # itemIDS = dataset['ItemID'].unique()

    newURM = np.zeros((len(userIDS), 27968), dtype=int)
    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            newURM[datasetCOO.row[x]][datasetCOO.col[x]] = int(1)

    newURM = sp.csr_matrix(newURM)

    return newURM
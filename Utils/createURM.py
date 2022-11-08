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
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
        elif URM[datasetCOO.row[x]][datasetCOO.col[x]] != int(5) and datasetCOO.data[x] == 1:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(1)

    URM = sp.csr_matrix(URM)

    return URM

def createURMFormDataset(dataset):
    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    itemIDS = dataset['ItemID'].unique()

    URM = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
        elif URM[datasetCOO.row[x]][datasetCOO.col[x]] != int(5) and datasetCOO.data[x] == 1:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(1)

    URM = sp.csr_matrix(URM)

    return URM
def createBumpURM(dataset):

    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    itemIDS = dataset['ItemID'].unique()

    URM = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
        elif datasetCOO.data[x] == 1 and URM[datasetCOO.row[x]][datasetCOO.col[x]] < 4:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = URM[datasetCOO.row[x]][datasetCOO.col[x]] + 1

    URM = sp.csr_matrix(URM)

    return URM

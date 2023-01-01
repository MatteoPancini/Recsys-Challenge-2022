import pandas as pd
import scipy.sparse as sp
import numpy as np

# path are:
# - ../../Input/ if run from notebooks and main Hybrid .py
# - ../../../Input/ if run from python file

urmPath = "../../Input/interactions_and_impressions.csv"
icmTypePath = "../Input/data_ICM_type.csv"
icmLenghtPath = "../Input/data_ICM_length.csv"
'''
urmPath = "../Input/interactions_and_impressions.csv"
icmTypePath = "../../Input/data_ICM_type.csv"
icmLenghtPath = "../../Input/data_ICM_length.csv"
'''
targetUserPath = "../../../Input/data_target_users_test.csv"

sourceDataset = '../Dataset/Our/'
binsourceDataset ='../../Dataset/'

def createURM():
    dataset = pd.read_csv(urmPath)

    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    itemIDS = dataset['ItemID'].unique()

    URM = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    ones_list = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            if URM[datasetCOO.row[x]][datasetCOO.col[x]] <= 4:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
            else:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(7)
        elif datasetCOO.data[x] == 1:
            if URM[datasetCOO.row[x]][datasetCOO.col[x]] == 7:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] == 0:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = 2
                ones_list[datasetCOO.row[x]][datasetCOO.col[x]] += 1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] >= 2 and ones_list[datasetCOO.row[x]][datasetCOO.col[x]] < 3:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] += 1
                ones_list[datasetCOO.row[x]][datasetCOO.col[x]] += 1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] > 1 and URM[datasetCOO.row[x]][datasetCOO.col[x]] != 5 and \
                    URM[datasetCOO.row[x]][datasetCOO.col[x]] != 7 and ones_list[datasetCOO.row[x]][
                datasetCOO.col[x]] >= 3:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = URM[datasetCOO.row[x]][datasetCOO.col[x]] - 1

    URM = sp.csr_matrix(URM)
    return URM

def createURMBinary():
    dataset = pd.read_csv(urmPath)

    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    itemIDS = dataset['ItemID'].unique()

    URM = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    for x in range(len(datasetCOO.data)):
        if (datasetCOO.data[x] == 0 or datasetCOO.data[x] == 1) and URM[datasetCOO.row[x]][datasetCOO.col[x]] != 1:
            URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(1)

    URM = sp.csr_matrix(URM)
    return URM

def createURMWithNegative():

    dataset = pd.read_csv(urmPath)

    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()
    itemIDS = dataset['ItemID'].unique()

    URM = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    ones_list = np.zeros((len(userIDS), len(itemIDS)), dtype=int)
    negative_users = np.zeros((len(userIDS), 2), dtype=int)

    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            if URM[datasetCOO.row[x]][datasetCOO.col[x]] <= 4:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
                if negative_users[datasetCOO.row[x]][0] == 1 and negative_users[datasetCOO.row[x]][1] == datasetCOO.col[
                    x]:
                    negative_users[datasetCOO.row[x]][0] = 0
            else:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(7)
        elif datasetCOO.data[x] == 1:
            if URM[datasetCOO.row[x]][datasetCOO.col[x]] == 7:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
            elif negative_users[datasetCOO.row[x]][0] == 0:
                negative_users[datasetCOO.row[x]][0] = 1
                negative_users[datasetCOO.row[x]][1] = datasetCOO.col[x]
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = -1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] == 0:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = 2
                ones_list[datasetCOO.row[x]][datasetCOO.col[x]] += 1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] >= 2 and ones_list[datasetCOO.row[x]][datasetCOO.col[x]] < 3:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] += 1
                ones_list[datasetCOO.row[x]][datasetCOO.col[x]] += 1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] > 1 and URM[datasetCOO.row[x]][datasetCOO.col[x]] != 5 and URM[datasetCOO.row[x]][datasetCOO.col[x]] != 7 and ones_list[datasetCOO.row[x]][datasetCOO.col[x]] >= 3:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = URM[datasetCOO.row[x]][datasetCOO.col[x]] - 1

    URM = sp.csr_matrix(URM)
    return URM
def createBigURM():
    dataset = pd.read_csv(urmPath)

    dataset = dataset.drop(columns=['Impressions'])

    datasetCOO = sp.coo_matrix((dataset["Data"].values, (dataset["UserID"].values, dataset["ItemID"].values)))
    userIDS = dataset['UserID'].unique()

    URM = np.zeros((len(userIDS), 27968), dtype=int)
    ones_list = np.zeros((len(userIDS), 27968), dtype=int)
    for x in range(len(datasetCOO.data)):
        if datasetCOO.data[x] == 0:
            if URM[datasetCOO.row[x]][datasetCOO.col[x]] <= 4:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
            else:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(7)
        elif datasetCOO.data[x] == 1:
            if URM[datasetCOO.row[x]][datasetCOO.col[x]] == 7:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = int(5)
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] == 0:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = 2
                ones_list[datasetCOO.row[x]][datasetCOO.col[x]] += 1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] >= 2 and ones_list[datasetCOO.row[x]][datasetCOO.col[x]] < 3:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] += 1
                ones_list[datasetCOO.row[x]][datasetCOO.col[x]] += 1
            elif URM[datasetCOO.row[x]][datasetCOO.col[x]] > 1 and URM[datasetCOO.row[x]][datasetCOO.col[x]] != 5 and \
                    URM[datasetCOO.row[x]][datasetCOO.col[x]] != 7 and ones_list[datasetCOO.row[x]][
                datasetCOO.col[x]] >= 3:
                URM[datasetCOO.row[x]][datasetCOO.col[x]] = URM[datasetCOO.row[x]][datasetCOO.col[x]] - 1

    URM = sp.csr_matrix(URM)
    return URM

def createSmallICM():

    types = pd.read_csv(icmTypePath)
    lenght = pd.read_csv(icmLenghtPath)

    types = types.drop(columns=['data'], axis=1)
    types = types.rename(columns={'feature_id': 'type'})

    typesFiltered = types[types['item_id'] <= 24506]
    typesArray = typesFiltered['type'].to_numpy()
    itemsID = typesFiltered['item_id'].to_numpy()

    ICM = np.zeros((24507, 2), dtype=int)

    for x in range(len(itemsID)):
        ICM[itemsID[x]][0] = typesArray[x]

    lenght = lenght.drop(columns=['feature_id'], axis=1)
    lenght = lenght.rename(columns={'data': 'numberOfEpisodes'})

    """lenghtFiltered = lenght[lenght['item_id'] <= 24506]
    lenghtArray = lenghtFiltered['numberOfEpisodes'].to_numpy()
    itemsID = typesFiltered['item_id'].to_numpy()

    for x in range(len(itemsID)):
        ICM[itemsID[x]][1] = lenghtArray[x]"""

    ICM = sp.csr_matrix(ICM)

    return ICM


def createBigICM():

    types = pd.read_csv(icmTypePath)
    lenght = pd.read_csv(icmLenghtPath)

    types = types.drop(columns=['data'], axis=1)
    types = types.rename(columns={'feature_id': 'type'})

    typesArray = types['type'].to_numpy()
    itemsID = types['item_id'].to_numpy()


    ICM = np.zeros((27968, 2), dtype=int)

    for x in range(len(itemsID)):
        ICM[itemsID[x]][0] = typesArray[x]


    lenght = lenght.drop(columns=['feature_id'], axis=1)
    lenght = lenght.rename(columns={'data': 'numberOfEpisodes'})

    lenghtArray = lenght['numberOfEpisodes'].to_numpy()

    for x in range(len(itemsID)):
        ICM[itemsID[x]][1] = lenghtArray[x]

    ICM = sp.csr_matrix(ICM)

    return ICM

def load_URMTrainInit():
    URM_train_init = sp.load_npz(sourceDataset+'URM_train_init.npz')
    return URM_train_init

def load_URMTest():
    URM_test = sp.load_npz(sourceDataset + 'URM_test.npz')
    return URM_test


def load_K_URMTrain():
    URM_train_list = []

    URM_train_list.append(sp.load_npz(sourceDataset + 'URM_train0.npz'))
    URM_train_list.append(sp.load_npz(sourceDataset + 'URM_train1.npz'))
    URM_train_list.append(sp.load_npz(sourceDataset + 'URM_train2.npz'))

    return URM_train_list

def load_K_URMValid():
    URM_val_list = []

    URM_val_list.append(sp.load_npz(sourceDataset + 'URM_val0.npz'))
    URM_val_list.append(sp.load_npz(sourceDataset + 'URM_val1.npz'))
    URM_val_list.append(sp.load_npz(sourceDataset + 'URM_val2.npz'))

    return URM_val_list


def load_BinURMTrainInit():
    URM_train_init = sp.load_npz(binsourceDataset+'URM_train_init.npz')
    return URM_train_init

def load_BinURMTest():
    URM_test = sp.load_npz(binsourceDataset + 'URM_test.npz')
    return URM_test


def load_3K_BinURMTrain():
    URM_train_list = []

    URM_train_list.append(sp.load_npz(binsourceDataset + 'URM_train0.npz'))
    URM_train_list.append(sp.load_npz(binsourceDataset + 'URM_train1.npz'))
    URM_train_list.append(sp.load_npz(binsourceDataset + 'URM_train2.npz'))

    return URM_train_list

def load_3K_BinURMValid():
    URM_val_list = []

    URM_val_list.append(sp.load_npz(binsourceDataset + 'URM_val0.npz'))
    URM_val_list.append(sp.load_npz(binsourceDataset + 'URM_val1.npz'))
    URM_val_list.append(sp.load_npz(binsourceDataset + 'URM_val2.npz'))

    return URM_val_list

def load_1K_BinURMTrain():
    URM_train_list = []

    URM_train_list.append(sp.load_npz(binsourceDataset + 'URM_train0.npz'))

    return URM_train_list

def load_1K_BinURMValid():
    URM_val_list = []

    URM_val_list.append(sp.load_npz(binsourceDataset + 'URM_val0.npz'))

    return URM_val_list
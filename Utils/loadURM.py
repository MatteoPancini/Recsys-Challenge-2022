# Utility class to load URM dataset and crop into a CSR sparse matrix format
#TODO: eliminare una volta fatto creato createURM.py

def load_URM(URM_path):

    import pandas as pd
    import scipy.sparse as sps

    dataset = pd.read_csv(URM_path)

    dataset = dataset.drop(dataset[dataset.Data != 0].index, inplace=True)

    dataset['Data'].replace(0, 1)

    #userID_unique = dataset["UserID"].unique()
    #itemID_unique = dataset["ItemID"].unique()

    URM = sps.coo_matrix((dataset["Data"].values,
                          (dataset["UserID"].values, dataset["ItemID"].values)))

    return URM.tocsr()
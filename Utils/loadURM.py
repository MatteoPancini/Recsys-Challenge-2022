# Utility class to load URM dataset and crop into a CSR sparse matrix format

def load_URM(URM_path):
    import pandas as pd
    import scipy.sparse as sps

    dataset = pd.read_csv(URM_path)

    user_list = dataset['UserID'].tolist()
    item_list = dataset['ItemID'].tolist()
    # impressions_list = dataset['Impressions'].tolist()
    data_list = dataset['Data'].tolist()

    return sps.coo_matrix((data_list, (user_list, item_list))).tocsr()
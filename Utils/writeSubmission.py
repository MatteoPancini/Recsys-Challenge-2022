# Utility method to automatically write in the right format the submission file

import tqdm

def write_submission(recommender, target_users_path, out_path):
    import pandas as pd
    import csv
    import numpy as np

    targetUsers = pd.read_csv(target_users_path)['user_id']

    targetUsers = targetUsers.tolist()

    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in tqdm.tqdm(targetUsers):
            writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])

def write_submission_List_Combination(recommender1, recommender2, target_users_path, out_path):
    import pandas as pd
    import csv
    import numpy as np

    targetUsers = pd.read_csv(target_users_path)['user_id']

    targetUsers = targetUsers.tolist()

    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in tqdm.tqdm(targetUsers):
            rec1 = recommender1.recommend(userID, 10)
            rec2 = recommender2.recommend(userID, 10)

            k = 5
            pos = 1
            recommendations = rec1[0:5]

            for i in range(5):
                if rec2[i] not in recommendations:
                    recommendations.insert(pos, rec2[i])
                else:
                    recommendations.insert(pos, rec1[k])
                    k = k + 1
                pos = pos + 2

            writer.writerow([userID, str(np.array(recommendations))[1:-1]])
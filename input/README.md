# Dataset Description
IMPORTANT: All files are comma-separated (columns are separated with ',' ). 
Also the submission file must be comma-separated.

## URM: Interactions and Impressions
interactions_and_impressions.csv : Contains the training set, describing implicit preferences expressed by the users.
+ user_id : identifier of the user
+ item_id : identifier of the item (TV series)
+ impression_list : string containing the items that were present on the screen when the user interacted with the item in column item_id. Not all interactions have a corresponding impressions list.
+ data : "0" if the user watched the item, "1" if the user opened the item details page.
Note that there are multiple interactions between the same user and item when a user watches multiple episodes of a TV series (if a user has watched 5 episodes there will be 5 interactions with that item_id).

## Data ICM Lenght
data_ICM_length.csv : 
Contains the number of episodes of the items. TV series may have multiple episodes.

+ item_id : identifier of the item
+ feature_id : identifier of the feature, only one value (0) exists since this ICM only contains the feature "length"
+ data : number of episodes. Some values may be 0 due to incomplete data.

## Data ICM Type
data_ICM_type.csv: 
Contains the type of the items. An item can only have one type (a type is for example the genre of the TV Series).
All types are anonymized and described only by a numerical identifier (from 1 to 4).
+ item_id : identifier of the item
+ feature_id : identifier of the type
+ data : "1" if the item is described by the type

## Data Target Users Test
data_target_users_test.csv: 
Contains the ids of the users that should appear in your submission file. 
The submission file should contain all and only these users.

## Sample Submission
sample_submission.csv:
A sample submission file in the correct format: [user_id],[ordered list of recommended items]. 
Be careful with the spaces and be sure to recommend the correct number of items to every user.
IMPORTANT: first line is mandatory and must be properly formatted.

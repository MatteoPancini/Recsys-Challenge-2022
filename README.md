# Recommender Systems 2022 Challenge
This repo contains the code and the data used in the [Recommender System 2022 Challenge](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi/overview) at Politecnico di Milano. All the algorithms are in the Recommenders folder and most of them are taken from the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), which contains basic implementations of many recommenders and utility code.

We ended up with the following placement:
1. <b>Public leaderboard:</b> 9/85
2. <b>Private leaderboard:</b> 9/85

## 👤 Team
+ Matteo Pancini
+ Michael Vitali

## ‼️ Requirements

In order to run the code it is necessary to have:
- <b>Python:</b> 3.9
- <b>Conda</b>

Install the python dependecies with the following bash command:
```
pip install -r requirements.txt
```

## 💾 Dataset
The dataset used is in the Input folder and it contains all the implicit interactions that each user made with the tv series. The type of interaction considered are two different:
1. User watch an episode, with multiple interaction for each episode.
2. User watch the information about the TV series.

### 🕵️ Data Pre-processing
For our final submission we created a weighted URM starting from the dataset of interactions and impressions. We created a binary URM using some bonus increment in the case of specific couple of interaction:
- If a user watch only a serie without watching the information page it might be that it liked a little more the serie and we used an increment of +0.25.
- If the user watch both information page and at least one episode it might be that the user liked the serie. So, we used an increment of +0.4.

All the functions used for the creation of the dataset are in the Utils folder. We splitted the dataset generated in three parts and we saved it to have the possibility to compare the tuning result of the different algorithms.

## 🏅 Best Model
Our best model puts together, using the Linear Combination technique, the three best algorithms that worked better on the dataset:
1. SLIM
2. RP3beta
3. MF_IALS

SLIM is the best alogorithm over our dataset and we firstly put together it with RP3beta. Then, the previously hybrid created was put together with the IALS algorithm. 
All the algorithms, both single or hybrid, were trained using the TPSE algorithm implemented by the [Optuna](https://optuna.readthedocs.io/en/stable/index.html) library to find the best hyperparameter.

## 💡 Tips
+ Always start by analyzing your data to have an overview of the characteristics of your dataset
+ Start to implement the algorithms seen during the classes using a very simple representation of the dataset -> at this moment you will have in mind which will be the best performing algorithms
+ By using the best performing, and possibly different algorithms try the first Hybrids. Note that adding to the hybrid also recommenders that poorly perform alone but are different from the others will boost the final hybrid a lot!
+ Play with different hybridization techniques
+ When you are happy enough or you just don't know how to go on try to work with your data (encode differently the URM, stack ICM(s), look if there are outliers, add bonus/malus to ratings)
+ Gradient boosting algorithms (XGBoost, CatBoost, LightFM) MAY be all you need! Take a look at the winners [Final Notebook](https://github.com/Benedart/RecSys-2022-Challenge-Polimi/blob/main/challenge_notebooks/xgboost_backup.ipynb)... reverse engineering will be worth!
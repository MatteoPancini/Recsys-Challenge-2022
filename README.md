# Recommender Systems 2022 Challenge
This repo contains the code and the data used in the [Recommender System 2022 Challenge](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi/overview) at Politecnico di Milano. All the algorithms are in the [Recommenders](https://github.com/MichaelVitali/Recsys-Challenge-2022/tree/main/Recommenders) folder and most of them are taken from the [course repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), which contains basic implementations of many recommenders and utility code.

We ended up with the following placement:
1. <b>Public leaderboard:</b> 9/85
2. <b>Private leaderboard:</b> 9/85

## ‼️ Requirements

In order to run the code it is necessary to have:
- <b>Python:</b> 3.9
- <b>Conda</b>

Install the python dependecies with the following bash command:
```
pip install -r requirements.txt
```

## 👤 Team
+ Matteo Pancini
+ Michael Vitali

## 💾 Dataset
The dataset used is in the folder [Input](https://github.com/MichaelVitali/Recsys-Challenge-2022/tree/main/Input) and it contains all the implicit interactions that each user made with the tv series. The type of interaction considered are two different:
1. User watch an episode, with multiple interaction for each episode.
2. User watch the information about the serie tv.

### 🕵️ Data Pre-processing
For our final submission we created a weighted URM starting from the dataset of interactions and impressions. We created a binary URM using some bonus increment in the case of specific couple of interaction:
- If a user watch only a serie without watching the information page it might be that it liked a little more the serie and we used an increment of +0.25.
- If the user watch both information page and at least one episode it might be that the user liked the serie. So, we used an increment of +0.4.

All the functions used for the creation of the dataset are in [Utils](https://github.com/MichaelVitali/Recsys-Challenge-2022/tree/main/Utils) folder. We splitted the dataset generated in three parts and we saved it to have the possibility to compare the tuning result of the different algorithms.

## 🏅 Best Model
Our best model puts together, using the Linear Combination technique, the three best algorithms that worked better on the dataset:
1. [SLIM](https://github.com/MichaelVitali/Recsys-Challenge-2022/blob/main/Recommenders/SLIM/SLIMElasticNetRecommender.py)
2. [RP3beta](https://github.com/MichaelVitali/Recsys-Challenge-2022/blob/main/Recommenders/GraphBased/RP3betaRecommender.py)
3. [MF_IALS](https://github.com/MichaelVitali/Recsys-Challenge-2022/blob/main/Recommenders/Implicit/ImplicitALSRecommender.py)

SLIM is the best alogorithm over our dataset and we firstly put together it with RP3beta. Then, the previously hybrid created was put together with the IALS algorithm. 
All the algorithms, both single or hybrid, were trained using the TPE algorithm implemented by the [Optuna](https://optuna.readthedocs.io/en/stable/index.html) library to find the best hyperparameter.
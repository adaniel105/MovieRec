import os
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

HOME = os.getcwd()

data = fetch_movielens(genre_features=True)
movies = pd.read_csv(f"{HOME}/data/movies.csv")
movie_titles = movies["title"].str.lower().copy()
train = data["train"]

model = LightFM(learning_rate=0.05, loss="warp")
model.fit(train, epochs=10, num_threads=2)


class LightFMRecommender:
    def __init__(self):
        pass

    def create_rec(self, movie_name, number_of_recommend):
        try:
            self.model = model
            self.movie_titles = movie_titles
            self.movie_name = movie_name
            self.number_of_recommend = number_of_recommend
            similar_ids = self.get_similar_ids(model, movie_name, number_of_recommend)
            recommendations = self.get_instances(model, similar_ids, movie_titles)
            return recommendations
        except:
            print("Movie not found!")

    def get_similar_ids(self, model, movie_name, number_of_recommend):
        similar_ids = movie_titles.index[
            movie_titles.str.contains(str(movie_name).lower())
        ].tolist()
        similar_ids = similar_ids[0]

        movie_embed = (
            model.item_embeddings.T / np.linalg.norm(model.item_embeddings, axis=1)
        ).T

        query_embed = movie_embed[similar_ids]
        similarity = np.dot(movie_embed, query_embed)
        end = number_of_recommend + 1
        most_similar = np.argsort(-similarity)[1:end]

        return most_similar

    def get_instances(self, model, similar_ids, movie_titles):
        similar_movies = movie_titles[
            self.get_similar_ids(model, similar_ids, self.number_of_recommend)
        ]
        return similar_movies.tolist()

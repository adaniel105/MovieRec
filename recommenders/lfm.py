import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from data import data_preprocessing

movies = data_preprocessing.movies
movie_titles = data_preprocessing.movie_titles
data = fetch_movielens(genre_features=True)
train = data["train"]

model = LightFM(learning_rate=0.05, loss="warp")
model.fit(train, epochs=10, num_threads=2)

'''
WIP: Add LightFM Recommender
'''

class LightFMRecommender:
    def __init__(self):
        pass

    def create_rec(self, movie_name: str, number_of_recommend: int) -> list[str]:
        try:
            self.model = model
            self.movie_titles = movies["title"]
            self.movie_name = movie_name
            self.number_of_recommend = number_of_recommend
            similar_ids = self.get_similar_ids(model, movie_name, number_of_recommend)
            recommendations = self.get_instances(model, similar_ids, self.movie_titles)
            return recommendations
        except Exception:
            err_msg = ["Movie not found!"]
            return err_msg

    def get_similar_ids(
        self, model, movie_name: str, number_of_recommend: int
    ) -> list[int]:
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

    def get_instances(
        self, model, similar_ids: list[int], movie_titles: list[str]
    ) -> list[str]:
        similar_movies = movie_titles[
            self.get_similar_ids(model, similar_ids, self.number_of_recommend)
        ]
        return similar_movies.tolist()

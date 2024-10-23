import os
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


HOME = os.getcwd()


class KNNRecommender:
    def __init__(self):
        pass

    def create_rec(self, movie_name: str, number_of_recommend: int) -> list[str]:
        try:
            ratings, movies, movie_titles, idd_ = self.get_instances(movie_name)
            self.movie_name = movie_name
            self.movies = movies
            self.movie_titles = movie_titles
            self.ratings = ratings
            self.idd_ = idd_
            self.number_of_recommend = number_of_recommend

            (
                matrix_crs,
                user_mapper,
                self.movie_mapper,
                self.user_inv_mapper,
                self.movie_inv_mapper,
            ) = self.matrix(self.ratings)
            neighbour_ids = self.predict_(self.idd_, matrix_crs, k=number_of_recommend)
            recommendations = self.recommend(neighbour_ids, movie_titles)
            return recommendations
        except Exception:
            err_msg = ["Movie not found!"]
            return err_msg

    def matrix(self, df: pd.DataFrame):

        user_unique = len(df["userId"].unique())
        movie_unique = len(df["movieId"].unique())

        # Map Ids to indices
        user_mapper = dict(zip(np.unique(df["userId"]), list(range(user_unique))))
        movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(movie_unique))))

        # Map indices to IDs
        user_inv_mapper = dict(zip(list(range(user_unique)), np.unique(df["userId"])))
        movie_inv_mapper = dict(
            zip(list(range(movie_unique)), np.unique(df["movieId"]))
        )

        user_index = [user_mapper[i] for i in df["userId"]]
        movie_index = [movie_mapper[i] for i in df["movieId"]]

        matrix_crs = csr_matrix(
            (df["rating"], (movie_index, user_index)), shape=(movie_unique, user_unique)
        )

        return matrix_crs, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def predict_(
        self,
        movie_id: int,
        data: pd.DataFrame,
        k: int,
        metric="cosine",
        show_distance=False,
    ):

        neighbour_ids = []

        movie_ind = self.movie_mapper[movie_id]
        movie_vec = data[movie_ind]
        k += 1
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(data)
        movie_vec = movie_vec.reshape(1, -1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(self.movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids

    def get_instances(self, movie_name: str):

        ratings = pd.read_csv(f"{HOME}/data/ratings.csv")
        movies = pd.read_csv(f"{HOME}/data/movies.csv")
        movie_titles = dict(zip(movies["movieId"], movies["title"]))
        movie_table = pd.DataFrame(
            movie_titles.items(), columns=["movie_id", "movie_name"]
        )
        movie_table["movie_name"] = movie_table["movie_name"].str.lower()
        idd_ = pd.DataFrame(
            movie_table.movie_id.where(
                movie_table["movie_name"].str.contains(str(movie_name).lower())
            )
        )
        idd_ = int(idd_[idd_.movie_id.notna()]["movie_id"].iloc[0])
        return ratings, movies, movie_titles, idd_

    def recommend(self, similar_ids: list[int], movie_titles: list[str]):
        return [movie_titles[i] for i in similar_ids]

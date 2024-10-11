import pandas as pd
import os, re

HOME = os.getcwd()
movies = pd.read_csv(f"{HOME}/data/movies.csv")
movie_titles = movies["title"].str.lower().copy()


movies["year"] = 0


def split_title_and_yr(df):
    for title in movies["title"]:
        release_date = re.findall(r"\(\s*\+?(-?\d+)\s*\)", title)
        row_idx = movies.index[movies["title"] == title].tolist()[0]
        for m in release_date:
            if int(m) >= 1990:
                movies["year"][row_idx] = int(m)  # refactor with a nicer pd function

        movies["title"][row_idx] = movies["title"][row_idx].split("(")[0].strip()

import pandas as pd
import re

movies = pd.read_csv("movies.csv")
movies["year"] = 0


def split_title_and_yr(df):
    for title in movies["title"]:
        release_date = re.findall(r"\(\s*\+?(-?\d+)\s*\)", title)
        row_idx = movies.index[movies["title"] == title].tolist()[0]
        for m in release_date:
            if int(m) >= 1990:
                movies["year"][row_idx] = int(m)  # refactor with a nicer pd function

        movies["title"][row_idx] = movies["title"][row_idx].split("(")[0].strip()


split_title_and_yr(movies)

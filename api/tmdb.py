import os
import requests
from dotenv import load_dotenv

# from data_preprocessing import movies


load_dotenv()
READ_ACCESS_TOKEN = os.getenv("READ_ACCESS_TOKEN")

# year = movies["year"].loc[movies["title"] == title].tolist()[0]


def fetch_metadata(title: str = "Jumanji"):
    url = f"https://api.themoviedb.org/3/search/tv?query={title}&include_adult=false&language=en-US&page=1"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {READ_ACCESS_TOKEN}",
    }

    response = requests.get(url, headers=headers)
    response = response.json()
    overview = response["results"][0]["overview"]
    return overview

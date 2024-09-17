import os
import requests
from dotenv import load_dotenv

# from data_preprocessing import movies


load_dotenv()
READ_ACCESS_TOKEN = os.getenv("READ_ACCESS_TOKEN")

title = "Jumanji"
# year = movies["year"].loc[movies["title"] == title].tolist()[0]

url = f"https://api.themoviedb.org/3/search/tv?query={title}&include_adult=false&language=en-US&page=1"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {READ_ACCESS_TOKEN}",
}

response = requests.get(url, headers=headers)

response = response.json()


print(response["results"][0]["overview"])

import requests

# from data_preprocessing import movies


title = "Money Train"
# year = movies["year"].loc[movies["title"] == title].tolist()[0]

url = f"https://api.themoviedb.org/3/search/tv?query={title}&include_adult=false&language=en-US&page=1"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4ZTA5MjYyZTZmYTIxZmFkODA0MDExZWNkNDI1MmNiOCIsIm5iZiI6MTcyNjUzODY5MS43NDYxMjgsInN1YiI6IjY2ZTM2NzA3MDAwMDAwMDAwMGI5NWMyNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.c3rnjOkBEML5gVC1fYyiJQvdwmtn5GDTT5amBZ6-iSo",
}

response = requests.get(url, headers=headers)

response = response.json()


response["results"][0]["overview"]

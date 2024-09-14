from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from recommenders.knn import MovieRecommender
from dotenv import load_dotenv
import os
import requests

load_dotenv()
READ_ACCESS_TOKEN = os.getenv("READ_ACCESS_TOKEN")

static_folder = Path(__file__).parent.resolve(strict=True) / "static"
template_folder = Path(__file__).parent.resolve(strict=True) / "templates"

url = "https://api.themoviedb.org/3/find/tt11280740?external_source=imdb_id&language=en"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {READ_ACCESS_TOKEN}",
}
response = requests.get(url, headers=headers)
response = response.json()

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(static_folder)), name="static")
templates = Jinja2Templates(directory=str(template_folder))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommend")
async def recommend(
    request: Request, movie_name: str = Form(...), number_of_recommend: int = 5
):
    recommendations = MovieRecommender()
    movie_list = recommendations.create_rec(
        movie_name=movie_name, number_of_recommend=number_of_recommend
    )
    movie_name = response["tv_results"][0]["name"]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "movie_list": movie_list, "movie_name": movie_name},
    )

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from recommenders.knn import MovieRecommender

static_folder = Path(__file__).parent.resolve(strict=True) / "static"
template_folder = Path(__file__).parent.resolve(strict=True) / "templates"


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(static_folder)), name="static")
templates = Jinja2Templates(directory=str(template_folder))


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request, movie_name: str = "Spider-man", number_of_recommend: int = 7
):
    recommendations = MovieRecommender()
    movie_list = recommendations.create_rec(
        movie_name=movie_name, number_of_recommend=number_of_recommend
    )
    return templates.TemplateResponse(
        "index.html", {"request": request, "movie_list": movie_list}
    )

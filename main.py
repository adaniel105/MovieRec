from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from recommenders import knn, lfm


static_folder = Path(__file__).parent.resolve(strict=True) / "static"
template_folder = Path(__file__).parent.resolve(strict=True) / "templates"


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(static_folder)), name="static")
templates = Jinja2Templates(directory=str(template_folder))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommend")
async def recommend(
    request: Request,
    movie_name: str = Form(...),
    model=Form(...),
    number_of_recommend: int = 5,
):
    model = lfm.LightFMRecommender() if model == "lightfm" else knn.KNNRecommender()

    movie_list = model.create_rec(
        movie_name=movie_name, number_of_recommend=number_of_recommend
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "movie_list": movie_list,
            "movie_name": movie_name,
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
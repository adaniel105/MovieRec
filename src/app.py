from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

static_folder = Path(__file__).parent.resolve(strict=True) / "static"
template_folder = Path(__file__).parent.resolve(strict=True) / "templates"


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(static_folder)), name="static")
templates = Jinja2Templates(directory=str(template_folder))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "name": "recommender"}
    )

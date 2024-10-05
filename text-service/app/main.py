from fastapi import FastAPI

from app.api.texts_compare import compareapp

app = FastAPI()

app.include_router(compareapp)
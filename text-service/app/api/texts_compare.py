from fastapi import APIRouter, File, UploadFile
import pandas as pd
from app.api.compare import text_similarity

compareapp = APIRouter()

@compareapp.post('/compare')
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    #Чтение содержимого файлов
    contents1 = await file1.read()
    contents2 = await file2.read()
    #Преобразуем содержимое в строки
    text1 = contents1.decode("utf-8")
    text2 = contents2.decode("utf-8")
    #Вычисляем корреляционную матрицу
    similarity = text_similarity(text1, text2)
    return pd.DataFrame(similarity.todense())
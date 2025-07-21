# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:15:57 2024

@author: Br1CM

"""

import os
import warnings
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from chatbot_functions import compile_pipeline
from excercise_classification import (retrieve_data_video, pipeline_landmarks_to_excercise_knn, squat_prompt, 
                                      kpis_excercise, pullup_prompt, dip_prompt, pushup_prompt, unknown_excercise)
from datetime import datetime
import shutil

workpath = os.getcwd()
warnings.filterwarnings("ignore")
workflow = compile_pipeline()

# Crear app FastAPI
app = FastAPI()


@app.post("/ask_agent")
async def ask_agent(request: Request):
    data = await request.json()
    question = data.get('content')
    if not question:
        raise HTTPException(status_code=400, detail="No content provided")

    # add message to the RAG workflow
    inputs = {"message": question} 
    for output in app.stream(inputs):
        for key, value in output.items():
            response = value
    answer = value['generation']    
    response = {'user': 'Agent', 'content': answer}
    return JSONResponse(content=response)

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    video_folder = os.path.join(workpath, 'Videos')
    os.makedirs(video_folder, exist_ok=True)
    video_filename = 'VID' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".mp4"
    video_path = os.path.join(video_folder, video_filename)
    # Save the uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    video.file.close()
    # Process the video
    landmarks, angles, distances = retrieve_data_video(video_path)
    excercise, kpis = pipeline_landmarks_to_excercise_knn(landmarks, angles, distances)
    clean_kpis = kpis_excercise(kpis)
    if excercise == 'Sentadilla':
        respuesta_a_video = squat_prompt(clean_kpis)
    elif excercise == 'Dominada':
        respuesta_a_video = pullup_prompt(clean_kpis)
    elif excercise == 'Fondo':
        respuesta_a_video = dip_prompt(clean_kpis)
    elif excercise == 'Flexión':
        respuesta_a_video = pushup_prompt(clean_kpis)
    else:
        respuesta_a_video = unknown_excercise()
    return JSONResponse(content={'user': 'Agent', 'content': respuesta_a_video})

# To run: uvicorn projectAPI:app --host 0.0.0.0 --port 8000
    

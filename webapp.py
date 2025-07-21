# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:52:33 2024

@author: Br1CM
"""

import os
import streamlit as st
from chatbot_functions import compile_pipeline
from excercise_classification import (retrieve_data_video_show_st, pipeline_landmarks_to_excercise_knn, squat_prompt, 
                                      kpis_excercise, pullup_prompt, dip_prompt, pushup_prompt, unknown_excercise)

workpath = os.getcwd()

# get the chatbot workflow up
app = compile_pipeline()

# streamlit layout
st.logo("Images/FotoCalistenIA.png")
st.image('Images/logoCalistenIA.png', width=125)
st.set_page_config(layout="wide")



# divide the webapp in two columns
col1, col2 = st.columns([6,4])

# conversation window
with col1:
    st.header("Talk to Swai, the calisthenics chatbot")
    # Initialize chat history

    chatbox = st.container(height=300)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show historical chat
    with chatbox:
        for message in st.session_state.messages:
            st.chat_message(message['role']).write(message['content'])
        
    # User input

    if prompt := st.chat_input('¿En qué quieres que te ayude amigo calisténico?'):
        # show user's message
        with chatbox:
            st.chat_message('user').write(prompt)
        # add user's message to history
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # add message to the RAG workflow
        inputs = {"message": prompt} 
        for output in app.stream(inputs):
            for key, value in output.items():
                response = value
        answer = value['generation']
        # show LLM's answer
        with chatbox:
            st.chat_message("assistant").write(answer)
        # add LLM's message to history
        st.session_state.messages.append({'role': 'assistant', 'content': answer})

# Video processing window
with col2:
    st.header("Show me your moves!")
    # video input
    video_container_height = 300
    video_frame = st.container(height=video_container_height)
    
    # container to show the video processing
    with video_frame:
        hueco_video = st.empty()
        
    # Folder to save user's video
    video_folder = os.path.join(workpath, 'Videos')

    # upload video button
    video = st.file_uploader("Sube tu vídeo haciendo ejercicio aquí:", type = ["MP4", "MOV", "MPEG", "GIF"],
                             accept_multiple_files=False)
    
    # processing video
    if video is not None:
        # read the file in bytes
        video_bytes = video.read()
        
        video_path = os.path.join(video_folder, video.name)  # absolute path of the vide
        
        # write and save the video
        with open(video_path, "wb") as f:
            f.write(video_bytes)
    
        # show video on web and process
        landmarks, angles, distances = retrieve_data_video_show_st(video_path, hueco_video, video_container_height)
        excercise, kpis = pipeline_landmarks_to_excercise_knn(landmarks, angles, distances)
        clean_kpis = kpis_excercise(kpis)

        # depending on the excercise predicted, study the video and give an answer
        if excercise == 'Sentadilla':
            respuesta_a_video = squat_prompt(clean_kpis)
            # add LLM's message to history
            st.session_state.messages.append({'role': 'assistant', 'content': respuesta_a_video})
            with chatbox:
                # show LLM's answer in the chat
                st.chat_message("assistant").write(respuesta_a_video)
        elif excercise == 'Dominada':
            respuesta_a_video = pullup_prompt(clean_kpis)
            # add LLM's message to history
            st.session_state.messages.append({'role': 'assistant', 'content': respuesta_a_video})
            with chatbox:
                # show LLM's answer in the chat
                st.chat_message("assistant").write(respuesta_a_video)
        elif excercise == 'Fondo':
            respuesta_a_video = dip_prompt(clean_kpis)
            # add LLM's message to history
            st.session_state.messages.append({'role': 'assistant', 'content': respuesta_a_video})
            with chatbox:
                # show LLM's answer in the chat
                st.chat_message("assistant").write(respuesta_a_video)
        elif excercise == 'Flexión':
            respuesta_a_video = pushup_prompt(clean_kpis)
            # add LLM's message to history
            st.session_state.messages.append({'role': 'assistant', 'content': respuesta_a_video})
            with chatbox:
                # show LLM's answer in the chat
                st.chat_message("assistant").write(respuesta_a_video)
        else:
            respuesta_a_video = unknown_excercise()
            # add LLM's message to history
            st.session_state.messages.append({'role': 'assistant', 'content': respuesta_a_video})
            with chatbox:
                # show LLM's answer in the chat
                st.chat_message("assistant").write(respuesta_a_video)
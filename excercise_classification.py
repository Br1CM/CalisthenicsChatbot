# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:55:59 2024

@author: Br1CM
"""
import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from collections import Counter
from ollama import Client
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
workpath = os.getcwd()
import warnings
warnings.filterwarnings("ignore")

# Create a connection to the localized LLM 
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:11434")
client = Client(host=LLAMA_SERVER_URL)

#------ FUNCTIONS FOR EXERCISE CLASSIFICATION BASED ON VIDEO AND EFFECTIVELY ASSESSMENT ON THE EXCERCISE WITH LLM--------


def create_angle(a, b, c):
    """
    Given 3 points in the space, calculate the angle created by the AB line and the BC line.

    Args:
        a: List
        Point A
        b: List
        Point B
        c: List
        Point C
        
    Returns:
        float: the angle in degrees created (theta in [0, 180])
    """
    A = np.array(a)
    B = np.array(b)
    C = np.array(c)
    # Vectors BA y BC
    BA = A - B
    BC = C - B
    # Dot product between vectors BA y BC
    dot_product = np.dot(BA, BC)
    # Vector norms BA y BC
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    # Angle between vectors' cosine
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    # Angle in radians
    theta_radians = np.arccos(cos_theta)
    # Angle in degrees
    theta_degrees = np.degrees(theta_radians)
    if 180 <= theta_degrees <= 360:
        theta_degrees_good = 360 - theta_degrees
    else: 
        theta_degrees_good = theta_degrees
    return theta_degrees_good

def create_distance(A, B):
    """
    Given 2 points in the space, calculate the euclidean distance between them

    Args:
        a: List
        Point A [A_1, A_2, A_3]
        b: List
        Point B [B_1, B_2, B_3]
        
        
    Returns:
        float: distance 
    """
    # Unpack coordinates
    x1, y1, z1 = A
    x2, y2, z2 = B
    
    # Calcular la distancia usando la fórmula euclidiana
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return distance

def retrieve_data_video_show_st(path, video_frame, container_height):
    """
    Given a path to a video, pass it through cv model and get the landmarks data.
    At the same time, lets the user see the video processing.

    Args:
        path: str
        Path to the video file
        video_frame: Obj
        container from which it would be visualized in the app
        container height: int
        height of the container
        
    Returns:
        landmarks: List
        Raw data from each frames' landmark
        angles: Dict
        Angles created by different landmarks
        distances: Dict
        Distances between different landmarks
    """
    cap = cv2.VideoCapture(path)
    landmarks_raw = []
    angles = {'elbow_l':[], 'elbow_r':[],
              'shoulder_l': [], 'shoulder_r':[],
              'hip_l': [], 'hip_r': [],
              'knee_l': [], 'knee_r': []}
    distances = {'wrists': [], 'elbows': [], 
                 'shoulders': [], 'hips': [],
                 'knees': [], 'ankles': []}
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.8 , min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
        
            # Recolor image to RGB
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                try:
                    landmarks_frame = results.pose_landmarks.landmark
                    landmarks_raw.append(landmarks_frame)
                    left_wrist = [landmarks_frame[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                    left_elbow = [landmarks_frame[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                    left_shoulder = [landmarks_frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                     landmarks_frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                    left_hip = [landmarks_frame[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                landmarks_frame[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                    left_knee = [landmarks_frame[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                 landmarks_frame[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                    left_ankle = [landmarks_frame[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                    right_wrist = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                    right_elbow = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                    right_shoulder = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                     landmarks_frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                    right_hip = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                 landmarks_frame[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                    right_knee = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                    right_ankle = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                   landmarks_frame[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                    
                    angles['elbow_l'].append(create_angle(left_wrist, left_elbow, left_shoulder))
                    angles['elbow_r'].append(create_angle(right_wrist, right_elbow, right_shoulder))
                    angles['shoulder_l'].append(create_angle(left_elbow, left_shoulder, left_hip))
                    angles['shoulder_r'].append(create_angle(right_elbow, right_shoulder, right_hip))
                    angles['hip_l'].append(create_angle(left_shoulder, left_hip, left_knee))
                    angles['hip_r'].append(create_angle(right_shoulder, right_hip, right_knee))
                    angles['knee_l'].append(create_angle(left_hip, left_knee, left_ankle))
                    angles['knee_r'].append(create_angle(right_hip, right_knee, right_ankle))
                    distances['wrists'].append(create_distance(left_wrist, right_wrist))
                    distances['elbows'].append(create_distance(left_elbow, right_elbow))
                    distances['shoulders'].append(create_distance(left_shoulder, right_shoulder))
                    distances['hips'].append(create_distance(left_hip, right_hip))
                    distances['knees'].append(create_distance(left_knee, right_knee))
                    distances['ankles'].append(create_distance(left_ankle, right_ankle))
                except Exception as e:
                    print(e)
                    pass
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(148,7,240), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(149,158,0), thickness=2, circle_radius=2) 
                                         )               
        
                # ancho frame
                vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # Obtiene el alto del frame
                vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if vid_width > vid_height:
                    video_frame.image(image, channels='BGR', use_column_width=True)
                elif vid_width < vid_height:
                    video_frame.image(image, channels='BGR', width = int((container_height/vid_height)*vid_width)-10)
                if cv2.waitKey(25) == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    return landmarks_raw, angles, distances

def retrieve_data_video(path):
    """
    Given a path to a video, pass it through cv model and get the landmarks data.

    Args:
        path: str
        Path to the video file
        
    Returns:
        landmarks: List
        Raw data from each frames' landmark
        angles: Dict
        Angles created by different landmarks
        distances: Dict
        Distances between different landmarks
    """
    cap = cv2.VideoCapture(path)
    landmarks_raw = []
    angles = {'elbow_l':[], 'elbow_r':[],
              'shoulder_l': [], 'shoulder_r':[],
              'hip_l': [], 'hip_r': [],
              'knee_l': [], 'knee_r': []}
    distances = {'wrists': [], 'elbows': [], 
                 'shoulders': [], 'hips': [],
                 'knees': [], 'ankles': []}
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.7 , min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
        
            # Recolor image to RGB
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
          
                # Make detection
                results = pose.process(image)
                try:
                    landmarks_frame = results.pose_landmarks.landmark
                    landmarks_raw.append(landmarks_frame)
                    left_wrist = [landmarks_frame[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                    left_elbow = [landmarks_frame[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                    left_shoulder = [landmarks_frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                     landmarks_frame[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                    left_hip = [landmarks_frame[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                landmarks_frame[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                    left_knee = [landmarks_frame[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                 landmarks_frame[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                    left_ankle = [landmarks_frame[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks_frame[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                    right_wrist = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                    right_elbow = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                    right_shoulder = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                     landmarks_frame[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                    right_hip = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                 landmarks_frame[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                    right_knee = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                  landmarks_frame[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                    right_ankle = [landmarks_frame[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks_frame[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                   landmarks_frame[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                    
                    angles['elbow_l'].append(create_angle(left_wrist, left_elbow, left_shoulder))
                    angles['elbow_r'].append(create_angle(right_wrist, right_elbow, right_shoulder))
                    angles['shoulder_l'].append(create_angle(left_elbow, left_shoulder, left_hip))
                    angles['shoulder_r'].append(create_angle(right_elbow, right_shoulder, right_hip))
                    angles['hip_l'].append(create_angle(left_shoulder, left_hip, left_knee))
                    angles['hip_r'].append(create_angle(right_shoulder, right_hip, right_knee))
                    angles['knee_l'].append(create_angle(left_hip, left_knee, left_ankle))
                    angles['knee_r'].append(create_angle(right_hip, right_knee, right_ankle))
                    distances['wrists'].append(create_distance(left_wrist, right_wrist))
                    distances['elbows'].append(create_distance(left_elbow, right_elbow))
                    distances['shoulders'].append(create_distance(left_shoulder, right_shoulder))
                    distances['hips'].append(create_distance(left_hip, right_hip))
                    distances['knees'].append(create_distance(left_knee, right_knee))
                    distances['ankles'].append(create_distance(left_ankle, right_ankle))
                except Exception as e:
                    print(e)
                    pass
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if cv2.waitKey(25) == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    return landmarks_raw, angles, distances

def landmarks_to_dataframe(rawdata, angles, distances):
    """
    Given the raw data retrieved from the video, get it in the structure of a DataFrame

    Args:
        rawdata: List
        Landmark's raw data
        angles: Dict
        Angles created by different landmarks
        distances: Dict
        Distances between different landmarks
    
    Returns:
        pd.DataFrame: DataFrame containing the landmarks position data (each row represents a frame of the video)
        pd.DataFrame: DataFrame containing the excercise angles and distances (each row represents a frame of the video)
    """
    landmarks_dict = {
    'NOSE_x' : [],
    'NOSE_y' : [],
    'NOSE_z' : [],
    'LEFT_SHOULDER_x' : [],
    'LEFT_SHOULDER_y' : [],
    'LEFT_SHOULDER_z' : [],
    'RIGHT_SHOULDER_x' : [],
    'RIGHT_SHOULDER_y' : [],
    'RIGHT_SHOULDER_z' : [],
    'LEFT_ELBOW_x' : [],
    'LEFT_ELBOW_y' : [],
    'LEFT_ELBOW_z' : [],
    'RIGHT_ELBOW_x' : [],
    'RIGHT_ELBOW_y' : [],
    'RIGHT_ELBOW_z' : [],
    'LEFT_WRIST_x' : [],
    'LEFT_WRIST_y' : [],
    'LEFT_WRIST_z' : [],
    'RIGHT_WRIST_x' : [],
    'RIGHT_WRIST_y' : [],
    'RIGHT_WRIST_z' : [],
    'LEFT_PINKY_x' : [],
    'LEFT_PINKY_y' : [],
    'LEFT_PINKY_z' : [],
    'RIGHT_PINKY_x' : [],
    'RIGHT_PINKY_y' : [],
    'RIGHT_PINKY_z' : [],
    'LEFT_INDEX_x' : [],
    'LEFT_INDEX_y' : [],
    'LEFT_INDEX_z' : [],
    'RIGHT_INDEX_x' : [],
    'RIGHT_INDEX_y' : [],
    'RIGHT_INDEX_z' : [],
    'LEFT_THUMB_x' : [],
    'LEFT_THUMB_y' : [],
    'LEFT_THUMB_z' : [],
    'RIGHT_THUMB_x' : [],
    'RIGHT_THUMB_y' : [],
    'RIGHT_THUMB_z' : [],
    'LEFT_HIP_x' : [],
    'LEFT_HIP_y' : [],
    'LEFT_HIP_z' : [],
    'RIGHT_HIP_x' : [],
    'RIGHT_HIP_y' : [],
    'RIGHT_HIP_z' : [],
    'LEFT_KNEE_x' : [],
    'LEFT_KNEE_y' : [],
    'LEFT_KNEE_z' : [],
    'RIGHT_KNEE_x' : [],
    'RIGHT_KNEE_y' : [],
    'RIGHT_KNEE_z' : [],
    'LEFT_ANKLE_x' : [],
    'LEFT_ANKLE_y' : [],
    'LEFT_ANKLE_z' : [],
    'RIGHT_ANKLE_x' : [],
    'RIGHT_ANKLE_y' : [],
    'RIGHT_ANKLE_z' : [],
    'LEFT_ELBOW_ANGLE': [],
    'RIGHT_ELBOW_ANGLE': [],
    'LEFT_SHOULDER_ANGLE': [],
    'RIGHT_SHOULDER_ANGLE': [],
    'LEFT_HIP_ANGLE': [],
    'RIGHT_HIP_ANGLE': [],
    'LEFT_KNEE_ANGLE': [],
    'RIGHT_KNEE_ANGLE': []
    }
    
    excercise_kpi = {'WRISTS_DISTANCE': [],
    'ELBOWS_DISTANCE': [],
    'SHOULDERS_DISTANCE': [],
    'HIPS_DISTANCE': [],
    'KNEES_DISTANCE': [],
    'ANKLES_DISTANCE': [],
    'LEFT_ELBOW_ANGLE': [],
    'RIGHT_ELBOW_ANGLE': [],
    'LEFT_SHOULDER_ANGLE': [],
    'RIGHT_SHOULDER_ANGLE': [],
    'LEFT_HIP_ANGLE': [],
    'RIGHT_HIP_ANGLE': [],
    'LEFT_KNEE_ANGLE': [],
    'RIGHT_KNEE_ANGLE': []
        }
    for index in range(len(rawdata)):
        landmarks_dict['NOSE_x'].append(rawdata[index][mp_pose.PoseLandmark.NOSE.value].x)
        landmarks_dict['NOSE_y'].append(1-rawdata[index][mp_pose.PoseLandmark.NOSE.value].y)
        landmarks_dict['NOSE_z'].append(rawdata[index][mp_pose.PoseLandmark.NOSE.value].z)
        landmarks_dict['LEFT_SHOULDER_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)
        landmarks_dict['LEFT_SHOULDER_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        landmarks_dict['LEFT_SHOULDER_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_SHOULDER.value].z)
        landmarks_dict['RIGHT_SHOULDER_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
        landmarks_dict['RIGHT_SHOULDER_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        landmarks_dict['RIGHT_SHOULDER_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z)
        landmarks_dict['LEFT_ELBOW_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_ELBOW.value].x)
        landmarks_dict['LEFT_ELBOW_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
        landmarks_dict['LEFT_ELBOW_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_ELBOW.value].z)
        landmarks_dict['RIGHT_ELBOW_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_ELBOW.value].x)
        landmarks_dict['RIGHT_ELBOW_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
        landmarks_dict['RIGHT_ELBOW_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_ELBOW.value].z)
        landmarks_dict['LEFT_WRIST_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_WRIST.value].x)
        landmarks_dict['LEFT_WRIST_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_WRIST.value].y)
        landmarks_dict['LEFT_WRIST_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_WRIST.value].z)
        landmarks_dict['RIGHT_WRIST_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_WRIST.value].x)
        landmarks_dict['RIGHT_WRIST_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
        landmarks_dict['RIGHT_WRIST_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_WRIST.value].z)
        landmarks_dict['LEFT_PINKY_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_PINKY.value].x)
        landmarks_dict['LEFT_PINKY_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_PINKY.value].y)
        landmarks_dict['LEFT_PINKY_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_PINKY.value].z)
        landmarks_dict['RIGHT_PINKY_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_PINKY.value].x)
        landmarks_dict['RIGHT_PINKY_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_PINKY.value].y)
        landmarks_dict['RIGHT_PINKY_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_PINKY.value].z)
        landmarks_dict['LEFT_INDEX_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_INDEX.value].x)
        landmarks_dict['LEFT_INDEX_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_INDEX.value].y)
        landmarks_dict['LEFT_INDEX_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_INDEX.value].z)
        landmarks_dict['RIGHT_INDEX_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_INDEX.value].x)
        landmarks_dict['RIGHT_INDEX_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_INDEX.value].y)
        landmarks_dict['RIGHT_INDEX_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_INDEX.value].z)
        landmarks_dict['LEFT_THUMB_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_THUMB.value].x)
        landmarks_dict['LEFT_THUMB_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_THUMB.value].y)
        landmarks_dict['LEFT_THUMB_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_THUMB.value].z)
        landmarks_dict['RIGHT_THUMB_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_THUMB.value].x)
        landmarks_dict['RIGHT_THUMB_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_THUMB.value].y)
        landmarks_dict['RIGHT_THUMB_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_THUMB.value].z)
        landmarks_dict['LEFT_HIP_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_HIP.value].x)
        landmarks_dict['LEFT_HIP_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_HIP.value].y)
        landmarks_dict['LEFT_HIP_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_HIP.value].z)
        landmarks_dict['RIGHT_HIP_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_HIP.value].x)
        landmarks_dict['RIGHT_HIP_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_HIP.value].y)
        landmarks_dict['RIGHT_HIP_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_HIP.value].z)
        landmarks_dict['LEFT_KNEE_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_KNEE.value].x)
        landmarks_dict['LEFT_KNEE_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_KNEE.value].y)
        landmarks_dict['LEFT_KNEE_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_KNEE.value].z)
        landmarks_dict['RIGHT_KNEE_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
        landmarks_dict['RIGHT_KNEE_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
        landmarks_dict['RIGHT_KNEE_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_KNEE.value].z)
        landmarks_dict['LEFT_ANKLE_x'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
        landmarks_dict['LEFT_ANKLE_y'].append(1-rawdata[index][mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
        landmarks_dict['LEFT_ANKLE_z'].append(rawdata[index][mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
        landmarks_dict['RIGHT_ANKLE_x'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
        landmarks_dict['RIGHT_ANKLE_y'].append(1-rawdata[index][mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
        landmarks_dict['RIGHT_ANKLE_z'].append(rawdata[index][mp_pose.PoseLandmark.RIGHT_ANKLE.value].z)
        landmarks_dict['LEFT_ELBOW_ANGLE'].append(angles['elbow_l'][index])
        landmarks_dict['RIGHT_ELBOW_ANGLE'].append(angles['elbow_r'][index])
        landmarks_dict['LEFT_SHOULDER_ANGLE'].append(angles['shoulder_l'][index])
        landmarks_dict['RIGHT_SHOULDER_ANGLE'].append(angles['shoulder_r'][index])
        landmarks_dict['LEFT_HIP_ANGLE'].append(angles['hip_l'][index])
        landmarks_dict['RIGHT_HIP_ANGLE'].append(angles['hip_r'][index])
        landmarks_dict['LEFT_KNEE_ANGLE'].append(angles['knee_l'][index])
        landmarks_dict['RIGHT_KNEE_ANGLE'].append(angles['knee_r'][index])
        excercise_kpi['WRISTS_DISTANCE'].append(distances['wrists'][index])
        excercise_kpi['ELBOWS_DISTANCE'].append(distances['elbows'][index])
        excercise_kpi['SHOULDERS_DISTANCE'].append(distances['shoulders'][index])
        excercise_kpi['HIPS_DISTANCE'].append(distances['hips'][index])
        excercise_kpi['KNEES_DISTANCE'].append(distances['knees'][index])
        excercise_kpi['ANKLES_DISTANCE'].append(distances['ankles'][index])
        excercise_kpi['LEFT_ELBOW_ANGLE'].append(angles['elbow_l'][index])
        excercise_kpi['RIGHT_ELBOW_ANGLE'].append(angles['elbow_r'][index])
        excercise_kpi['LEFT_SHOULDER_ANGLE'].append(angles['shoulder_l'][index])
        excercise_kpi['RIGHT_SHOULDER_ANGLE'].append(angles['shoulder_r'][index])
        excercise_kpi['LEFT_HIP_ANGLE'].append(angles['hip_l'][index])
        excercise_kpi['RIGHT_HIP_ANGLE'].append(angles['hip_r'][index])
        excercise_kpi['LEFT_KNEE_ANGLE'].append(angles['knee_l'][index])
        excercise_kpi['RIGHT_KNEE_ANGLE'].append(angles['knee_r'][index])
        
    return pd.DataFrame(landmarks_dict), pd.DataFrame(excercise_kpi)
    
def minmaxing_rows(df):
    """
    given a DataFrame, minmax all the rows

    Args:
        df: pd.DataDrame
    
    Returns:
        pd.DataFrame: DataFrame minmaxed
    """
    standarized_df = df.subtract(
        df.min(axis=1), axis=0).divide(
        df.max(axis=1) - df.min(axis=1), axis=0).combine_first(df)
    return standarized_df


def standarize_coordinates(dataframe):
    """
    Given the raw data retrieved from the video, standarize it by minmaxing over coordinates and frames

    Args:
        dataframe: pd.DataFrame
        Landmarks position data
    
    Returns:
        pd.DataFrame: DataFrame containing the standarized landmarks position and angles position
    """
    sorted_cols = list(dataframe.columns)
    x_coord = dataframe[[xcol for xcol in dataframe.columns if '_x' in xcol]]
    y_coord = dataframe[[ycol for ycol in dataframe.columns if '_y' in ycol]]
    z_coord = dataframe[[zcol for zcol in dataframe.columns if '_z' in zcol]]
    angles = dataframe[[anglecol for anglecol in dataframe.columns if '_ANGLE' in anglecol]]
    
    x_coord_ref = x_coord.sub(x_coord['NOSE_x'], axis=0)
    y_coord_ref = y_coord.sub(y_coord['NOSE_y'], axis=0)
    z_coord_ref = z_coord.sub(z_coord['NOSE_z'], axis=0)

    minmaxed_x_coord = minmaxing_rows(x_coord_ref)
    minmaxed_y_coord = minmaxing_rows(y_coord_ref)
    minmaxed_z_coord = minmaxing_rows(z_coord_ref)

    df_scaled_xy = minmaxed_x_coord.join(minmaxed_y_coord, how='inner')
    df_scaled_xyz = df_scaled_xy.join(minmaxed_z_coord, how='inner')
    df_scaled = df_scaled_xyz.join(angles, how='inner')
    
    return df_scaled[sorted_cols]

def create_sequences(df):
    """
    Given the Dataframe with data for every frame of the video, create data windows of 100 rows
    to feed the model

    Args:
        dataframe: pd.DataFrame
        standarized data coordinates and angles
    
    Returns:
        np.array: Array containing the dataFrames for every window
        
    """
    sequences = []
    sequence_length = 100
    variables = list(df.columns)
    for i in range(len(df) - sequence_length + 1):
        seq = df.iloc[i:i + sequence_length][variables].values
        sequences.append(seq)
    return np.array(sequences)

def run_model(sequences):
    """
    Given the sequences, predict user's excercise with the pre-trained LSTM model

    Args:
        sequences: np.array
        array of dataframes
    
    Returns:
        y_pred: np.array
        array of predictions for each sequence
    """
    model = load_model('./Models/PoseClassification/classification_model_seq100.keras')
    y_pred = model.predict(sequences)
    return y_pred


def excercise_from_prediction(y_pred):
    """
    Given the numerical predictions, find out which excercise is the predicted one
    based on the most common output in the sequences 

    Args:
        y_pred: np.array
        array of predictions
    
    Returns:
        predicted_excercise: str
        name of the excercise 
    """
    preds = [np.argmax(x) for x in y_pred]
    preds_no_rest = [x for x in preds if x!=4]
    count_preds = Counter(preds_no_rest)
    final_pred_num = count_preds.most_common(1)[0]
    num_to_excercise = {0: 'Dominada', 1: 'Fondo',
                        2: 'Flexión', 3: 'Sentadilla', 
                        4: 'Descanso'}
    predicted_excercise = num_to_excercise[final_pred_num[0]]
    return predicted_excercise


def pipeline_video_to_excercise(video_path: str):
    """
    orchestrate all the functionalities from video to prediction of excercise

    Args:
        video_path: str
        Path for the video file
    
    Returns:
        excercise: str
        name of the excercise   
    """
    landmarks_video, angles = retrieve_data_video(video_path)
    df_landmarks = landmarks_to_dataframe(landmarks_video, angles)
    df_standarized_landmarks = standarize_coordinates(df_landmarks)
    sequences_video = create_sequences(df_standarized_landmarks)
    prediction_from_model = run_model(sequences_video)
    excercise = excercise_from_prediction(prediction_from_model)
    return excercise

def knn_model(data_model):
    """
    Given the standarized data, predict user's excercise with the pre-trained KNN model

    Args:
        data_model: pd.DataFrame
        Standarized and processed data from video
    
    Returns:
        y_pred: np.array
        array of predictions for each sequence
    """
    with open('./Models/PoseClassification/classification_model_knn3.pickle', 'rb') as file:
        knn_model = pickle.load(file)
    y_pred = knn_model.predict(data_model)
    return y_pred

def excercise_from_prediction_knn(y_pred):
    """
    Given the numerical predictions, find out which excercise is the predicted one
    based on the most common output from the KNN model

    Args:
        y_pred: np.array
        array of predictions
    
    Returns:
        predicted_excercise: str
        name of the excercise 
    """
    count_preds = Counter(y_pred)
    final_pred_num = count_preds.most_common(1)[0]
    num_to_excercise = {0: 'Dominada', 1: 'Fondo',
                        2: 'Flexión', 3: 'Sentadilla', 
                        4: 'Stand Up'}
    predicted_excercise = num_to_excercise[final_pred_num[0]]

    return predicted_excercise

# Try on the webapp.py
def pipeline_landmarks_to_excercise_lstm(landmarks_video, angles, distances):
    """
    orchestrate all the functionalities from data retrieve to data cleansing and excercise prediction
    (lstm sequence model)

    Args:
        video_path: str
        Path for the video file
    
    Returns:
        excercise: str
        name of the excercise  
        excercise_kpis: pd.DataFrame
        Ready to use data from the video 
    """
    df_landmarks, excercise_kpis = landmarks_to_dataframe(landmarks_video, angles, distances)
    df_standarized_landmarks = standarize_coordinates(df_landmarks)
    sequences_video = create_sequences(df_standarized_landmarks)
    prediction_from_model = run_model(sequences_video)
    excercise = excercise_from_prediction(prediction_from_model)
    return excercise, excercise_kpis

# To be used in the webapp.py (best time performance with little loss of accuracy)
def pipeline_landmarks_to_excercise_knn(landmarks_video, angles, distances):
    """
    orchestrate all the functionalities from data retrieve to data cleansing and excercise prediction
    (knn classification model)

    Args:
        video_path: str
        Path for the video file
    
    Returns:
        excercise: str
        name of the excercise  
        excercise_kpis: pd.DataFrame
        Ready to use data from the video 
    """
    df_landmarks, excercise_kpis = landmarks_to_dataframe(landmarks_video, angles, distances)
    df_standarized_landmarks = standarize_coordinates(df_landmarks)
    prediction_from_model = knn_model(df_standarized_landmarks)
    excercise = excercise_from_prediction_knn(prediction_from_model)
    return excercise, excercise_kpis


def kpis_excercise(excercise_kpi):
    """
    Rename and bring the KPIs form the data in the video

    Args:
        excercise_kpi: pd.DataFrame
        Standarized and processed data from the video
    
    Returns:
        description_values: pd.DataFrame
        KPIs for posture correction
    """
    df_useful = excercise_kpi.rename(columns={
        'WRISTS_DISTANCE': 'DISTANCE BETWEEN HANDS', 'ELBOWS_DISTANCE': 'DISTANCE BETWEEN ELBOWS',
        'SHOULDERS_DISTANCE': 'DISTANCE BETWEEN SHOULDERS', 'HIPS_DSITANCE': 'DISTANCE BETWEEN HIPS',
        'KNEES_DISTANCE': 'DISTANCE BETWEEN KNEES', 'ANKLES_DISTANCE': 'DISTANCE BETWEEN FEET',
        'LEFT_ELBOW_ANGLE': 'LEFT ELBOW ANGLE','RIGHT_ELBOW_ANGLE': 'RIGHT ELBOW ANGLE',
        'LEFT_SHOULDER_ANGLE': 'LEFT SHOULDER ANGLE','RIGHT_SHOULDER_ANGLE': 'RIGHT SHOULDER ANGLE',
        'LEFT_HIP_ANGLE': 'LEFT HIP ANGLE','RIGHT_HIP_ANGLE': 'RIGHT HIP ANGLE',
        'LEFT_KNEE_ANGLE': 'LEFT KNEE ANGLE','RIGHT_KNEE_ANGLE': 'RIGHT KNEE ANGLE'})
    description_values = df_useful.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).loc[['min', 'max', 'mean','10%', '25%', '50%', '75%', '90%']]
    return description_values

# USING KPIS TO EFFECTIVELY PROMPT A POSTURE CORRECTION

def angle_in_range(angle, ang_min, ang_max):
    """
    Verify wether an angle is in between the boundaries
    
    Args:
        angle: float
        angle to verify
        ang_min: float
        lower boundary
        ang_max: float
        upper boundary
    
    Returns:
        Bool: true if inside, false otherwise
    """
    if ang_min <= angle <= ang_max:
        return True
    else:
        return False

def equal_distances(dist_1, dist_2):
    """
    Evaluate wether two distances are equal giving a +-15% of error to account
    distortions due to perspective in the videos
    
    Args:
        dist_1: float
        distance value 1
        dist_2: float
        distance value 2
        
    Returns:
        str: definition of the outcome
    """
    min_dist = dist_1 * 0.85
    max_dist = dist_1 * 1.15
    
    if min_dist <= dist_2 <= max_dist:
        is_equal_answer = 'The distances are equal'
    else:
        is_equal_answer = 'The distances differ more than they should'
    return is_equal_answer

def effective_angle_range(min_vid, max_vid, min_small_angle, max_small_angle, min_big_angle, max_big_angle):
    """
    Verify wether a joint angle from the video stays in range during the full video
    
    Args:
        min_vid: float
        minimum angle processed from the video
        max_vid: float
        maximum angle processed from the video
        min_small_angle: float
        lower boundary for the minimum angle in the video
        max_small_angle: float
        upper boundary for the minimum angle in the video
        min_big_angle: float
        lower boundary for the maximum angle in the video
        max_big_angle: float
        upper boundary for the maximum angle in the video
    
    Returns:
        str: defintion of the outcome 
    """
    is_min_angle_vid_correct = angle_in_range(min_vid, min_small_angle, max_small_angle)
    is_max_angle_vid_correct = angle_in_range(max_vid, min_big_angle, max_big_angle)
    if is_min_angle_vid_correct and is_max_angle_vid_correct:
        angle_response = 'Angle movement within the correct range.'
    elif is_max_angle_vid_correct:
        angle_response = 'Angle movement outside the range (lower than minimum angle).'
    elif is_min_angle_vid_correct:
        angle_response = 'Angle movement outside the range (higher than maximum angle).'
    else: 
        angle_response = f'Angle movement outside the range (range seen: [{min_vid} - {max_vid}]).'
    return angle_response


def squat_prompt(squat_kpis):
    """
    Create a prompt for the video processing based on the fact that the predicted excercise is a squat
    and the effective KPIs are processed for the concensualized good movements for this excercise
    
    Args:
        squat_kpis: pd.DataFrame
        KPIs from the video processed data
    
    Returns:
        str: LLM generated answer 
    """
    left_knee = effective_angle_range(squat_kpis['LEFT KNEE ANGLE']['10%'],
                                      squat_kpis['LEFT KNEE ANGLE']['90%'],
                                      80, 110,
                                      170, 190)
    right_knee = effective_angle_range(squat_kpis['RIGHT KNEE ANGLE']['10%'],
                                      squat_kpis['RIGHT KNEE ANGLE']['90%'],
                                      80, 110,
                                      170, 190)
    left_hip = effective_angle_range(squat_kpis['LEFT HIP ANGLE']['10%'],
                                      squat_kpis['LEFT HIP ANGLE']['90%'],
                                      20, 90,
                                      170, 190)
    right_hip = effective_angle_range(squat_kpis['RIGHT HIP ANGLE']['10%'],
                                      squat_kpis['RIGHT HIP ANGLE']['90%'],
                                      20, 90,
                                      170, 190)
    shoulder_leg_distances = equal_distances(squat_kpis['DISTANCE BETWEEN FEET']['mean'],
                                             squat_kpis['DISTANCE BETWEEN SHOULDERS']['mean'])

    
    # Create a list with the processed data to build the prompt
    kpis = [
        ('left knee angle', left_knee),
        ('right knee angle', right_knee),
        ('left hip angle', left_hip),
        ('right knee angle', right_hip),
        ('distance between shoulders compared to distance between feet', shoulder_leg_distances)
        ]

    # Prompt base
    prompt = """
                The reference for a good squat is: \n
                    - knee angles in a range between 80 degrees and 110 degrees. \n
                    - Hip angles in a range between 20 degrees and 180 degrees. \n
                    - Distance between shoulders similar to distance between feet. \n
                
                I sent a video doing the excercise.
                
                The following key points describe key points taken from the video.
                They will help evaluate the excercise.\n"""
    
    # Add KPI prompt info
    for kpi, state in kpis:
        prompt += f"- {kpi}: {state}\n"

    prompt += """Based on the key points presented, do you think I am doing correct squats?"""
    print('----------CREATING OLLAMA ANSWER FOR: SQUAT------------')
    response = client.chat(model='llama3.1', messages=[
        {
            'role': 'system',
            'content': """You are a calisthenics assistant. You need to give advice to a person doing squats.
                          Based on the key points presented, give a answer highlighting
                          the possible bad posturing they might be doing, explaining why
                          you think it is not correct. You must end your coaching text by
                          explaining that you could have processed the video incorrectly,
                          encouraging them to send another video with a better perspective
                          to capture perfectly the movement."""
        },
       {
         'role': 'user',
         'content': prompt
         ,
       },
        {'role': 'assistant',
        'content': 'My answer as a calisthenics assistant:'}
     ])
    answer = response['message']['content']
    print('----------ANSWER CREATED------------')
    return answer


def pullup_prompt(pullup_kpis):
    """
    Create a prompt for the video processing based on the fact that the predicted excercise is a pull-up
    and the effective KPIs are processed for the concensualized good movements for this excercise
    
    Args:
        pullup_kpis: pd.DataFrame
        KPIs from the video processed data
    
    Returns:
        str: LLM generated answer 
    """
    left_elbow = effective_angle_range(pullup_kpis['LEFT ELBOW ANGLE']['10%'],
                                       pullup_kpis['LEFT ELBOW ANGLE']['90%'],
                                       15, 45,
                                       170, 190)
    right_elbow = effective_angle_range(pullup_kpis['RIGHT ELBOW ANGLE']['10%'],
                                        pullup_kpis['RIGHT ELBOW ANGLE']['90%'],
                                        15, 45,
                                        170, 190)
    left_shoulder = effective_angle_range(pullup_kpis['LEFT SHOULDER ANGLE']['10%'],
                                          pullup_kpis['LEFT SHOULDER ANGLE']['90%'],
                                          10, 40,
                                          170, 190)
    right_shoulder = effective_angle_range(pullup_kpis['RIGHT SHOULDER ANGLE']['10%'],
                                           pullup_kpis['RIGHT SHOULDER ANGLE']['90%'],
                                           10, 40,
                                           170, 190)
    left_hip = effective_angle_range(pullup_kpis['LEFT HIP ANGLE']['10%'],
                                     pullup_kpis['LEFT HIP ANGLE']['90%'],
                                     140, 180,
                                     160, 190)
    right_hip = effective_angle_range(pullup_kpis['RIGHT HIP ANGLE']['10%'],
                                      pullup_kpis['RIGHT HIP ANGLE']['90%'],
                                      140, 180,
                                      160, 190)
    shoulder_hand_distances = equal_distances(pullup_kpis['DISTANCE BETWEEN HANDS']['mean'],
                                             pullup_kpis['DISTANCE BETWEEN SHOULDERS']['mean'])

    # Create a list with the processed data to build the prompt
    kpis = [
        ('left elbow angle', left_elbow),
        ('right elbow angle', right_elbow),
        ('left shoulder angle', left_shoulder),
        ('right shoulder angle', right_shoulder),
        ('left hip angle', left_hip),
        ('right knee angle', right_hip),
        ('distance between hands compared to distance between shoulders', shoulder_hand_distances)
        ]

    # Prompt base
    prompt = """
                The reference for a good pullup is: \n
                    - elbow angles in a range between 20 degrees and 180 degrees for full range of motion. \n
                    - shoulder angles in a range between 20 degrees and 180 degrees. \n
                    - Hip angles in a range between 140 degrees and 190 degrees to avoid kipping. \n
                    - Distance between shoulders similar to distance between hands to avoid injuries.\n
                
                I sent a video doing the excercise.
                
                The following key points describe key points taken from the video.
                They will help evaluate the excercise.\n"""
    # Add KPI prompt info
    for kpi, state in kpis:
        prompt += f"- {kpi}: {state}\n"

    prompt += """Based on the key points presented, do you think I am doing correct pull ups?"""
    print('----------CREATING OLLAMA ANSWER FOR: PULL UP------------')
    response = client.chat(model='llama3.1', messages=[
        {
            'role': 'system',
            'content': """You are a calisthenics assistant. You need to give advice to a person doing pull ups.
                          Based on the key points presented, give a answer highlighting
                          the possible bad posturing they might be doing, explaining why
                          you think it is not correct. You must end your coaching motivational text by
                          explaining that you could have processed the video incorrectly,
                          encouraging them to send another video with a better perspective
                          to capture perfectly the movement.
                          Answer as if you had seen the video."""
        },
       {
         'role': 'user',
         'content': prompt
         ,
       },
        {'role': 'assistant',
        'content': 'My answer as a calisthenics assistant who saw the video:'}
     ])
    answer = response['message']['content']
    print('----------ANSWER CREATED------------')
    return answer



def dip_prompt(dip_kpis):
    """
    Create a prompt for the video processing based on the fact that the predicted excercise is a dip
    and the effective KPIs are processed for the concensualized good movements for this excercise
    
    Args:
        dip_kpis: pd.DataFrame
        KPIs from the video processed data
    
    Returns:
        str: LLM generated answer 
    """
    left_elbow = effective_angle_range(dip_kpis['LEFT ELBOW ANGLE']['10%'],
                                       dip_kpis['LEFT ELBOW ANGLE']['90%'],
                                       75, 100,
                                       170, 190)
    right_elbow = effective_angle_range(dip_kpis['RIGHT ELBOW ANGLE']['10%'],
                                        dip_kpis['RIGHT ELBOW ANGLE']['90%'],
                                        75, 100,
                                        170, 190)
    left_shoulder = effective_angle_range(dip_kpis['LEFT SHOULDER ANGLE']['10%'],
                                          dip_kpis['LEFT SHOULDER ANGLE']['90%'],
                                          0,42,
                                          45,100)
    right_shoulder = effective_angle_range(dip_kpis['RIGHT SHOULDER ANGLE']['10%'],
                                           dip_kpis['RIGHT SHOULDER ANGLE']['90%'],
                                           0,42,
                                           45,100)
    left_hip = effective_angle_range(dip_kpis['LEFT HIP ANGLE']['10%'],
                                     dip_kpis['LEFT HIP ANGLE']['90%'],
                                     140, 180,
                                     160, 190)
    right_hip = effective_angle_range(dip_kpis['RIGHT HIP ANGLE']['10%'],
                                      dip_kpis['RIGHT HIP ANGLE']['90%'],
                                      140, 180,
                                      160, 190)
    
    # Create a list with the processed data to build the prompt
    kpis = [
        ('left elbow angle', left_elbow),
        ('right elbow angle', right_elbow),
        ('left shoulder angle', left_shoulder),
        ('right shoulder angle', right_shoulder),
        ('left hip angle', left_hip),
        ('right knee angle', right_hip)
        ]

    # Prompt base
    prompt = """
                The reference for a good dip is: \n
                    - elbow angles in a range between 80 degrees and 180 degrees for full range of motion. \n
                    - shoulder angles in a range between 0 degrees and 90 degrees. \n
                    - Hip angles in a range between 140 degrees and 190 degrees to avoid kipping. \n
                
                I sent a video doing the excercise.
                
                The following key points describe key points taken from the video.
                They will help evaluate the excercise.\n"""
    
    # Add KPI prompt info
    for kpi, state in kpis:
        prompt += f"- {kpi}: {state}\n"

    prompt += """Based on the key points presented, do you think I am doing correct dips?"""
    print('----------CREATING OLLAMA ANSWER FOR: DIPS------------')
    response = client.chat(model='llama3.1', messages=[
        {
            'role': 'system',
            'content': """You are a calisthenics assistant. You need to give advice to a person doing dips.
                          Based on the key points presented, give a answer highlighting
                          the possible bad posturing they might be doing, explaining why
                          you think it is not correct. You must end your coaching motivational text by
                          explaining that you could have processed the video incorrectly,
                          encouraging them to send another video with a better perspective
                          to capture perfectly the movement
                          Answer as if you had seen the video."""
        },
       {
         'role': 'user',
         'content': prompt
         ,
       },
        {'role': 'assistant',
        'content': 'My answer as a calisthenics assistant who saw the video:'}
     ])
    answer = response['message']['content']
    print('----------ANSWER CREATED------------')
    return answer


def pushup_prompt(pushup_kpis):
    """
    Create a prompt for the video processing based on the fact that the predicted excercise is a push-up
    and the effective KPIs are processed for the concensualized good movements for this excercise
    
    Args:
        pushup_kpis: pd.DataFrame
        KPIs from the video processed data
    
    Returns:
        str: LLM generated answer 
    """
    left_elbow = effective_angle_range(pushup_kpis['LEFT ELBOW ANGLE']['10%'],
                                       pushup_kpis['LEFT ELBOW ANGLE']['90%'],
                                       50, 90,
                                       170, 190)
    right_elbow = effective_angle_range(pushup_kpis['RIGHT ELBOW ANGLE']['10%'],
                                        pushup_kpis['RIGHT ELBOW ANGLE']['90%'],
                                        50, 90,
                                        170, 190)
    left_shoulder = effective_angle_range(pushup_kpis['LEFT SHOULDER ANGLE']['10%'],
                                          pushup_kpis['LEFT SHOULDER ANGLE']['90%'],
                                          0,42,
                                          45,110)
    right_shoulder = effective_angle_range(pushup_kpis['RIGHT SHOULDER ANGLE']['10%'],
                                           pushup_kpis['RIGHT SHOULDER ANGLE']['90%'],
                                           0,42,
                                           45,110)
    left_hip = effective_angle_range(pushup_kpis['LEFT HIP ANGLE']['10%'],
                                     pushup_kpis['LEFT HIP ANGLE']['90%'],
                                     160, 180,
                                     160, 190)
    right_hip = effective_angle_range(pushup_kpis['RIGHT HIP ANGLE']['10%'],
                                      pushup_kpis['RIGHT HIP ANGLE']['90%'],
                                      160, 180,
                                      160, 190)
    shoulder_hand_distances = equal_distances(pushup_kpis['DISTANCE BETWEEN HANDS']['mean'],
                                             pushup_kpis['DISTANCE BETWEEN SHOULDERS']['mean'])
    
    # Create a list with the processed data to build the prompt
    kpis = [
        ('left elbow angle', left_elbow),
        ('right elbow angle', right_elbow),
        ('left shoulder angle', left_shoulder),
        ('right shoulder angle', right_shoulder),
        ('left hip angle', left_hip),
        ('right knee angle', right_hip),
        ('distance between hands compared to distance between shoulders', shoulder_hand_distances)
        ]

    # Prompt base
    prompt = """
                The reference for a good dip is: \n
                    - elbow angles in a range between 60 degrees and 180 degrees for full range of motion. \n
                    - shoulder angles in a range between 0 degrees and 90 degrees. \n
                    - Hip angles in a range between 160 degrees and 190 degrees to strengthen the core. \n
                
                I sent a video doing the excercise.
                
                The following key points describe key points taken from the video.
                They will help evaluate the excercise.\n"""
    
    # Add KPI prompt info
    for kpi, state in kpis:
        prompt += f"- {kpi}: {state}\n"

    prompt += """Based on the key points presented, do you think I am doing correct pushups?"""
    print('----------CREATING OLLAMA ANSWER FOR: PUSH UPS------------')
    response = client.chat(model='llama3.1', messages=[
        {
            'role': 'system',
            'content': """You are a calisthenics assistant. You need to give advice to a person doing push ups.
                          Based on the key points presented, give a answer highlighting
                          the possible bad posturing they might be doing, explaining why
                          you think it is not correct. You must end your coaching motivational text by
                          explaining that you could have processed the video incorrectly,
                          encouraging them to send another video with a better perspective
                          to capture perfectly the movement. 
                          Answer as if you had seen the video."""
        },
       {
         'role': 'user',
         'content': prompt
         ,
       },
        {'role': 'assistant',
        'content': 'My answer as a calisthenics assistant who saw the video:'}
     ])
    answer = response['message']['content']
    print('----------ANSWER CREATED------------')
    return answer


def unknown_excercise():
    """
    Create a prompt for the video processing based on the fact that the predicted excercise is outside the limits of the model
    
        str: LLM generated answer 
    """
    print('----------CREATING OLLAMA ANSWER FOR: UNKNOWN EXCERCISE------------')
    response = client.chat(model='llama3.1', messages=[
        {
            'role': 'system',
            'content': """You are a calisthenics assistant. You need to give advice to a person doing an excercise.
                          They sent you a video that you could not process. Explain that you are sorry and it is an inconvinience.
                          Ask them politely to send another video, with another perspective or better light, 
                          so that you can process it correctly.
                          Answer as if you had seen the video."""
        },
       {
         'role': 'user',
         'content': ''' This is my excercise. Can you help me?'''
         ,
       },
        {'role': 'assistant',
        'content': 'My answer as a calisthenics assistant that cannot understand the video:'}
     ])
    answer = response['message']['content']
    print('----------ANSWER CREATED------------')
    return answer
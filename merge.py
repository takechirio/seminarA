import cv2
import random
import math
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import distance
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def identify_hand(landmark):
  landmark_x=list()
  landmark_y=list()
  for i in range(21):
    landmark_x.append(landmark[i].x)
    landmark_y.append(landmark[i].y)
  
  fingers=list() #親指、人差し指、中指、薬指、小指
  straight_finger=0

  for i in range(5):
    x=np.array(landmark_x[4*i+1:4*(i+1)])
    y=np.array(landmark_y[4*i+1:4*(i+1)])
    coef=np.corrcoef(x,y)
    fingers.append(abs(coef[0][1]))
    if (abs(coef[0][1])>0.9):
      straight_finger+=1

  #print(straight_finger)

  if straight_finger==5:
    #print("paa")
    return 1
  elif straight_finger==1 or straight_finger==0:
   # print("guu")
    return 2
  
  return -1

# Webカメラから入力
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
out = cv2.VideoWriter('output/sample.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), 15, (int(width), int(height)))

fore=(50,50)

speed=20.0 #1回の画面で動くピクセルすう
angle=math.pi/3

rec1=(random.randint(1,width-100),random.randint(1,height-100))
rec2=(rec1[0]+fore[0],rec1[1]+fore[1])
pre=rec1

previous_hands_pos = [0,0]
now_hands_pos = [0,0] 

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    rec1=(int(pre[0]+speed*math.cos(angle)),int(pre[1]+speed*math.sin(angle)))
    rec2=(rec1[0]+fore[0],rec1[1]+fore[1])

    if rec1[0]+fore[0]>width or rec1[0]<0:
       angle=math.pi-angle
    if rec1[1]+fore[1]>height or rec1[1]<0:
       angle=-1*angle
    pre=rec1


    cv2.rectangle(image, rec1,rec2, (0, 255, 0), thickness=4)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
 
    # 検出された手の骨格をカメラ画像に重ねて描画
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
      height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

      identify_hand(results.multi_hand_landmarks[0].landmark)
      previous_hands_pos = now_hands_pos
      now_hands_pos = [results.multi_hand_landmarks[0].landmark[1].x*width,results.multi_hand_landmarks[0].landmark[1].y*height]
      dist = distance.euclidean(previous_hands_pos, now_hands_pos)
      # ラジアン単位を取得
      radian = math.atan2(previous_hands_pos[1] - now_hands_pos[1], previous_hands_pos[0] - now_hands_pos[0])
      # ラジアン単位から角度を取得
      degree = radian * (180 / math.pi)
      print(degree)

      landmark_x=list()
      landmark_y=list()
      for i in range(21):
        x=results.multi_hand_landmarks[0].landmark[i].x*width
        y=results.multi_hand_landmarks[0].landmark[i].y*height
        landmark_x.append(x)
        landmark_y.append(y)

      #print(results.multi_handedness)

      #print(x,y)
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        cv2.drawMarker(image,(100,100),(0,0,0),markerType=cv2.MARKER_STAR, markerSize=10)
    #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

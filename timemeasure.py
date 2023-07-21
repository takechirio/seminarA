import cv2
import mediapipe as mp
import numpy as np
import math
from scipy.spatial import distance
import time
import argparse
import pandas as pd
import random
from matplotlib import pyplot as pyp
import statistics

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
angle = 0
 
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
    print("paa")
    return 1
  elif straight_finger==1 or straight_finger==0:
    print("guu")
    return 2
  
  return -1

def comp(fore_img, image, dx,dy):
      h, w = fore_img.shape[:2]
      fore_x_min, fore_x_max = 0, w
      fore_y_min, fore_y_max = 0, h
 
      back_h, back_w = image.shape[:2]
      back_x_min, back_x_max = dx, dx+w
      back_y_min, back_y_max = dy, dy+h
      
      if back_x_min < 0:
          fore_x_min = fore_x_min - back_x_min
          back_x_min = 0
         
      if back_x_max > back_w:
          fore_x_max = fore_x_max - (back_x_max - back_w)
          back_x_max = back_w
 
      if back_y_min < 0:
          fore_y_min = fore_y_min - back_y_min
          back_y_min = 0
         
      if back_y_max > back_h:
          fore_y_max = fore_y_max - (back_y_max - back_h)
          back_y_max = back_h
      
      #大きさを取得して画像内に入るように大きさを制限

      roi = image[back_y_min:back_y_max, back_x_min:back_x_max]
      sync_img = fore_img[fore_y_min:fore_y_max, fore_x_min:fore_x_max]
      if(fore_x_max>fore_x_min and fore_y_max > fore_y_min):
        img2gray = cv2.cvtColor(sync_img,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(sync_img,sync_img,mask = mask)
        dst = cv2.add(img1_bg,img2_fg)
        image[back_y_min:back_y_max, back_x_min:back_x_max] = dst
          
      return image

def move_obj(fore_image,x,y,v):
   vx = 0
   vy = 0
   #if(hands == 'guu'):
   #x,y = parabolic(x,y,v,vx,vy)

   #elif(hands == 'paa'):
     # None
   

def touch_judge(hand_x,hand_y,dx,dy,fore_img,image):
      h, w = fore_img.shape[:2]
 
      back_h, back_w = image.shape[:2]
      back_x_min, back_x_max = dx, dx+w
      back_y_min, back_y_max = dy, dy+h
      
      if(back_x_min<hand_x and hand_x < back_x_max and back_y_min < hand_y and hand_y < back_y_max):
         return True
      else:
         return False
      
def reflect(angle,fore_img, image, dx,dy):
      global reflect_flag
      h, w = fore_img.shape[:2]
 
      back_h, back_w = image.shape[:2]
      back_x_min, back_x_max = dx, dx+w
      back_y_min, back_y_max = dy, dy+h
      
      if back_x_min < 0 or back_x_max > back_w:
          angle=180-angle
          reflect_flag = True
 
      if back_y_min < 0 or back_y_max > back_h:
          angle=-1*angle
          reflect_flag = True

      return angle

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video_path', type=str, default='', help='Path to the video file.')
args = parser.parse_args()
if args.video_path != '':
  cap = cv2.VideoCapture(args.video_path)
else:
  cap = cv2.VideoCapture(0)

width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

px=random.randint(1,width-100)
py=random.randint(1,height-100)

obj_touched = False
obj_touch_now = False
hand_velocity = 0
previous_hands_pos = [0,0]
now_hands_pos = [0,0] 
previous_hand_time = 0
now_hand_time = 0
dist = 0
reflect_flag = True
count = 0
nowtime = 0
pasttime = 0
change_time = 0
bomb_flag = False
fore_img = cv2.imread(r"data\ball.png")
fore_img = cv2.resize(fore_img, (100, 100))
bomb_img = cv2.imread(r"data\bomb.png")
bomb_img = cv2.resize(bomb_img, (100, 100))
start_time = time.perf_counter()
previous_hand_time = start_time
bomb_first_flag = True
measuretime = 0
previousmeasuretime = 0
frametime = []
handtime = []
posetime = []
balltime = []
first = True

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        break
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      measuretime = time.perf_counter()
      hands_results = hands.process(image)
      handtime.append(time.perf_counter() - measuretime)
      measuretime = time.perf_counter()

      #計測フレーム数を増加
      count +=1
      pasttime = nowtime
      nowtime = time.perf_counter()
 
      # 検出されたポーズの骨格をカメラ画像に重ねて描画
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
        
      
      frametime.append(time.perf_counter() - measuretime)
      measuretime = time.perf_counter()
      
      if hands_results.multi_hand_landmarks: 
        x=hands_results.multi_hand_landmarks[0].landmark[9].x
        y=hands_results.multi_hand_landmarks[0].landmark[9].y
        height, weight = image.shape[:2]
        previous_hands_pos = now_hands_pos
        now_hands_pos = [x*weight,y*height]
        previous_hand_time = now_hand_time
        now_hand_time = nowtime
        duration = now_hand_time - previous_hand_time             
        for hand_landmarks in hands_results.multi_hand_landmarks:
          for hand_landmark in hand_landmarks.landmark:
            if(touch_judge(hand_landmark.x*weight,hand_landmark.y*height,int(px),int(py),fore_img,image)):
              obj_touched = True
              obj_touch_now = True      


      if(obj_touched):
        if(obj_touch_now and reflect_flag and now_hand_time - change_time >0.5):
          if(identify_hand(hands_results.multi_hand_landmarks[0].landmark)==1):
             bomb_flag = True
          change_time = now_hand_time
          reflect_flag = False
          dist = distance.euclidean(previous_hands_pos, now_hands_pos)
          hand_velocity = dist/duration
          obj_vec = hand_velocity/2
          # ラジアン単位を取得
          radian = -1*math.atan2(previous_hands_pos[1] - now_hands_pos[1], now_hands_pos[0] - previous_hands_pos[0] )
          # ラジアン単位から角度を取得
          angle = radian * (180 / math.pi)
        #x,y = move_img()
        #画像の位置変更
        angle = reflect(angle,fore_img, image, int(px),int(py))
        vx = obj_vec*math.cos(angle*math.pi/180.0)
        vy = obj_vec*math.sin(angle*math.pi/180.0)
        px = px+vx*(nowtime - pasttime)
        py = py+vy*(nowtime - pasttime)
      
      if(bomb_flag):
        if(bomb_first_flag):
           bomb_first_flag = False
           bom_x = int(px)
           bom_y = int(py)
           bom_start_time = time.perf_counter()
        if(time.perf_counter()-bom_start_time <2.0):
          image = comp(bomb_img,image,bom_x,bom_y)
      else:
        image = comp(fore_img,image,int(px),int(py))
      
      balltime.append(time.perf_counter() - measuretime)
      measuretime = time.perf_counter()
      obj_touch_now = False
      cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
      if(nowtime -start_time > 30.0):
         print("Frame is")
         print(count/(nowtime -start_time))
         break
      if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()

pyp.title("hand detection")
pyp.ylabel("frequency")
pyp.xlabel("processing time(s/frame)")
pyp.hist(handtime,bins=30)
pyp.legend()
pyp.show()
'''
pyp.title("ball")
pyp.ylabel("frequency")
pyp.xlabel("processing time(s/frame)")
pyp.hist(balltime,bins=30)
pyp.legend()
pyp.show()

 

date = []
date.append(statistics.mean(handtime))
date.append(statistics.mean(posetime))
date.append(statistics.mean(balltime))
labels = ['hand', 'pose', 'ball']

pyp.pie(date, startangle=90, counterclock=False,  autopct='%.1f%%', pctdistance=0.8, labels=labels)
pyp.show()
'''
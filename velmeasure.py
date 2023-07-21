import cv2
import mediapipe as mp
import numpy as np
import math
from scipy.spatial import distance
import time
import argparse
import pandas as pd
import random
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
  landmark_z=list()
  for i in range(21):
    landmark_x.append(landmark[i].x)
    landmark_y.append(landmark[i].y)
    landmark_z.append(landmark[i].z)
  
  fingers=list() #親指、人差し指、中指、薬指、小指
  straight_finger=0

  for i in range(5):
    x=np.array(landmark_x[4*i+1:4*(i+1)])
    y=np.array(landmark_y[4*i+1:4*(i+1)])
    coef=np.corrcoef(x,y)
    fingers.append(abs(coef[0][1]))
    if (abs(coef[0][1])>0.9):
      straight_finger+=1

  if straight_finger==5:
    for i in range(5):
       finger_root=distance.euclidean((landmark_x[0],landmark_y[0]),(landmark_x[4*i+1],landmark_y[4*i+1]))
       finger_tip=distance.euclidean((landmark_x[0],landmark_y[0]),(landmark_x[4*(i+1)],landmark_y[4*(i+1)]))
       if(finger_tip<finger_root):
          straight_finger-=1
    if(straight_finger==5):
      #print("paa")
      return 1
    else:
      #print("guu2")
      return 2
  elif straight_finger==1 or straight_finger==0:
    abs_z=0
    for i in range(5):
       #print(abs(landmark_z[0]-landmark_z[4*(i+1)]))
       if(abs(landmark_z[0]-landmark_z[4*(i+1)])>0.1):
          abs_z+=1
    
    if(abs_z>3):
       #print("paa2")
       return 2
    else:
     # print("guu")
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

def in_rect(image,pose_landmark,target):
    #ここで11,12,23,24の座標がほしい->エラーはく
    h,w=image.shape[:2]
    a = (pose_landmark[11].x*w, pose_landmark[11].y*h)
    b = (pose_landmark[12].x*w, pose_landmark[12].y*h)
    #cとd入れ替えた
    c = (pose_landmark[24].x*w, pose_landmark[24].y*h)
    d = (pose_landmark[23].x*w, pose_landmark[23].y*h)
    e = (target[0], target[1])

    # 原点から点へのベクトルを求める
    vector_a = np.array(a)
    vector_b = np.array(b)
    vector_c = np.array(c)
    vector_d = np.array(d)
    vector_e = np.array(e)

    # 点から点へのベクトルを求める
    vector_ab = vector_b - vector_a
    vector_ae = vector_e - vector_a
    vector_bc = vector_c - vector_b
    vector_be = vector_e - vector_b
    vector_cd = vector_d - vector_c
    vector_ce = vector_e - vector_c
    vector_da = vector_a - vector_d
    vector_de = vector_e - vector_d

    # 外積を求める
    vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
    vector_cross_bc_be = np.cross(vector_bc, vector_be)
    vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
    vector_cross_da_de = np.cross(vector_da, vector_de)
    result=(vector_cross_ab_ae < 0 and vector_cross_bc_be < 0 and vector_cross_cd_ce < 0 and vector_cross_da_de < 0)
    return result
      
def reflect(angle,fore_img, image, dx,dy):
      global reflect_flag
      global obj_vec
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

#ボールの初期位置
px=random.randint(1,width-100)
py=random.randint(1,height-100)

#フラグ
game_finish_flag=False
obj_touched = False
obj_touch_now = False
bomb_flag = False
bomb_first_flag = True

hand_velocity = 0
obj_vec=300
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


fore_img = cv2.imread(r"data/ball.png")
fore_img = cv2.resize(fore_img, (100, 100))
bomb_img = cv2.imread(r"data/bomb.png")
bomb_img = cv2.resize(bomb_img, (100, 100))
start_time = time.perf_counter()
previous_hand_time = start_time

vellist=[]
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
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
      pose_results = pose.process(image)
      hands_results = hands.process(image)

      #計測フレーム数を増加
      count +=1
      pasttime = nowtime
      nowtime = time.perf_counter()
 
      # 検出されたポーズの骨格をカメラ画像に重ねて描画
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if hands_results.multi_hand_landmarks:
        x=hands_results.multi_hand_landmarks[0].landmark[12].x
        y=hands_results.multi_hand_landmarks[0].landmark[12].y
        height, weight = image.shape[:2]
        previous_hands_pos = now_hands_pos
        now_hands_pos = [x*weight,y*height]
        previous_hand_time = now_hand_time
        now_hand_time = nowtime
        duration = now_hand_time - previous_hand_time
        dist = distance.euclidean(previous_hands_pos, now_hands_pos)
        vellist.append(dist/duration)
        if(len(vellist)>10):
           vellist.pop(0)
        hand_velocity = sum(vellist)/len(vellist)
        print(hand_velocity)
           

        for hand_landmarks in hands_results.multi_hand_landmarks:
          hand=identify_hand(hand_landmarks.landmark)
          if(hand==1):
              cv2.putText(image,"paa",(int(hand_landmarks.landmark[0].x),int(hand_landmarks.landmark[0].y)),cv2.FONT_HERSHEY_SIMPLEX,1.0,color=(0, 255, 0),thickness=2,lineType=cv2.LINE_4)
          elif(hand==2):
              cv2.putText(image,"guu",(int(hand_landmarks.landmark[0].x),int(hand_landmarks.landmark[0].y)),cv2.FONT_HERSHEY_SIMPLEX,1.0,color=(0, 255, 0),thickness=2,lineType=cv2.LINE_4)
          for hand_landmark in hand_landmarks.landmark:
            if(touch_judge(hand_landmark.x*weight,hand_landmark.y*height,int(px),int(py),fore_img,image)):
              obj_touched = True
              obj_touch_now = True
        
        for hand_landmarks in hands_results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
        
      mp_drawing.draw_landmarks(
          image,
          pose_results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      if(obj_touched):
        angle = reflect(angle,fore_img, image, int(px),int(py))
        if(reflect_flag and (not game_finish_flag) and (not bomb_flag)):
            obj_vec=obj_vec+10
            reflect_flag=False
        if(obj_touch_now and now_hand_time - change_time >0.5):
          if(identify_hand(hands_results.multi_hand_landmarks[0].landmark)==1):
            bomb_flag = True
          change_time = now_hand_time
          #reflect_flag = False
          #dist = distance.euclidean(previous_hands_pos, now_hands_pos)
          #hand_velocity = dist/duration
          # ラジアン単位を取得
          radian = -1*math.atan2(previous_hands_pos[1] - now_hands_pos[1], now_hands_pos[0] - previous_hands_pos[0] )
          # ラジアン単位から角度を取得
          angle = radian * (180 / math.pi)
          #print(hand_velocity)

        #画像の位置変更
        vx = obj_vec*math.cos(angle*math.pi/180.0)
        vy = obj_vec*math.sin(angle*math.pi/180.0)
        px = px+vx*(nowtime - pasttime)
        py = py+vy*(nowtime - pasttime)

        #TODO ゲームオーバー
        if(pose_results.pose_landmarks):
          if(in_rect(image,pose_results.pose_landmarks.landmark,(px,py))):
            print("GAME OVER")
            game_finish_flag=True
            obj_touched=False
            obj_touch_now=False
            bomb_first_flag=True
            bomb_flag=False
        
      if(game_finish_flag):
        image = cv2.flip(image, 1)
        cv2.putText(image,"GAME OVER",(100,300),cv2.FONT_HERSHEY_SIMPLEX,5.0,color=(0, 0, 255),thickness=2,lineType=cv2.LINE_4)
      elif(bomb_flag):
        if(bomb_first_flag):
          bomb_first_flag = False
          bom_x = int(px)
          bom_y = int(py)
          bom_start_time = time.perf_counter()
        if(time.perf_counter()-bom_start_time <2.0):
          image = comp(bomb_img,image,bom_x,bom_y)
        image = cv2.flip(image, 1)
      else:
        image = comp(fore_img,image,int(px),int(py))
        image = cv2.flip(image, 1)
      
      cv2.putText(image,str(obj_vec),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.0,color=(0, 255, 0),thickness=2,lineType=cv2.LINE_4)
        
      obj_touch_now = False
      cv2.imshow('MediaPipe Pose', image)
      if(nowtime -start_time > 200.0):
         print("Frame is")
         print(count/(nowtime -start_time))
         break
      if cv2.waitKey(5) & 0xFF == 27:
        break

      #TODO リセットボタン
      if cv2.waitKey(5) & 0xFF == 114:
        obj_vec=200
        px=random.randint(1,width-100)
        py=random.randint(1,height-100)
        obj_touched=False
        obj_touch_now=False
        bomb_first_flag=True
        bomb_flag=False
         
cap.release()
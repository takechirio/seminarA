import cv2
import mediapipe as mp
import numpy as np
import math
from scipy.spatial import distance
import time
import argparse
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
 
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

def move_img(fore_image,x,y,v):
   vx = 0
   vy = 0
   #if(hands == 'guu'):
   x,y = parabolic(x,y,v,vx,vy)

   #elif(hands == 'paa'):
     # None
   
   
def parabolic():
  g = 9.8
  v0 = 50
  angle = 45.0
  dt = 0.1
  vx = v0*math.cos(angle*math.pi/180.0)
  vy = -v0*math.sin(angle*math.pi/180.0)
  px = px+vx*dt
  py = py+vy*dt
  vy = vy+g*dt
  dx = int(px)    # 横方向の移動距離
  dy = int(py)    # 縦方向の移動距離
  return dx,dy

def touch_judge(hand_x,hand_y,dx,dy,fore_img,image):
      h, w = fore_img.shape[:2]
 
      back_h, back_w = image.shape[:2]
      back_x_min, back_x_max = dx, dx+w
      back_y_min, back_y_max = dy, dy+h
      
      if(back_x_min<hand_x and hand_x < back_x_max and back_y_min < hand_y and hand_y < back_y_max):
         return True
      else:
         return False
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video_path', type=str, default='', help='Path to the video file.')
args = parser.parse_args()
if args.video_path != '':
  cap = cv2.VideoCapture(args.video_path)
else:
  cap = cv2.VideoCapture(0)
px=400
py=350
g = 9.8
v0 = 50
angle = 45.0
dt = 0.1

cat_touched = False
cat_firstflag = True
hand_velocity = 0
previous_hands_pos = [0,0]
now_hands_pos = [0,0] 
previous_time = 0
now_time = 0
dist = 0

vx = v0*math.cos(angle*math.pi/180.0)
vy = -v0*math.sin(angle*math.pi/180.0)
fore_img = cv2.imread("mediapipe\data\cat.png")
fore_img = cv2.resize(fore_img, (100, 150))
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
 
      # 検出されたポーズの骨格をカメラ画像に重ねて描画
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if hands_results.multi_hand_landmarks:
        x=hands_results.multi_hand_landmarks[0].landmark[1].x
        y=hands_results.multi_hand_landmarks[0].landmark[1].y
        height, weight = image.shape[:2]
        #print(x*weight,y*height)
        previous_hands_pos = now_hands_pos
        now_hands_pos = [x*weight,y*height]
        dist = distance.euclidean(previous_hands_pos, now_hands_pos)
        #print(dist)
        previous_time = now_time
        now_time = time.perf_counter()

        hand_velocity = dist/(now_time - previous_time)

        print(hand_velocity)
        if(touch_judge(x*weight,y*height,dx,dy,fore_img,image)):
           cat_touched = True
        '''
        for hand_landmarks in hands_results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          cv2.drawMarker(image,(100,100),(0,0,0),markerType=cv2.MARKER_STAR, markerSize=10)
        '''
      mp_drawing.draw_landmarks(
          image,
          pose_results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      dx = int(px)    # 横方向の移動距離
      dy = int(py)    # 縦方向の移動距離

      if(cat_touched):
        if(cat_firstflag):
          print("awrawawrarw")
          print(hand_velocity)
          cat_vec = hand_velocity/10
          cat_firstflag = False
          vx = -cat_vec*math.cos(angle*math.pi/180.0)
          vy = -cat_vec*math.sin(angle*math.pi/180.0)
        #x,y = move_img()
        #画像の位置変更
        px = px+vx*dt
        py = py+vy*dt
        vx = vx
        vy = vy+g*dt
      image = comp(fore_img,image,dx,dy)
      cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
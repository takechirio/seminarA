import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
 
# Webカメラから入力
px=100
y=100
cap = cv2.VideoCapture(0)
fore_img = cv2.imread("mediapipe\data\cat.png")
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
      px+=2
      fore_img = cv2.resize(fore_img, (50, 75))
      h, w = image.shape[:2]
      dx = 250    # 横方向の移動距離
      dy = 100    # 縦方向の移動距離
      M = np.array([[1, 0, px], [0, 1, y]], dtype=float)
      image = cv2.warpAffine(fore_img, M, (w, h), image, borderMode=cv2.BORDER_TRANSPARENT)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      pose_results = pose.process(image)
      hands_results = hands.process(image)
 
      # 検出されたポーズの骨格をカメラ画像に重ねて描画
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if hands_results.multi_hand_landmarks:
        x=hands_results.multi_hand_landmarks[0].landmark[1].x
        y=hands_results.multi_hand_landmarks[0].landmark[1].y
        print(x,y)
        for hand_landmarks in hands_results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          cv2.drawMarker(image,(100,100),(0,0,0),markerType=cv2.MARKER_STAR, markerSize=10)
      mp_drawing.draw_landmarks(
          image,
          pose_results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
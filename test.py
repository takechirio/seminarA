import cv2
import mediapipe as mp
import numpy as np
import math
from scipy.spatial import distance
import time
import argparse

fore_img = cv2.imread(r"data\ball.png")
fore_img = cv2.resize(fore_img, (100, 100))
cv2.imshow('MediaPipe Pose', fore_img)
cv2.waitKey()
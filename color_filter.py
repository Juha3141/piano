import cv2
import mediapipe as mp
import sysv_ipc
import os , sys , pickle
import argparse , sys
import numpy as np

from struct import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

parser = argparse.ArgumentParser()
parser.add_argument('--video' , help="Select the video device(default : /dev/video0)" , default="/dev/video0")
parser.add_argument('--debug' , help="Select whether the program runs on debug mode" , default="0")
args = parser.parse_args()

lower_g_h = 30
lower_g_s = 0
lower_g_v = 0

upper_g_h = 80
upper_g_s = 255
upper_g_v = 255

def flgh(x):
    global lower_g_h
    lower_g_h = x

def flgs(x):
    global lower_g_s
    lower_g_s = x
def flgv(x):
    global lower_g_v
    lower_g_v = x
def fugh(x):
    global upper_g_h
    upper_g_h = x
def fugs(x):
    global upper_g_s
    upper_g_s = x
def fugv(x):
    global upper_g_v
    upper_g_v = x


with mp_hands.Hands(
    model_complexity = 0 ,
    max_num_hands = 2 , 
    min_tracking_confidence = 0.4 ,  
    min_detection_confidence = 0.4
) as hands:
    print(f"current pid : {os.getpid()}")
    debug = True
    cap = cv2.VideoCapture(args.video)

    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if video_width == 0:
        print("failed reading the video frame!")
        exit(-1)
    
    width = 800
    height = int(video_height*(width/video_width))
    print(f"width = {width}\nheight = {height}")
    if width == 0 or height == 0:
        print("Failed reading the video device!")
        exit(-1)

    cv2.namedWindow("mask")
    cv2.createTrackbar("lower_g_h" , "mask" , 0 , 255 , flgh)
    cv2.createTrackbar("lower_g_s" , "mask" , 0 , 255 , flgs)
    cv2.createTrackbar("lower_g_v" , "mask" , 0 , 255 , flgv)
    cv2.createTrackbar("upper_g_h" , "mask" , 0 , 255 , fugh)
    cv2.createTrackbar("upper_g_s" , "mask" , 0 , 255 , fugs)
    cv2.createTrackbar("upper_g_v" , "mask" , 0 , 255 , fugv)

    # 30 0   0
    # 80 255 255

    while cap.isOpened():
        ret , frame = cap.read()
        frame = cv2.resize(frame , (width , height));
        # frame = cv2.flip(frame , 1)
        hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)

        lower_green = np.array([lower_g_h , lower_g_s , lower_g_v])
        upper_green = np.array([upper_g_h , upper_g_s , upper_g_v])
        print(lower_g_h)
        mask = cv2.inRange(hsv , lower_green , upper_green)

        cv2.imshow("mask" , mask)

        cv2.imshow("camera" , frame)
        if debug and cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

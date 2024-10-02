import cv2
import mediapipe as mp
import sysv_ipc
import os , sys , pickle
import argparse , sys

from struct import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

parser = argparse.ArgumentParser()
parser.add_argument('--video' , help="Select the video device(default : /dev/video0)" , default="/dev/video0")
parser.add_argument('--debug' , help="Select whether the program runs on debug mode" , default="0")
args = parser.parse_args()

with mp_hands.Hands(
    model_complexity = 0 ,
    max_num_hands = 2 , 
    min_tracking_confidence = 0.4 ,  
    min_detection_confidence = 0.4
) as hands:
    print(f"current pid : {os.getpid()}")
    debug = args.debug == '1'
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

    memory_hand_info = sysv_ipc.SharedMemory(3141591)
    memory_img = sysv_ipc.SharedMemory(3141592)

    # Send the width, height and channel information
    memory_hand_info.write(pack("iii" , int(width) , int(height) , 3))
    while cap.isOpened():
        ret , frame = cap.read()
        frame = cv2.resize(frame , (width , height));
        # frame = cv2.flip(frame , 1)

        debug_frame = frame
        if not ret:
            print("Cannot receive frame!")
            break
        results = hands.process(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB))

        left_hand_landmarks_coords_x = [0]*21
        left_hand_landmarks_coords_y = [0]*21
        right_hand_landmarks_coords_x = [0]*21
        right_hand_landmarks_coords_y = [0]*21
        
        if results.multi_hand_landmarks:
            hand_type = []
            for hand in results.multi_handedness:
                hand_type.append(hand.classification[0].label)
            
            for (k , hand_landmarks) in enumerate(results.multi_hand_landmarks):
                for i , l in enumerate(hand_landmarks.landmark):
                    x = int(l.x*frame.shape[1])
                    y = int(l.y*frame.shape[0])
                    if hand_type[k] == "Left":
                        left_hand_landmarks_coords_x[i] = x
                        left_hand_landmarks_coords_y[i] = y
                    elif hand_type[k] == "Right":
                        right_hand_landmarks_coords_x[i] = x
                        right_hand_landmarks_coords_y[i] = y

                    text_coord = (10 , 10) if hand_type[k] == "Left" else (100 , 10)
                    color = (255 , 0 , 0) if hand_type[k] == "Left" else (0 , 0 , 255)
                    if debug == True:
                        cv2.circle(debug_frame , (x , y) , 5 , color , -1)
                        cv2.putText(debug_frame , str(i) , (x+10 , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , color , 2)
                    cv2.putText(frame , hand_type[k] , text_coord , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , color , 2)
        if debug: cv2.imshow("camera" , debug_frame)

        # write the frame to the shared memory
        memory_img.write(frame)

        # write the landmarks information to the shared
        hand_info_struct = pack("iii" , int(width) , int(height) , 3)
        hand_info_struct += pack("ii" , int(sum(left_hand_landmarks_coords_x)+sum(left_hand_landmarks_coords_y) != 0) , 
                                        int(sum(right_hand_landmarks_coords_x)+sum(right_hand_landmarks_coords_y) != 0))
        hand_info_struct += pack(f"{len(left_hand_landmarks_coords_x)}i" , *left_hand_landmarks_coords_x)
        hand_info_struct += pack(f"{len(left_hand_landmarks_coords_y)}i" , *left_hand_landmarks_coords_y)
        hand_info_struct += pack(f"{len(right_hand_landmarks_coords_x)}i" , *right_hand_landmarks_coords_x)
        hand_info_struct += pack(f"{len(right_hand_landmarks_coords_y)}i" , *right_hand_landmarks_coords_y)
        memory_hand_info.write(hand_info_struct)

        if debug and cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
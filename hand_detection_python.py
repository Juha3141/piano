import cv2
import mediapipe as mp
import sysv_ipc
import os , sys , pickle
import argparse , sys

from struct import *

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.holistic
hands = mp_hands.Holistic()

parser = argparse.ArgumentParser()
parser.add_argument('--video' , help="Select the video device(default : /dev/video0)" , default="/dev/video0")
parser.add_argument('--debug' , help="Select whether the program runs on debug mode" , default="0")
args = parser.parse_args()

def main():
    debug = args.debug == '1'
    cap = cv2.VideoCapture(args.video)

    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if video_width == 0:
        print("failed reading the video frame!")
        return
    
    width = 800
    height = int(video_height*(width/video_width))
    print(f"width = {width}\nheight = {height}")
    if width == 0 or height == 0:
        print("Failed reading the video device!")
        return 

    memory_hand_info = sysv_ipc.SharedMemory(3141591)
    memory_img = sysv_ipc.SharedMemory(3141592)

    # Send the width, height and channel information
    memory_hand_info.write(pack("iii" , int(width) , int(height) , 3))

    while cap.isOpened():
        ret , frame = cap.read()
        frame = cv2.resize(frame , (width , height));
        if not ret:
            print("Cannot receive frame!")
            break

        results = hands.process(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB))

        left_hand_landmarks_coords_x = [0]*21
        left_hand_landmarks_coords_y = [0]*21
        right_hand_landmarks_coords_x = [0]*21
        right_hand_landmarks_coords_y = [0]*21
        
        if results.left_hand_landmarks:
            for i , l in enumerate(results.left_hand_landmarks.landmark):
                x = int(l.x*frame.shape[1])
                y = int(l.y*frame.shape[0])
                # cv2.circle(frame , (x , y) , 5 , (0 , 255 , 0))
                # cv2.putText(frame , str(i) , (x+10 , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 255))

                left_hand_landmarks_coords_x[i] = x
                left_hand_landmarks_coords_y[i] = y
        
        if results.right_hand_landmarks:
            for i , l in enumerate(results.right_hand_landmarks.landmark):
                x = int(l.x*frame.shape[1])
                y = int(l.y*frame.shape[0])
                # cv2.circle(frame , (x , y) , 5 , (0 , 255 , 0))
                # cv2.putText(frame , str(i) , (x+10 , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255 , 0 , 0))
                
                right_hand_landmarks_coords_x[i] = x
                right_hand_landmarks_coords_y[i] = y

        if debug: cv2.imshow("camera" , frame)

        # write the frame to the shared memory
        memory_img.write(frame)

        # write the landmarks information to the shared
        hand_info_struct = pack("iii" , int(width) , int(height) , 3)
        hand_info_struct += pack("ii" , int(results.left_hand_landmarks != None) , int(results.right_hand_landmarks != None))
        hand_info_struct += pack(f"{len(left_hand_landmarks_coords_x)}i" , *left_hand_landmarks_coords_x)
        hand_info_struct += pack(f"{len(left_hand_landmarks_coords_y)}i" , *left_hand_landmarks_coords_y)
        hand_info_struct += pack(f"{len(right_hand_landmarks_coords_x)}i" , *right_hand_landmarks_coords_x)
        hand_info_struct += pack(f"{len(right_hand_landmarks_coords_y)}i" , *right_hand_landmarks_coords_y)
        memory_hand_info.write(hand_info_struct)

        if debug and cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#include <iostream>
#include <cstdlib>
#include <sys/types.h>
#include <sys/shm.h>
#include <pthread.h>

#include <opencv2/opencv.hpp>

using namespace cv;

struct hand_img_info {
    int width;
    int height;
    int channels;

    int left_hand_visible;
    int right_hand_visible;

    int left_hand_landmarks_xlist[21];
    int left_hand_landmarks_ylist[21];
    int right_hand_landmarks_xlist[21];
    int right_hand_landmarks_ylist[21];
};

void *execute_python(void *data) {
    std::system("python3 hand_detection_python.py --video /dev/video2 --debug 1");
    // signal the main process that the program has ended
    ((bool*)data)[0] = true;

    pthread_exit(0x00);
}

int main(int argc , char **argv) {
    int img_width = 0 , img_height = 0 , img_channels = 0;
    const int info_mem_size = 4096;
    const int shm_info_key = 3141591;
    const int shm_img_key = 3141592;

    pthread_t python_program;

    bool python_ended = false;

    void *addr_img , *addr_info;
    int shmid_info = shmget(shm_info_key , info_mem_size , IPC_CREAT|0666);
    addr_info = shmat(shmid_info , 0x00 , 0);
    if(addr_info == (void *)-1) {
        std::cout << "addr_info assignment failed!\n";
        return -1;
    }
    int python_thread_id = pthread_create(&python_program , 0x00 , execute_python , &python_ended);
    if(python_thread_id < 0) {
        std::cout << "Failed creating the thread for python!\n";
        shmdt(addr_info);
        return -1;
    }

    // addr_info packet : contains myriads of information from the resolution of image to the landmarks of the hand
    std::cout << "waiting for the connection ...\n";
    memset(addr_info , 0 , info_mem_size);
    // wait for the data to arrive
    while(((unsigned long *)addr_info)[0] == 0) { }
    img_width = ((struct hand_img_info *)addr_info)->width;
    img_height = ((struct hand_img_info *)addr_info)->height;
    img_channels = ((struct hand_img_info *)addr_info)->channels;
    std::cout << "img_width = " << img_width << "\n";
    std::cout << "img_height = " << img_height << "\n";
    std::cout << "img_channels = " << img_channels << "\n";

    // open the shared memory for the image
    int padding = 512;
    int shmid_img = shmget(shm_img_key , 1024*1024 , IPC_CREAT|0666);

    addr_img = shmat(shmid_img , 0x00 , 0);
    if(addr_img == (void *)-1) {
        std::cout << "addr_img assignment failed!\n";
        shmdt(addr_info);
        return -1;
    }

    memset(addr_img , 0 , img_width*img_height*img_channels);
    namedWindow("win1");
    while(1) {
        if(((unsigned long *)addr_img)[0] == 0x00) continue;

        Mat frame = Mat(Size(640 , 480) , CV_8UC3);
        struct hand_img_info *hands_info = (struct hand_img_info *)addr_info;

        // copy the image to the frame
        memcpy(frame.ptr(0) , addr_img , 640*480*3);

        if(hands_info->right_hand_visible) { 
            for(int i = 0; i < 21; i++) {
                int x = hands_info->right_hand_landmarks_xlist[i] , y = hands_info->right_hand_landmarks_ylist[i];
                std::cout << "right_hand(" << i << ") = " << x << " , " << y << "\n";
                circle(frame , Point(x , y) , 2 , Scalar(0x00 , 0xff , 0x00) , 1);
            }
        }

        if(hands_info->left_hand_visible) { 
            for(int i = 0; i < 21; i++) {
                int x = hands_info->left_hand_landmarks_xlist[i] , y = hands_info->left_hand_landmarks_ylist[i];
                std::cout << "left_hand(" << i << ") = " << x << " , " << y << "\n";
                circle(frame , Point(x , y) , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
            }
        }
        
        imshow("win1" , frame);
        if(python_ended == true) { std::cout << "thread ended! exiting...\n"; break; }
        if(waitKey(1) == 27) break;
    }
    
    pthread_detach(python_program);
    shmdt(addr_info);
    shmdt(addr_img);
}
#ifndef _HAND_DETECTION_AGENT_HPP_
#define _HAND_DETECTION_AGENT_HPP_

#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <signal.h>

#include <opencv2/opencv.hpp>

typedef struct {
    int width;
    int height;
    int channels;

    int left_hand_visible;
    int right_hand_visible;

    int left_hand_landmarks_xlist[21];
    int left_hand_landmarks_ylist[21];
    int right_hand_landmarks_xlist[21];
    int right_hand_landmarks_ylist[21];
}hands_info_t;

struct GlobalHandDetectionSystem {
    // singleton design
    static GlobalHandDetectionSystem *get_self(void) {
        static GlobalHandDetectionSystem *p = 0x00;
        if(!p) p = new GlobalHandDetectionSystem;
        return p;
    }
    char video_device[64];

    pid_t agent_process_id;

    int screen_width;
    int screen_height;
    int screen_channels;

    void *shm_addr_imgmem = 0; // address to the SHM of image
    void *shm_addr_infomem = 0;  // address to the SHM of info structure
};

namespace hand_detection {
    // ai agent, which is the python program
    void *ai_agent(void *data);

    bool initialize_agent(const char *video_device);
    void execute_agent(void);
    bool fetch_hand_data(cv::Mat &current_frame , hands_info_t &hands);
    bool check_agent_running(void);

    void end(void);
}

#endif
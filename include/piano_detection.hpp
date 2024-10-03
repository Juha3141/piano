#ifndef _PIANO_DETECTION_HPP_
#define _PIANO_DETECTION_HPP_

#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <string>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define PIANO_KEY_C      1
#define PIANO_KEY_Csharp 2
#define PIANO_KEY_Dflat  2
#define PIANO_KEY_D      3
#define PIANO_KEY_Dsharp 4
#define PIANO_KEY_Eflat  4
#define PIANO_KEY_E      5
#define PIANO_KEY_F      6
#define PIANO_KEY_Fsharp 7
#define PIANO_KEY_Gflat  7
#define PIANO_KEY_G      8
#define PIANO_KEY_Gsharp 9
#define PIANO_KEY_Aflat  9
#define PIANO_KEY_A      10
#define PIANO_KEY_Asharp 11
#define PIANO_KEY_Bflat  11
#define PIANO_KEY_B      12

#define PIANO_KEY_TYPE_LEFT   1
#define PIANO_KEY_TYPE_MIDDLE 2
#define PIANO_KEY_TYPE_RIGHT  3
#define PIANO_KEY_TYPE_NONE   4

struct piano_keys_info {
    cv::Mat piano_image;
    // false : black keys upper than white keys
    // true  : black keys lower than white keys
    bool flipped;

    // List of RotatedRects of the keys
    std::vector<cv::RotatedRect>keys_rectangle_list;
    std::vector<cv::Point>keys_rectangle_pivot;
    std::vector<std::vector<std::pair<cv::RotatedRect , int>>*>separated_keys_rectangle_list;

    // mean width, height, area of the keys
    double mean_key_width;
    double mean_key_height;
    double median_key_width;
    double median_key_height;
    
    double mean_key_y;
    double std_dev_key_y;

    // Best fit line of center of masses of keys
    // Best fit line : y=bx+a
    double cm_bestfit_b , cm_bestfit_a;

    // List of the distance between the keys' center of masses and best fit line
    std::vector<double>cm_dist_from_bestfit_list;
    double mean_dist_from_bestfit;
    // Distance between #0 and #1 = #0
    std::vector<double>dist_between_keys_list;
    double mean_dist_between_keys;
    double median_dist_between_keys;

    // first : note of the key , second : 0-52 index of the note
    std::vector<std::pair<int , int>>key_notes;

/* for white keys */

    // List of the key notes and shapes
    // shape of the white key
    std::vector<int>white_key_shapes;
};

#define NOTE(n)   ((n) & 0x0f)
#define OCTAVE(n) ((n) >> 4)

void get_min_max_x_point(std::vector<cv::Point> &cont_obj , cv::Point &min_x , cv::Point &max_x);
void get_min_max_y_point(std::vector<cv::Point> &cont_obj , cv::Point &min_y , cv::Point &max_y);

void create_features_info(cv::Mat img , std::vector<cv::KeyPoint>&keypoints , cv::Mat &descriptors);
void image_detection(cv::Mat img , std::vector<cv::KeyPoint>&keypoints);
void get_bounding_rect_contour(const std::vector<cv::Point>&contour , std::vector<cv::Point>&bounding_rect);
void rotated_rect_to_contour(const cv::RotatedRect &rect , std::vector<cv::Point>&contour);
void relocate_rotated_rect_list(std::vector<cv::RotatedRect>&rect_list , int x , int y);
double calculate_median(std::vector<double>&data_list);
double calculate_standard_deviation(std::vector<double>&data_list , double mean);

double euclidean_distance(cv::Point p1 , cv::Point p2);

typedef void(*filter_func_t)(cv::Mat &filter);

class PianoRecognition {
    public:
        PianoRecognition(const char *default_video , int m , int w , int h , int f) { set_video_input(default_video , m , w , h , f); }
        PianoRecognition(const char *default_video , int m , int w , int h , int f , cv::Mat template_piano) {
            set_video_input(default_video , m , w , h , f);
            set_template_piano_image(template_piano);
        }
        PianoRecognition(const char *default_video , cv::Mat template_piano) : video_input({0 , 0 , 0 , 0}) {
            set_template_piano_image(template_piano); 
            is_piano_detected = false;
        }

        void set_frame_filter(filter_func_t filter) {
            filter_function = filter;
        }
        
        // major processes
        bool process_piano_calibration(void);

        bool recognize_piano(cv::Mat img , std::vector<cv::Point>&contour);
        
        // black keys are more easy to recognize --> Recognize black keys fist and recognize the white key
        void detect_black_keys(cv::Mat piano_image , struct piano_keys_info &keys_info);
        void detect_white_keys(cv::Mat piano_image , struct piano_keys_info &keys_info , const struct piano_keys_info &black_keys_info);
        
        void adjust_key_angles(struct piano_keys_info &keys_info);
        void remove_key_outliers(struct piano_keys_info &keys_info);
        void write_keys_info(struct piano_keys_info &keys_info);

        void detect_missing_white_keys(struct piano_keys_info &keys_info);
        void detect_missing_black_keys(struct piano_keys_info &black_keys_info , struct piano_keys_info &white_keys_info);
        void detect_white_key_shapes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);
        void detect_white_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);
        void detect_black_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);

        void doublecheck_white_keys(struct piano_keys_info &white_keys_info);
        void doublecheck_black_keys(struct piano_keys_info &black_keys_info);

        void fill_missing_white_keys(struct piano_keys_info &white_keys_info);

        // deprecated
        // void fill_missing_white_keys(struct piano_keys_info &white_keys_info);
        void separate_piano_sections(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);

        inline void set_template_piano_image(cv::Mat template_piano) {
            this->template_piano = template_piano;
            create_features_info(template_piano , template_keypoints , template_descriptors);
        }
        inline void set_video_input(const char *default_video_input , int mode , int width , int height , int fps) {
            strcpy(video_input.device , default_video_input);
            video_input.mode = mode;
            video_input.width = width;
            video_input.height = height;
            video_input.fps = fps;
        }
        inline void open_video_capture(cv::VideoCapture &video) {
            video = cv::VideoCapture(this->video_input.device , this->video_input.mode);
            if(this->video_input.width != 0)  { video.set(cv::CAP_PROP_FRAME_WIDTH , this->video_input.width); }
            if(this->video_input.height != 0) { video.set(cv::CAP_PROP_FRAME_HEIGHT , this->video_input.height); }
            if(this->video_input.fps != 0)    { video.set(cv::CAP_PROP_FPS , this->video_input.fps); }
        }
        inline void set_detected_piano_img(const cv::Mat bounding_rect_img , const std::vector<cv::Point>&bounding_contour , const cv::RotatedRect &bounding_rect) {
            // copy the frame
            bounding_rect_img.copyTo(this->piano_bounding_rect_img);
            // copy the contour
            std::copy(bounding_contour.begin() , bounding_contour.end() , std::back_inserter(this->piano_bounding_contour));
            // copy the bounding rect
            memcpy(&this->piano_bounding_rect , &bounding_rect , sizeof(cv::RotatedRect));
            
            is_piano_detected = true;
        }

        static cv::Mat filter_piano_image(cv::Mat frame);

        bool is_piano_detected = false;

        int piano_loc_x , piano_loc_y;
        // the raw, unprocessed frame that contains the image of piano
        cv::Mat piano_bounding_rect_img;
        // the bounding contour 
        std::vector<cv::Point>piano_bounding_contour;
        cv::RotatedRect piano_bounding_rect;
        
    private:
        struct {
            char device[40];
            int mode;
            int width;
            int height;
            int fps;
        }video_input;

        cv::Mat template_piano;
        std::vector<cv::KeyPoint>template_keypoints;
        cv::Mat template_descriptors;

        filter_func_t filter_function;
};

const char *number_to_note_string(int note_number);

#endif
#ifndef _PIANO_AREA_DETECTION_HPP_
#define _PIANO_AREA_DETECTION_HPP_

#include <piano_detection.hpp>

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

#endif
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

typedef struct piano_keys_info_s {
    // RotatedRect of the keys
    std::vector<cv::RotatedRect>keys_rectangle_list;

    double mean_key_width;
    double mean_key_height;
    double median_key_width;
    double median_key_height;

    // Best fit line of center of masses of keys
    // Best fit line : y=bx+a
    double cm_bestfit_b , cm_bestfit_a;

    // first : note of the key , second : 0-52 (Or 0-36 for black) index of the note
    std::vector<std::pair<int , int>>key_notes;
    
    // list of pivot points of the rectangles
    std::vector<cv::Point>keys_rectangle_pivot;

    // Distance between #0 and #1 = #0
    std::vector<double>dist_between_keys_list;
}piano_keys_info_t;

typedef struct white_piano_keys_info_s : piano_keys_info_t {
    // shape of the white key
    std::vector<int>white_key_shapes;
    std::vector<std::pair<int , int>>missing_key_spots_list;
    std::vector<std::pair<int , int>>missing_key_count_list;
}white_piano_keys_info_t;

typedef struct black_piano_keys_info_s : piano_keys_info_t {

}black_piano_keys_info_t;

struct PianoInfo {
    public:
        inline void set_piano_image(cv::Mat &img) { img.copyTo(piano_image); }

        /***** Key detection *****/
        void detect_black_keys(void);
        void detect_white_keys(void);
        
        // white keys
        void white_key_adjust_angles(piano_keys_info_t &keys_info);
        void detect_white_missing_spots(void);
        
        static void adjust_key_angles(piano_keys_info_t &keys_info);
        static void adjust_key_widths(piano_keys_info_t &keys_info);

        /***** Note detection *****/
        
        // white keys
        void detect_white_key_shapes(void);
        void detect_white_key_notes(void);
        void adjust_wrong_white_notes(void);
        void fill_missing_white_keys(void);

        // black keys
        void detect_black_key_notes(void);

        cv::Mat piano_image;
        cv::RotatedRect piano_bounding_rect;
        bool flipped;

        white_piano_keys_info_t white_keys_info;
        black_piano_keys_info_t black_keys_info;

    private:
        void create_white_adjusted_cm_list(void);
        // bool  : true = white, false = black
        // Point : center of mass point
        // int   : index based on each keys_rectangle_list[] array
        std::vector<std::tuple<bool , cv::Point , int>>key_adjusted_cm_list;
};

typedef enum white_or_black_e {
    white = 1 , 
    black = 0
}white_or_black_t;

typedef void(*filter_func_t)(cv::Mat &filter);

#define NOTE(n)   ((n) & 0x0f)
#define OCTAVE(n) ((n) >> 4)

// feature detection stuff
void create_features_info(cv::Mat img , std::vector<cv::KeyPoint>&keypoints , cv::Mat &descriptors);
void image_detection(cv::Mat img , std::vector<cv::KeyPoint>&keypoints);

// bounding rect stuff
void get_bounding_rect_contour(const std::vector<cv::Point>&contour , std::vector<cv::Point>&bounding_rect);
void rotated_rect_to_contour(const cv::RotatedRect &rect , std::vector<cv::Point>&contour);
void relocate_rotated_rect_list(std::vector<cv::RotatedRect>&rect_list , int dx , int dy);

// arithmetic operations
double calculate_median(std::vector<double>&data_list);
double calculate_percentile(std::vector<double>&data_list , double percentile);
double calculate_standard_deviation(std::vector<double>&data_list , double mean);
double euclidean_distance(cv::Point p1 , cv::Point p2);

double rotational_matrix_x(double x , double y , double theta , double x0 , double y0);
double rotational_matrix_y(double x , double y , double theta , double x0 , double y0);
cv::Point2f rotational_matrix(cv::Point2f p , double theta , cv::Point2f pivot);

// rotated rect stuff
void adjust_rotated_rect_height(cv::RotatedRect &rect , int new_height , bool direction=true);
void adjust_rotated_rect_width(cv::RotatedRect &rect , int new_width , bool direction=true);

// piano keys info stuff
void write_keys_info(piano_keys_info_t &keys_info);
void write_pivot_info(PianoInfo &piano_info , white_or_black_t white_or_black);

// piano note stuff
const char *number_to_note_string(int note_number);

#endif
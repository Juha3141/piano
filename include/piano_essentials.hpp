#ifndef _PIANO_ESSENTIALS_HPP_
#define _PIANO_ESSENTIALS_HPP_

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
    cv::RotatedRect piano_bounding_rect;
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

typedef void(*filter_func_t)(cv::Mat &filter);

#define NOTE(n)   ((n) & 0x0f)
#define OCTAVE(n) ((n) >> 4)

void create_features_info(cv::Mat img , std::vector<cv::KeyPoint>&keypoints , cv::Mat &descriptors);
void image_detection(cv::Mat img , std::vector<cv::KeyPoint>&keypoints);
void get_bounding_rect_contour(const std::vector<cv::Point>&contour , std::vector<cv::Point>&bounding_rect);
void rotated_rect_to_contour(const cv::RotatedRect &rect , std::vector<cv::Point>&contour);
void relocate_rotated_rect_list(std::vector<cv::RotatedRect>&rect_list , int x , int y);
double calculate_median(std::vector<double>&data_list);
double calculate_percentile(std::vector<double>&data_list , double percentile);
double calculate_standard_deviation(std::vector<double>&data_list , double mean);
double euclidean_distance(cv::Point p1 , cv::Point p2);

void adjust_rotated_rect_height(cv::RotatedRect &rect , int new_height , bool direction=true);
void adjust_rotated_rect_width(cv::RotatedRect &rect , int new_width , bool direction=true);

void write_keys_info(struct piano_keys_info &keys_info);
void write_pivot_info(struct piano_keys_info &keys_info);

const char *number_to_note_string(int note_number);

double rotational_matrix_x(double x , double y , double theta , double x0 , double y0);
double rotational_matrix_y(double x , double y , double theta , double x0 , double y0);
cv::Point2f rotational_matrix(cv::Point2f p , double theta , cv::Point2f pivot);

#endif
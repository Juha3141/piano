#include <piano_detection.hpp>

using namespace cv;
using namespace xfeatures2d;

void create_features_info(Mat img , std::vector<KeyPoint>&keypoints , Mat &descriptors) {
    Ptr<ORB>detector = ORB::create(img.size().width/2);
    detector->detectAndCompute(img , Mat() , keypoints , descriptors);
}

void get_min_max_x_point(std::vector<Point> &cont_obj , Point &min_x , Point &max_x) {
    std::vector<Point>::iterator min_x_it = std::min_element(cont_obj.begin() , cont_obj.end() , [](const auto &a , const auto &b) { return (bool)(a.x < b.x); });
    std::vector<Point>::iterator max_x_it = std::max_element(cont_obj.begin() , cont_obj.end() , [](const auto &a , const auto &b) { return (bool)(a.x < b.x); });
    min_x = *min_x_it;
    max_x = *max_x_it;
}

void get_min_max_y_point(std::vector<Point> &cont_obj , Point &min_y , Point &max_y) {
    std::vector<Point>::iterator min_y_it = std::min_element(cont_obj.begin() , cont_obj.end() , [](const auto &a , const auto &b) { return (bool)(a.y < b.y); });
    std::vector<Point>::iterator max_y_it = std::max_element(cont_obj.begin() , cont_obj.end() , [](const auto &a , const auto &b) { return (bool)(a.y < b.y); });
    min_y = *min_y_it;
    max_y = *max_y_it;
}

void get_bounding_rect_contour(const std::vector<cv::Point>&contour , std::vector<cv::Point>&bounding_rect) {
    Point2f contour_minarea[4];
    RotatedRect rotated_rect = minAreaRect(contour);
    rotated_rect.points(contour_minarea);
    for(int i = 0; i < 4; i++) { bounding_rect.push_back(contour_minarea[i]); }
}

void rotated_rect_to_contour(const RotatedRect &rect , std::vector<Point>&contour) {
    Point2f contour_minarea[4];
    rect.points(contour_minarea);
    for(int i = 0; i < 4; i++) { contour.push_back(contour_minarea[i]); }
}

void relocate_rotated_rect_list(std::vector<cv::RotatedRect>&rect_list , int x , int y) {
    for(int i = 0; i < rect_list.size(); i++) {
        rect_list[i].center.x += x;
        rect_list[i].center.y += y;
    }
}

double euclidean_distance(Point p1 , Point p2) {
    return sqrt(pow((p1.x-p2.x) , 2)+pow((p1.y-p2.y) , 2));
}

double calculate_median(std::vector<double>&data_list) {
    std::sort(data_list.begin() , data_list.end());
    return (data_list.size()%2)
        ? (data_list[(data_list.size()+1)/2])
        : ((data_list[(data_list.size()+1)/2]+data_list[((data_list.size()+1)/2)+1])/2);
}

/// @brief Calculate the percentile from the list of data
/// @param data_list data list
/// @param percentile 0 ~ 1
/// @return value of the percentile
double calculate_percentile(std::vector<double>&data_list , double percentile) {
    std::sort(data_list.begin() , data_list.end());
    double position = percentile*(data_list.size()+1);
    double between = position-((double)((int)position));
    double first = data_list[floor(position)-1] , second = data_list[ceil(position)-1];
    return first+between*(second-first);
}

double calculate_standard_deviation(std::vector<double>&data_list , double mean) {
    double standard_deviation = 0;
    for(double x : data_list) {
        double diff = x-mean;
        standard_deviation += (diff*diff);
    }
    return sqrt(standard_deviation/(double)data_list.size());
}

/// @brief Filter the image to binary for contour processing
/// @param frame original frame
/// @return filtered frame
cv::Mat PianoRecognition::filter_piano_image(cv::Mat frame) {
    Mat blurred_frame , grayscaled_frame , binary_frame;
    medianBlur(frame , blurred_frame , 11);
    if(blurred_frame.type() != CV_8UC1) cvtColor(blurred_frame , grayscaled_frame , COLOR_RGB2GRAY);
    else grayscaled_frame = blurred_frame;
    
    threshold(grayscaled_frame , binary_frame , -1 , 255 , THRESH_OTSU|THRESH_BINARY);
    return binary_frame;
}

bool PianoRecognition::process_piano_calibration(void) {
    VideoCapture video;
    this->open_video_capture(video);
    namedWindow("win1");
    std::vector<Point>contour;
    Rect bounding_rect;
    std::vector<Point>prev_contour;
    Rect prev_bounding_rect;

    int overlap_streak = 0;
    const int overlap_streak_threshold = 40;
    
    Mat debug_copy;
    while(1) {
        Mat frame;
        if(video.read(frame) == false) {
            std::cout << "video error!\n";
            return false;
        }
        this->filter_function(frame);
        int width = 800;
        std::cout << "resized : " << Size(width , ((float)frame.size().height*((float)width/(float)frame.size().width))) << "\n";
        resize(frame , frame , Size(width , ((float)frame.size().height*((float)width/(float)frame.size().width))));

        // for debugging purpose
        
        frame.copyTo(debug_copy);

        if(recognize_piano(frame , contour) == false) {
            putText(debug_copy , "No piano found!" , Point(0 , 30) , FONT_HERSHEY_SIMPLEX , 1 , Scalar(0xff , 0x00 , 0x00) , 3);
            overlap_streak = std::max(overlap_streak-1 , 0);
        }
        else {
            bounding_rect = boundingRect(contour);

            // std::vector<std::vector<Point>>contour_list = {contour};
            // drawContours(debug_copy , contour_list , 0 , Scalar(0x00 , 0xff , 0x00) , 4);
            // rectangle(debug_copy , bounding_rect , Scalar(0x00 , 0x00 , 0xff) , 4);
        }

        double area_overlap = (bounding_rect & prev_bounding_rect).area();
        double area_current = bounding_rect.area();
        std::cout << "area_overlap = " << area_overlap << "\n";
        std::cout << "area_current = " << area_current << "\n";
        if(area_overlap >= area_current*0.90 && area_overlap != 0) {
            overlap_streak += 2;
            std::cout << " ---- overlap_streak : " << overlap_streak << "\n";
            
            // the piano is finally recognized
            if(overlap_streak >= overlap_streak_threshold) {
                cv::RotatedRect rotated_rect;
                rotated_rect = minAreaRect(contour);
                
                set_detected_piano_img(frame , contour , rotated_rect);
                break;
            }
        }
        
        imshow("win1" , debug_copy);
        if(waitKey(1) == 27) return false;

        // copy prev
        std::copy(contour.begin() , contour.end() , std::back_inserter(prev_contour));
        memcpy(&prev_bounding_rect , &bounding_rect , sizeof(Rect));
    }
    video.release();
    return true;
}

bool PianoRecognition::recognize_piano(Mat img , std::vector<Point>&contour) {
    std::vector<std::vector<Point>>contours;
    
    // match the object with the template
    std::vector<KeyPoint>keypoints_query;
    Mat descriptors_query;
    std::vector<std::vector<DMatch>>knn_matches;
    std::vector<DMatch>good_matches;

    Ptr<BFMatcher>matcher = BFMatcher::create(NORM_HAMMING);
    // use knn match
    create_features_info(img , keypoints_query , descriptors_query);
    matcher->knnMatch(descriptors_query , template_descriptors , knn_matches , 2);

    // only select the good matches
    const float match_thresh_ratio = 0.79;
    for(int i = 0; i < knn_matches.size(); i++) {
        if(knn_matches[i][0].distance < match_thresh_ratio*knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    // not match
    std::cout << "good_matches : " << good_matches.size() << "\n";
    if(good_matches.size() < 3) { return false; }
    
    // matched, find the contours
    Mat binary_img = PianoRecognition::filter_piano_image(img);
    findContours(binary_img , contours , RETR_LIST , CHAIN_APPROX_SIMPLE);

    // calculate how much matching features is contained in the contour
    int target_contour_index = 0;
    int max_hit = -1;
    Mat img_copy , out;
    img.copyTo(img_copy);
    for(int i = 0; i < contours.size(); i++) {
        std::vector<Point>cont_obj = contours.at(i);

        std::vector<Point>bounding_rect_contour;
        get_bounding_rect_contour(cont_obj , bounding_rect_contour);
        int hit = 0;
        for(DMatch m : good_matches) { hit += pointPolygonTest(bounding_rect_contour , keypoints_query[m.queryIdx].pt , false) > 0; }

        if(max_hit < hit) {
            target_contour_index = i;
            max_hit = hit;
        }

        std::vector<std::vector<Point>>cl = {bounding_rect_contour};
        drawContours(img_copy , cl , 0 , Scalar::all(0xff) , 2);
    }
    drawContours(img_copy , contours , target_contour_index , Scalar::all(0x88) , 2);
    namedWindow("winff2" , WINDOW_NORMAL);
    imshow("winff2" , img_copy);
    resizeWindow("winff2" , 1024 , 768);

    drawMatches(img_copy , keypoints_query , this->template_piano , this->template_keypoints , good_matches , out);
    namedWindow("winff" , WINDOW_NORMAL);
    imshow("winff" , out);
    resizeWindow("winff" , 1024 , 768);
    std::copy(contours[target_contour_index].begin() , contours[target_contour_index].end() , std::back_inserter(contour));
    
    return true;
}

#define DEBUG

/// @brief Write the keys info to the structure based on the informations provided
///        Necessary information consists: 
///         1. keys_info.keys_rectangle_list
/// @param keys_info 
void PianoRecognition::write_keys_info(struct piano_keys_info &keys_info) {
    // sort by x coordinate
    std::sort(keys_info.keys_rectangle_list.begin() , keys_info.keys_rectangle_list.end() , [](const auto &a , const auto &b) {
        return (a.center.x < b.center.x);
    });

    std::vector<Point>center_points;
    for(RotatedRect r : keys_info.keys_rectangle_list) { center_points.push_back(r.center); }

    std::cout << "Calculating the best fit line...\n";
    // calculate the best fits of the center masses
    if(center_points.size() == 0) {
        std::cout << "calculation failed! recalibration required..\n";
        return;
    }
    Vec4f best_fit_line;
    fitLine(center_points , best_fit_line , DIST_L2 , 0 , 0.01 , 0.01);
    double bestfit_vx = best_fit_line[0] , bestfit_vy = best_fit_line[1] , bestfit_x0 = best_fit_line[2] , bestfit_y0 = best_fit_line[3];
    double bestfit_b = bestfit_vy/bestfit_vx;
    double bestfit_a = -(bestfit_b*bestfit_x0)+bestfit_y0;

    keys_info.mean_dist_from_bestfit = 0;
    keys_info.mean_dist_between_keys = 0;
    keys_info.median_dist_between_keys = 0;
    keys_info.mean_key_y = 0;

    std::vector<double>dist_median_list , y_list;

    std::cout << "Calculating the distance between the line...\n";
    if(keys_info.keys_rectangle_list.size() <= 1) {
        std::cout << "No keys in the key_rectangle_list!!\n";
        return;
    }
    keys_info.cm_bestfit_b = bestfit_b;
    keys_info.cm_bestfit_a = bestfit_a;
    for(int i = 0; i < keys_info.keys_rectangle_list.size(); i++) {
        /* Calculate the center of mass's distance from the line */
        // use the formula : https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        Point center = keys_info.keys_rectangle_list[i].center;
        double distance_from_best_fit = abs(bestfit_b*center.x-center.y+bestfit_a)/sqrt((bestfit_b*bestfit_b)+1.0f);
        // push the distance to the list
        keys_info.cm_dist_from_bestfit_list.push_back(distance_from_best_fit);
        keys_info.mean_dist_from_bestfit += distance_from_best_fit;
        
        /* Calculate the distance between the keys(based on center of mass) */
        // skip the last one
        if(i >= keys_info.keys_rectangle_list.size()-1) { keys_info.dist_between_keys_list.push_back(-1); continue; }
        // distance between two centers
        double distance_between_keys = euclidean_distance(keys_info.keys_rectangle_list[i].center , keys_info.keys_rectangle_list[i+1].center);
        
        // y-coord
        y_list.push_back(keys_info.keys_rectangle_list[i].center.y);
        keys_info.mean_key_y += keys_info.keys_rectangle_list[i].center.y;

        keys_info.dist_between_keys_list.push_back(distance_between_keys);
        dist_median_list.push_back(distance_between_keys);
        keys_info.mean_dist_between_keys += distance_between_keys;
        std::cout << i << " : distance_between_keys = " << distance_between_keys << "\n";
    }
    keys_info.mean_key_y /= keys_info.keys_rectangle_list.size();
    keys_info.mean_dist_from_bestfit /= keys_info.keys_rectangle_list.size();
    keys_info.mean_dist_between_keys /= (keys_info.keys_rectangle_list.size()-1);

    keys_info.median_dist_between_keys = calculate_median(dist_median_list);
    keys_info.std_dev_key_y = calculate_standard_deviation(y_list , keys_info.mean_key_y);
    std::cout << "mean_dist_between_keys = " << keys_info.mean_dist_between_keys << "\n";
}

static void remove_item_from_piano_info(struct piano_keys_info &keys_info , int i) {
    keys_info.keys_rectangle_list.erase(keys_info.keys_rectangle_list.begin()+i);
    if(i > 0 && i < keys_info.cm_dist_from_bestfit_list.size()-1) {
        // remove two components and recalibrate the distance
        keys_info.dist_between_keys_list[i] = euclidean_distance(keys_info.keys_rectangle_list[i-1].center , keys_info.keys_rectangle_list[i+1].center);
    }
    keys_info.dist_between_keys_list.erase(keys_info.dist_between_keys_list.begin()+i-1);
    keys_info.cm_dist_from_bestfit_list.erase(keys_info.cm_dist_from_bestfit_list.begin()+i);
}

/// @brief Adjust the width of the rotated rectangle
/// @param rect 
/// @param new_width 
/// @param direction false : left, true : right
static void adjust_rotated_rect_width(RotatedRect &rect , int new_width , bool direction=true) {
    int d = (rect.size.width-new_width)/2;

    rect.center.x += (direction ? 1.0f : -1.0f)*d*cos(rect.angle*M_PI/180.0f);
    rect.center.y -= (direction ? 1.0f : -1.0f)*d*sin(rect.angle*M_PI/180.0f);

    rect.size.width = new_width;
}

/// @brief Adjust the outlier keys
/// @param keys_info piano keys info structure
void PianoRecognition::adjust_white_outliers(struct piano_keys_info &white_keys_info) {
    double angle_median;
    double angle_mean = 0;
    double angle_sd;
    std::vector<double>rect_angle_list;
    for(RotatedRect r : white_keys_info.keys_rectangle_list) { rect_angle_list.push_back(r.angle); angle_mean += r.angle; }
    angle_mean /= white_keys_info.keys_rectangle_list.size();
    angle_median = calculate_median(rect_angle_list);
    angle_sd = calculate_standard_deviation(rect_angle_list , angle_mean);

    Mat image_copy = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC3);
    cvtColor(white_keys_info.piano_image , image_copy , COLOR_GRAY2BGR);
    
    double Q1 = calculate_percentile(rect_angle_list , 0.25);
    double Q3 = calculate_percentile(rect_angle_list , 0.75);

    double left_whisker = Q1-1.5*(Q3-Q1);
    double right_whisker = Q3+1.5*(Q3-Q1);

    int region_count = 5;
    int angle_regional_mean[region_count];
    int region_size = white_keys_info.keys_rectangle_list.size()/region_count;
    int j = 0 , n = 0;
    angle_regional_mean[0] = 0;
    for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
        if(white_keys_info.keys_rectangle_list[i].angle > left_whisker && white_keys_info.keys_rectangle_list[i].angle < right_whisker) {
            angle_regional_mean[j] += white_keys_info.keys_rectangle_list[i].angle;
            std::cout << "region #" << j << " : " << i << "(" << white_keys_info.keys_rectangle_list[i].angle << ")\n";
            n++;
        }
        if((i+1)%region_size == 0 && i != 0) {
            angle_regional_mean[j] /= n;
            std::cout << "region #" << j << " average angle : " << angle_regional_mean[j] << "\n";
            n = 0;
            angle_regional_mean[++j] = 0;
        }
    }

    for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
#ifdef DEBUG
        std::vector<Point>dc;
        rotated_rect_to_contour(white_keys_info.keys_rectangle_list[i] , dc);
        drawContours(image_copy , std::vector<std::vector<Point>>({dc}) , -1 , Scalar(0x00 , 0xff , 0xff) , 1);
#endif
        if(white_keys_info.keys_rectangle_list[i].angle >= right_whisker||white_keys_info.keys_rectangle_list[i].angle <= left_whisker) {
            // rotate the rectangle by its tip point
            Point2f pts[4];
            RotatedRect rr = white_keys_info.keys_rectangle_list[i];
            
            rr.points(pts);
            double target_angle = angle_regional_mean[(int)(i/region_size)];
            double delta_theta = target_angle-rr.angle;

            std::cout << "target_angle : " << target_angle << "\n";
            std::cout << "angle        : " << white_keys_info.keys_rectangle_list[i].angle << "\n";
            std::cout << "delta_theta : " << delta_theta << "\n";
            std::cout << "width  : " << rr.size.width << "\n";
            std::cout << "height : " << rr.size.height << "\n";
            int x0 = rr.center.x+(rr.size.height/2)*sin(rr.angle*M_PI/180.0f);
            int y0 = rr.center.y-(rr.size.height/2)*cos(rr.angle*M_PI/180.0f);
            int x = white_keys_info.keys_rectangle_list[i].center.x , y = white_keys_info.keys_rectangle_list[i].center.y;

            int rotated_x = (x-x0)*cos(delta_theta*M_PI/180.0f)-(y-y0)*sin(delta_theta*M_PI/180.0f);
            int rotated_y = (x-x0)*sin(delta_theta*M_PI/180.0f)+(y-y0)*cos(delta_theta*M_PI/180.0f);
            rotated_x += x0; rotated_y += y0;
            std::cout << "(x0 , y0) : (" << x0 << " , " << y0 << ")\n";
            circle(image_copy , Point(x0 , y0) , 1 , Scalar(0x00 , 0x00 , 0xff) , 2);

            circle(image_copy , Point(rr.center.x , rr.center.y) , 1 , Scalar(0xff , 0x00 , 0xff) , 2);
            circle(image_copy , Point(rotated_x , rotated_y) , 1 , Scalar(0xff , 0x00 , 0x00) , 2);

            white_keys_info.keys_rectangle_list[i].center.x = rotated_x;
            white_keys_info.keys_rectangle_list[i].center.y = rotated_y;
            
            white_keys_info.keys_rectangle_list[i].angle = target_angle;

#ifdef DEBUG
            std::vector<Point>dc;
            rotated_rect_to_contour(white_keys_info.keys_rectangle_list[i] , dc);
            drawContours(image_copy , std::vector<std::vector<Point>>({dc}) , -1 , Scalar(0xff , 0xff , 0x00) , 1);
#endif
        }
    }
    imshow("adjust_white_outliers" , image_copy);
    
}

void PianoRecognition::adjust_black_outliers(struct piano_keys_info &black_keys_info) {
    std::cout << "removing black outliers...\n";
    Mat image = Mat::zeros(black_keys_info.piano_image.size() , CV_8UC3);

    // remove the outliers by calculating distribution of y coord
    double dist_mean = std::accumulate(black_keys_info.cm_dist_from_bestfit_list.begin() , black_keys_info.cm_dist_from_bestfit_list.end() , 0);
    dist_mean /= black_keys_info.cm_dist_from_bestfit_list.size();
    double dist_sd = calculate_standard_deviation(black_keys_info.cm_dist_from_bestfit_list , dist_mean);

    double left_whisker = dist_mean-(3.0f*dist_sd);
    double right_whisker = dist_mean+(3.0f*dist_sd);
    for(int i = 0; i < black_keys_info.keys_rectangle_list.size(); i++) {
        if(black_keys_info.cm_dist_from_bestfit_list[i] >= right_whisker||black_keys_info.cm_dist_from_bestfit_list[i] <= left_whisker) {
            std::cout << "outlier found! index : " << i << "\n";
            remove_item_from_piano_info(black_keys_info , i);
            i--;
        }
#ifdef DEBUG
        std::vector<Point>c;
        rotated_rect_to_contour(black_keys_info.keys_rectangle_list[i] , c);
        drawContours(image , std::vector<std::vector<Point>>({c}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);
#endif
    }
#ifdef DEBUG
    std::cout << "black_keys_standard_deviation = " << black_keys_info.std_dev_key_y << "\n";
    // line(image , Point(0 , left_whisker) , Point(image.size().width , left_whisker) , Scalar(0x00 , 0x00 , 0xff) , 2);
    // line(image , Point(0 , black_keys_info.mean_key_y) , Point(image.size().width , black_keys_info.mean_key_y) , Scalar(0x00 , 0xff , 0xff) , 2);
    // line(image , Point(0 , right_whisker) , Point(image.size().width , right_whisker) , Scalar(0x00 , 0x00 , 0xff) , 2);
    imshow("outliers_bound" , image);
#endif
    // adjust the width of big ones
    for(int i = 0; i < black_keys_info.keys_rectangle_list.size(); i++) {
        if(black_keys_info.keys_rectangle_list[i].size.width >= 1.5*black_keys_info.median_key_width) {
            int new_width = black_keys_info.median_key_width;
            adjust_rotated_rect_width(black_keys_info.keys_rectangle_list[i] , new_width , false);
        }
    }
}

void PianoRecognition::detect_missing_white_keys(struct piano_keys_info &keys_info) {
#ifdef DEBUG
    Mat new_constructed_img;
#endif
    bool new_added = false;
    cvtColor(keys_info.piano_image , new_constructed_img , COLOR_GRAY2BGR);
    // remove the outlier by comparing distance from the best-fit line
    std::cout << "median distance : " << keys_info.median_dist_between_keys << "\n";
    std::vector<std::pair<RotatedRect , int>> *vect = new std::vector<std::pair<RotatedRect , int>>;
    for(int i = 0; i < keys_info.dist_between_keys_list.size(); i++) {
        // compare with the nearby keys
        vect->push_back(std::pair<RotatedRect , int>(keys_info.keys_rectangle_list[i] , i));
        if(keys_info.dist_between_keys_list[i] >= keys_info.keys_rectangle_list[i].size.width*1.5) {
            std::cout << "missing keys found!, index = " << i << "\n";
            keys_info.separated_keys_rectangle_list.push_back(vect);

            vect = new std::vector<std::pair<RotatedRect , int>>;
        }
    }
    keys_info.separated_keys_rectangle_list.push_back(vect);
    for(std::vector<std::pair<RotatedRect , int>>*vect : keys_info.separated_keys_rectangle_list) {
        std::cout << "----- consecutive part -----\n";
        for(std::pair<RotatedRect , int>p : *vect) {
            std::cout << p.second << " , ";
        }
        std::cout << "\n";
    }

    if(!new_added) return;
    keys_info.dist_between_keys_list.clear();
    keys_info.cm_dist_from_bestfit_list.clear();
    write_keys_info(keys_info);
}

void PianoRecognition::detect_missing_black_keys(struct piano_keys_info &black_keys_info , struct piano_keys_info &white_keys_info) {
    if(black_keys_info.dist_between_keys_list.size() == 0) {
        std::cout << "dist_between_keys_list empty!\n";
        return;
    }
    int missing_black_keys = 36-black_keys_info.keys_rectangle_list.size();
    int black_key_distance_list[] = {
        2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,
    };
    int black_key_distance_cumulative[35] = {0 , };
    for(int i = 1; i < 35; i++) {
        black_key_distance_cumulative[i] = black_key_distance_cumulative[i-1]+black_key_distance_list[i];
        std::cout << "c : " << black_key_distance_cumulative[i] << "\n";
    }
    
    Mat image = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC3);

    RNG rng((unsigned int)time(0));
    std::vector<int>key_count_list;
    for(int i = 0; i < black_keys_info.keys_rectangle_list.size()-1; i++) {
        // compare with the nearby keys
        std::cout << "-- " << i << "--\n";
        std::cout << "distance  : " << black_keys_info.dist_between_keys_list[i] << "\n";
        
        double width = white_keys_info.median_key_width;//*cos(black_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
        
        std::vector<Point2f>pts;
        black_keys_info.keys_rectangle_list[i].points(pts);
        int current_x = (pts[2].x+pts[3].x)/2;
        int current_y = (pts[2].y+pts[3].y)/2;
        for(int j = 0; j < white_keys_info.keys_rectangle_list.size(); j++) {
            std::vector<Point>contour;
            rotated_rect_to_contour(white_keys_info.keys_rectangle_list[j] , contour);
            if(pointPolygonTest(contour , Point2f(current_x , current_y) , false) > 0) {
                width = white_keys_info.keys_rectangle_list[j].size.width;//*cos(black_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
                break;
            }
        }
        Scalar color = Scalar(rng.uniform(0 , 180) , rng.uniform(0 , 180) , rng.uniform(0 , 180));
        
        std::cout << "width   : " << width << "\n";
        int key_count = round((double)black_keys_info.dist_between_keys_list[i]/width);
        std::cout << "key_count : " << key_count << "\n";
        bool match = (key_count == black_key_distance_list[i]);
        std::cout << "match     : " << match << "\n";
        key_count_list.push_back(key_count);
    }

    for(int i = 0; i < key_count_list.size(); i++) {
        if(key_count_list[i] == black_key_distance_list[i]) continue;
        
        std::cout << "black key #" << i << "\n";
        
    }
    imshow("blackpoint" , image);
}

/// @brief Adjust the height of the rotated rectangle
/// @param rect The target rotated rectangle
/// @param new_height new height
/// @param direction true = Up (default), false = Down
static void adjust_rotated_rect_height(RotatedRect &rect , int new_height , bool direction=true) {
    /* transform the location & height */
    // 1. Elongate the rectangle's size
    int d = (new_height-rect.size.height)/2;

    // 2. To match the location of the rectangle prior to the elongation, adjust the 
    //    location of the rectangle
    rect.center.x += (direction ? 1.0f : -1.0f)*d*sin(rect.angle*M_PI/180.0f);
    rect.center.y -= (direction ? 1.0f : -1.0f)*d*cos(rect.angle*M_PI/180.0f);
    
    rect.size.height = new_height;
}

/// @brief Create the rectangular contour area of the white keys from the image of piano.
/// @param piano_image The processed image of the piano. The image should only contain the piano and should be gray-scaled. 
/// @param keys_info 
void PianoRecognition::detect_white_keys(Mat piano_image , struct piano_keys_info &keys_info , const struct piano_keys_info &black_keys_info) {
    Mat piano_image_padding , truncated , canny , adaptive , line_enlarged;
    int rectangles_count = 0;

    int padding = 30;

#ifdef DEBUG
    RNG rng((unsigned int)time(0));
#endif

    piano_image.copyTo(keys_info.piano_image);
    int img_width = piano_image.size().width;
    int img_height = piano_image.size().height;
    std::cout << "(detect_white_keys) 1. truncating the black region... \n";
    if(black_keys_info.keys_rectangle_list.size() == 0) {
        std::cout << "No black keys found!!\n";
        return;
    }
    
    // check whether the image should be flipped
    bool flipped = false;
    int black_average_center_y = 0;
    int black_min_y = 0x7fffffff;
    int black_max_y = 0;
    for(RotatedRect rr : black_keys_info.keys_rectangle_list) {
        Point2f pts[4];
        rr.points(pts);
        black_average_center_y += rr.center.y; 
        std::vector<int>y_vect = {(int)pts[0].y , (int)pts[1].y , (int)pts[2].y , (int)pts[3].y};
        black_min_y = std::min(black_min_y , *std::min_element(y_vect.begin() , y_vect.end()));
        black_max_y = std::max(black_max_y , *std::max_element(y_vect.begin() , y_vect.end()));
    }
    black_average_center_y /= black_keys_info.keys_rectangle_list.size();
    if(black_average_center_y >= piano_image.size().height/2) flipped = true;
    std::cout << "flipped = " << flipped << "\n";
    std::cout << "black_min_y = " << black_min_y << "\n";
    std::cout << "black_max_y = " << black_max_y << "\n";

    copyMakeBorder(piano_image , piano_image_padding , padding , padding , padding , padding , BORDER_CONSTANT , Scalar(0));

    int truncated_height = 0;
    std::vector<Point>truncating_contour;
    // add the corner points to the contour

    // sort the contours by x coordinates

    
    truncating_contour.insert(truncating_contour.begin() , Point(0 , flipped ? black_min_y : black_max_y));
    truncating_contour.insert(truncating_contour.begin()+1 , Point(0 , flipped ? piano_image_padding.size().height : 0));
    for(RotatedRect rr : black_keys_info.keys_rectangle_list) {
        std::vector<Point>contour;
        rotated_rect_to_contour(rr , contour);
        for(int i = 0; i < contour.size(); i++) {
            contour[i].x += padding;
            contour[i].y += padding;
        }
        if(flipped) { truncating_contour.push_back(contour[1]); truncating_contour.push_back(contour[2]); }
        else { truncating_contour.push_back(contour[0]); truncating_contour.push_back(contour[3]); }
    }
    std::sort(truncating_contour.begin() , truncating_contour.end() , [](const auto &a , const auto &b) { return (a.x < b.x); });
    
    truncating_contour.push_back(Point(piano_image_padding.size().width , flipped ? black_min_y : black_max_y));
    truncating_contour.push_back(Point(piano_image_padding.size().width , flipped ? piano_image_padding.size().height : 0));
    Mat testimg;
    morphologyEx(piano_image , testimg , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(3 , 3)) , Point(-1 , -1) , 3);

    piano_image_padding.copyTo(truncated);
    // draw the contour that covers all the black keys
    drawContours(truncated , std::vector<std::vector<Point>>({truncating_contour}) , -1 , 0xff , -1);

#ifdef DEBUG
    Mat debug1;
    cvtColor(truncated , debug1 , COLOR_GRAY2BGR);
    copyMakeBorder(debug1 , debug1 , 20 , 20 , 20 , 20 , BORDER_CONSTANT , Scalar(0x00 , 0x00 , 0x00));
    for(int i = 0; i < truncating_contour.size(); i++) {
        circle(debug1 , Point(truncating_contour[i].x+20 , truncating_contour[i].y+20) , 1 , Scalar(0x00 , 0xff , 0x00) , -1);
        putText(debug1 , std::to_string(i) , Point(truncating_contour[i].x+20 , truncating_contour[i].y+20) , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0xff , 0x00) , 2);
    }
    imshow("debug1" , debug1);
#endif
    // adaptive thresholding & erode
    adaptiveThreshold(truncated , adaptive , 255 , ADAPTIVE_THRESH_GAUSSIAN_C , THRESH_BINARY , 11 , 5);
    drawContours(adaptive , std::vector<std::vector<Point>>({truncating_contour}) , -1 , 0x00 , 1);

    int erode_iteration = 2;
    erode(adaptive , line_enlarged , Mat() , Point(-1 , -1) , erode_iteration);
    imshow("adaptive_white" , adaptive);

    std::vector<std::vector<Point>>keys_contours_1;
    std::vector<RotatedRect>keys_rect_list_1;

    findContours(line_enlarged , keys_contours_1 , RETR_EXTERNAL , CHAIN_APPROX_SIMPLE);
#ifdef DEBUG
    Mat copy_piano_img;
    cvtColor(piano_image_padding , copy_piano_img , COLOR_GRAY2BGR);
#endif
    for(int i = 0; i < keys_contours_1.size(); i++) {
        // if the arc length of the contour is bigger than the sum of image's width and height
        if(arcLength(keys_contours_1[i] , false) >= img_width+img_height) continue;

#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 180) , rng.uniform(0 , 180) , rng.uniform(0 , 180));
        RotatedRect dr = minAreaRect(keys_contours_1[i]);
        std::vector<Point>dc;
        rotated_rect_to_contour(dr , dc);
        drawContours(copy_piano_img , std::vector<std::vector<Point>>({dc}) , -1 , color , 1);
#endif
        // check the bounding rect to see if width is longer than height
        // If width is longer than the height, it signifies that it is not a key. 
        Rect bounding_box = boundingRect(keys_contours_1[i]);
        if(bounding_box.width > bounding_box.height) continue;
        /* Get the bounding rectangles of the white keyboard */
        RotatedRect r = minAreaRect(keys_contours_1[i]);
        // make the shorter side to width and long sides to height
        if(r.size.width > r.size.height) {
            std::swap(r.size.width , r.size.height);
            r.angle -= 90.0f;
        }
        // push to the final list
        keys_rect_list_1.push_back(r);
    }

#ifdef DEBUG
    Mat copy_piano_img2;
    cvtColor(piano_image , copy_piano_img2 , COLOR_GRAY2BGR);
    imshow("white_piano_image" , piano_image);
    imshow("white_copy_piano_img" , copy_piano_img);
#endif
    std::vector<double>median_width_list , median_height_list;
    keys_info.mean_key_width = 0;
    keys_info.mean_key_height = 0;
    for(int i = 0; i < keys_rect_list_1.size(); i++) {
        // find the closest black keys from the key and set the key's height to align with black's height

        // index of the black key closest to the key
        int black_min_dist_from_key = 0;
        double min_dist = 0x7fffffff;
        for(int j = 0; j < black_keys_info.keys_rectangle_list.size(); j++) {
            double distance = euclidean_distance(keys_rect_list_1[i].center , black_keys_info.keys_rectangle_list[j].center);
            if(min_dist > distance) {
                min_dist = distance;
                black_min_dist_from_key = j;
            }
        }
        // found the closest black key
        RotatedRect closest_black = black_keys_info.keys_rectangle_list[black_min_dist_from_key];
        Point2f black_points[4] , white_points[4];
        closest_black.points(black_points);
        keys_rect_list_1[i].points(white_points);
        
        // adjust the height of the key
        Point2f black_p = (black_points[1].y < black_points[2].y) ? black_points[1] : black_points[2];
        Point2f white_p = (white_points[0].y > white_points[3].y) ? white_points[0] : white_points[3];
        if(flipped) {
            black_p = (black_points[0].y > black_points[3].y) ? black_points[0] : black_points[3];
            white_p = (white_points[1].y < white_points[2].y) ? white_points[1] : white_points[2];
        }
        white_p.x -= padding;
        white_p.y -= padding;
        double distance = abs(black_p.y-white_p.y);
        adjust_rotated_rect_height(keys_rect_list_1[i] , distance , !flipped);

        // inflate the width
        keys_rect_list_1[i].size.width += erode_iteration*2;

        // calculate mean and median
        keys_info.mean_key_width += keys_rect_list_1[i].size.width;
        keys_info.mean_key_height += keys_rect_list_1[i].size.height;
        median_width_list.push_back(keys_rect_list_1[i].size.width);
        median_height_list.push_back(keys_rect_list_1[i].size.height);

        keys_rect_list_1[i].center.x -= padding;
        keys_rect_list_1[i].center.y -= padding;
        keys_info.keys_rectangle_list.push_back(keys_rect_list_1[i]);
        
#ifdef DEBUG
        std::vector<Point>c1,c2;
        rotated_rect_to_contour(closest_black , c1);
        rotated_rect_to_contour(keys_rect_list_1[i] , c2);
        Scalar color = Scalar(rng.uniform(0 , 180) , rng.uniform(0 , 180) , rng.uniform(0 , 180));
        drawContours(copy_piano_img2 , std::vector<std::vector<Point>>({c1 , c2}) , -1 , color , 1);
#endif
    }
    keys_info.median_key_width = calculate_median(median_width_list);
    keys_info.median_key_height = calculate_median(median_height_list);
    imshow("copy_piano_img2" , copy_piano_img2);
}

void PianoRecognition::detect_black_keys(Mat piano_image , struct piano_keys_info &keys_info) {
    Mat only_black_keys , black_keys_enclosed , piano_image_padding;

    int rectangles_count = 0;

    int img_width = piano_image.size().width;
    int img_height = piano_image.size().height;
    int padding = 30;

    piano_image.copyTo(keys_info.piano_image);
    copyMakeBorder(piano_image , piano_image_padding , padding , padding , padding , padding , BORDER_CONSTANT , Scalar(0));

    // black key mask
    std::cout << "(detect_black_keys) 1. Performing close operation & thresholding... \n";
    threshold(piano_image_padding , only_black_keys , -1 , 255 , THRESH_BINARY|THRESH_OTSU);
    // inRange(piano_image_padding , Scalar(0x00 , 0x00 , 0x00) , Scalar(0x55 , 0x55 , 0x55) , only_black_keys);
    // only_black_keys = ~only_black_keys;
    // morphologyEx(only_black_keys , only_black_keys , MORPH_CLOSE , getStructuringElement(MORPH_RECT , Size(3 , 3)));
    dilate(only_black_keys , only_black_keys , cv::Mat() , Point(-1 , -1) , 1);

    imshow("only_thresholded" , only_black_keys);

    std::cout << "(detect_black_keys) 2. Performing convex hull... \n";
    std::vector<std::vector<Point>>contours;
    only_black_keys.copyTo(black_keys_enclosed);
    findContours(only_black_keys , contours , RETR_EXTERNAL , CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<Point>>hull(contours.size());
    for(int i = 0; i < contours.size(); i++) {
        convexHull(contours[i] , hull[i]);
        drawContours(black_keys_enclosed , hull , i , 0xff , 10);
    }
    morphologyEx(black_keys_enclosed , black_keys_enclosed , MORPH_CLOSE , getStructuringElement(MORPH_RECT , Size(3 , 3)));

    std::vector<std::vector<Point>>black_keys_contours_1 , black_keys_contours;
    findContours(black_keys_enclosed , black_keys_contours_1 , RETR_LIST , CHAIN_APPROX_SIMPLE);

    RNG rng((unsigned int)time(0));

    std::cout << "(detect_black_keys) 3. Removing unnecessary rectangles... \n";
    Mat colorful;
    std::vector<RotatedRect>keys_rect_list_1;
    cvtColor(piano_image_padding , colorful , COLOR_GRAY2BGR);

    std::vector<double>median_width_list , median_height_list;
    // prepare for the key info
    keys_info.mean_key_width = 0;
    keys_info.mean_key_height = 0;

    for(int i = 0; i < black_keys_contours_1.size(); i++) {
        /* filter out contours that are too small or too big */
        // if the arc length of the contour is smaller than the 1% of the image's width
        if(arcLength(black_keys_contours_1[i] , false) < 0.01*img_width) { std::cout << "removed " << i << "\n"; continue; }
        // if the arc length of the contour is bigger than the sum of image's width and height
        if(arcLength(black_keys_contours_1[i] , false) >= img_width+img_height) continue;

        RotatedRect r = minAreaRect(black_keys_contours_1[i]);
        // make the shorter side to width and long sides to height
        if(r.size.width > r.size.height) {
            std::swap(r.size.width , r.size.height);
            r.angle -= 90.0f;
        }
        // push to the list
        keys_rect_list_1.push_back(r);

        /* calculate the mean area, width and height */
        /* Also calculate the maximum height of the rectangles */

        keys_info.mean_key_width += r.size.width;
        keys_info.mean_key_height += r.size.height;
        rectangles_count++;
#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        std::vector<Point>cr;
        get_bounding_rect_contour(black_keys_contours_1[i] , cr);
        // drawContours(colorful , std::vector<std::vector<Point>>{cr} , -1 , color , 2);
#endif
        median_width_list.push_back(r.size.width);
        median_height_list.push_back(r.size.height);
    }
    keys_info.mean_key_width /= rectangles_count;
    keys_info.mean_key_height /= rectangles_count;

    keys_info.median_key_width = calculate_median(median_width_list);
    keys_info.median_key_height = calculate_median(median_height_list);
#ifdef DEBUG
    std::cout << "keys_info.mean_key_width : " << keys_info.mean_key_width << "\n";
    std::cout << "keys_info.mean_key_height : " << keys_info.mean_key_height << "\n";
#endif

    for(int i = 0; i < keys_rect_list_1.size(); i++) {
        if(keys_rect_list_1[i].size.width < keys_info.mean_key_width*0.55) continue;
        if(std::min(keys_rect_list_1[i].size.width , keys_rect_list_1[i].size.height) > keys_info.mean_key_width*1.6) continue;
        if(keys_rect_list_1[i].size.height < keys_info.mean_key_height) {
            adjust_rotated_rect_height(keys_rect_list_1[i] , keys_info.mean_key_height);
        }
#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        std::vector<Point>cr;
        rotated_rect_to_contour(keys_rect_list_1[i] , cr);
        drawContours(colorful , std::vector<std::vector<Point>>{cr} , -1 , color , -1);
#endif
        keys_rect_list_1[i].size.width += 2;
        keys_rect_list_1[i].center.x -= padding;
        keys_rect_list_1[i].center.y -= padding;
        // push the rectangle to the final list
        keys_info.keys_rectangle_list.push_back(keys_rect_list_1[i]);
    }
    // imshow("piano_image" , piano_image);
    // imshow("only_black_keys" , only_black_keys);
    imshow("black_keys_enclosed" , black_keys_enclosed);
    imshow("colorful" , colorful);
}

/// @brief Detect the shapes of the white keys
/// @param white_keys_info 
/// @param black_keys_info 
void PianoRecognition::detect_white_key_shapes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {
    Mat white_mat(white_keys_info.piano_image.size() , CV_8UC1);
    Mat black_mat(white_keys_info.piano_image.size() , CV_8UC1);

#define KEY_NO_BLACK    1
#define KEY_BLACK_LEFT  2
#define KEY_BLACK_RIGHT 3
#define KEY_BLACK_BOTH  4

    int white_key_index = 0;
    std::vector<int>error_index_list;
    for(RotatedRect white_rect : white_keys_info.keys_rectangle_list) {
        bool left = false;
        bool right = false;

        std::vector<std::vector<Point>>overlapping_contours;
        Point2f white_rect_pts[4];
        white_rect.points(white_rect_pts);
        std::vector<Point>left_side_contour = {white_rect_pts[0] , white_rect_pts[0] , white_rect_pts[1] , white_rect_pts[1]};
        std::vector<Point>right_side_contour = {white_rect_pts[2] , white_rect_pts[2] , white_rect_pts[3] , white_rect_pts[3]};
        for(RotatedRect black_rect : black_keys_info.keys_rectangle_list) {
            std::vector<Point>intersect;
            if(rotatedRectangleIntersection(white_rect , black_rect , intersect) == INTERSECT_NONE) continue;
            
            Mat overlap_check_1 = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC1);
            Mat overlap_check_2 = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC1);
            Mat overlap_check_3 = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC1);
            std::vector<Point>black_rect_contour;
            rotated_rect_to_contour(black_rect , black_rect_contour);
            drawContours(overlap_check_1 , std::vector<std::vector<Point>>({black_rect_contour}) , -1 , 0xff , -1);
            drawContours(overlap_check_2 , std::vector<std::vector<Point>>({left_side_contour}) , -1 , 0xff , -1);
            drawContours(overlap_check_3 , std::vector<std::vector<Point>>({right_side_contour}) , -1 , 0xff , -1);

            int overlap_left_check = countNonZero(overlap_check_1 & overlap_check_2);
            int overlap_right_check = countNonZero(overlap_check_1 & overlap_check_3);
            if(overlap_left_check >= 1 && overlap_right_check >= 1) {
                std::cout << "error!" << "\n";
                if(white_rect.center.x > black_rect.center.x) {
                    left = true;
                }
                else {
                    right = true;
                }
                error_index_list.push_back(white_key_index);
            }
            else {
                if(overlap_left_check >= 1) { left = true; }
                if(overlap_right_check >= 1) { right = true; }
            }
        }

        Scalar color;
        int key_type;
        switch((left << 1)|right) {
            case 0:    key_type = KEY_NO_BLACK;    break;
            case 0b01: key_type = KEY_BLACK_RIGHT; break; // right
            case 0b11: key_type = KEY_BLACK_BOTH;  break; // both
            case 0b10: key_type = KEY_BLACK_LEFT;  break; // left
        }
        white_keys_info.white_key_shapes.push_back(key_type);
        white_key_index++;
    }
    std::cout << "----- white_keys_info.white_key_shapes -----\n";
    for(int i = 0; i < white_keys_info.white_key_shapes.size(); i++) {
        std::cout << white_keys_info.white_key_shapes[i] << " , ";
    }
    std::cout << "\n";
}

const char *number_to_note_string(int note_number) {
    switch(NOTE(note_number)) {
        case PIANO_KEY_C:      return "C";
        case PIANO_KEY_Csharp: return "C#";
        case PIANO_KEY_D:      return "D";
        case PIANO_KEY_Dsharp: return "D#";
        case PIANO_KEY_E:      return "E";
        case PIANO_KEY_F:      return "F";
        case PIANO_KEY_Fsharp: return "F#";
        case PIANO_KEY_G:      return "G";
        case PIANO_KEY_Gsharp: return "G#";
        case PIANO_KEY_A:      return "A";
        case PIANO_KEY_Asharp: return "A#";
        case PIANO_KEY_B:      return "B";
    }
    return "?";
}

void PianoRecognition::detect_white_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {
    double mean_y = 0;
    /* C : 1
     * D : 3
     * E : 5
     * F : 6
     * G : 8
     * A : 10
     * B : 12
     */
    /*                                   0    1              2              3              4              5              6              7              8*/
    int key_shape_list[]              = {3,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 1};
    int key_shape_list_notes_octave[] = {0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 5,5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7, 8};
    int key_shape_list_notes[] = {10,12, 1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    for(RotatedRect rr : white_keys_info.keys_rectangle_list) { mean_y += rr.center.y; }
    mean_y /= (double)white_keys_info.keys_rectangle_list.size();

    std::cout << "mean_y : " << mean_y << "\n";
    double current_x = 0 , current_y = mean_y;
    current_x = white_keys_info.keys_rectangle_list[0].center.x;
    std::cout << "------ detect_white_key_notes ------\n";
    double previous_width = white_keys_info.keys_rectangle_list[0].size.width*cos(white_keys_info.keys_rectangle_list[0].angle*M_PI/180.0f);
    int previous_hit_index = 0;
    double distance = 0;
    int current_offset = 0;
    while(1) {
        bool detected = false;
        int detected_index = 0;
        for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
            std::vector<Point>contour;
            rotated_rect_to_contour(white_keys_info.keys_rectangle_list[i] , contour);
            if(pointPolygonTest(contour , Point2f(current_x , current_y) , false) >= 0) {
                std::cout << "hit , index = " << i << "\n";
                current_x = white_keys_info.keys_rectangle_list[i].center.x;
                current_y = white_keys_info.keys_rectangle_list[i].center.y;
                previous_width = white_keys_info.keys_rectangle_list[i].size.width*cos(white_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
                
                detected = true;
                detected_index = i;
                break;
            }
        }
        // find the closest thing from the current offset
        if(detected == true) {
            int min_distance = 0x7fffffff;
            int min_dist_piano_index = -1;
            for(int i = 0; i < 52; i++) {
                if(key_shape_list[i] == white_keys_info.white_key_shapes[detected_index]) {
                    std::cout << "match : " << i << ", note : " << number_to_note_string(key_shape_list_notes[i]) << "\n";
                    std::cout << "distance between the offset : " << abs(current_offset-i) << "\n";
                    if(min_distance > abs(current_offset-i)) {
                        min_dist_piano_index = i;
                        min_distance = abs(current_offset-i);
                    }
                }
            }
            std::cout << "correlating index : " << min_dist_piano_index << " , note : " << number_to_note_string(key_shape_list_notes[min_dist_piano_index]) << "\n";
            current_offset = min_dist_piano_index;
            white_keys_info.key_notes.push_back(std::pair<int , int>((key_shape_list_notes[min_dist_piano_index])|(key_shape_list_notes_octave[min_dist_piano_index] << 8) , min_dist_piano_index));
        }
        if(detected_index == white_keys_info.keys_rectangle_list.size()-1 || current_x >= white_keys_info.piano_image.size().width) { break; }
        std::cout << "x : " << current_x << "\n";
        current_x += previous_width;
        current_offset++;
    }
    return;
}

void PianoRecognition::detect_black_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {
    for(int b = 0; b < black_keys_info.keys_rectangle_list.size(); b++) {
        // first : index , second : note
        std::vector<std::pair<int , int>>intersecting_white_keys;
        for(int w = 0; w < white_keys_info.keys_rectangle_list.size(); w++) {
            std::vector<Point>intersect;
            int intersection_type = rotatedRectangleIntersection(black_keys_info.keys_rectangle_list[b] , white_keys_info.keys_rectangle_list[w] , intersect);
            if(intersection_type == INTERSECT_PARTIAL) {
                intersecting_white_keys.push_back(std::pair<int , int>(w , white_keys_info.key_notes[w].first));
            }
        }
        int black_note = 0;
        std::sort(intersecting_white_keys.begin() , intersecting_white_keys.end() , [](const auto &a , const auto &b) { return a.second < b.second; });
        if(intersecting_white_keys.size() == 1) {
            std::cout << "  size = 1" << "\n";
            switch((NOTE(intersecting_white_keys[0].second))) {
                case PIANO_KEY_F: black_note = intersecting_white_keys[0].second-1; break;
                case PIANO_KEY_B: black_note = intersecting_white_keys[0].second+1; break;
            }
        }
        if(intersecting_white_keys.size() == 2) {
            std::cout << "  size = 2" << "\n";
            if(NOTE(intersecting_white_keys[0].second+1) == NOTE(intersecting_white_keys[1].second-1)) {
                black_note = intersecting_white_keys[0].second+1;
            }
        }
        std::cout << "black index : " << b << " , note : " << number_to_note_string(black_note) << "(" << black_note << ")\n";
        black_keys_info.key_notes.push_back(std::pair<int , int>(black_note , b));
    }
}

void PianoRecognition::fill_missing_white_keys(struct piano_keys_info &white_keys_info) {
    int total_missing_count = 52-white_keys_info.keys_rectangle_list.size();
    int filled_count = 0;
    if(total_missing_count == 0) return;
    // first  : missing_key_index
    // second : missing_key_piano_index (0-52)
    // third  : missing_key_count
    std::vector<std::tuple<int , int , int>>missing_keys;
    int index_dist_between_keys = 0;
    for(int i = 1; i < white_keys_info.key_notes.size(); i++) {
        index_dist_between_keys = white_keys_info.key_notes[i].second-white_keys_info.key_notes[i-1].second;
        if(index_dist_between_keys != 1) {
            std::cout << "missing keys found! index : " << i-1 << " , count : " << index_dist_between_keys-1 << "\n";
            missing_keys.push_back(std::tuple<int , int , int>(i-1 , white_keys_info.key_notes[i-1].second , index_dist_between_keys-1));
        }
    }

    // first : index , second : number of missing keys
    if(missing_keys.size() == 0) return;
    std::cout << "missing keys : " << missing_keys.size() << "\n";
    // if(white_keys_info.keys_rectangle_list.size() == 52) return;
    int key_shape_list_notes[] = {10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    int key_shape_list_notes_octave[] = {0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 5,5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7, 8};

    Mat test_img = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC3);
    std::vector<int>newly_created_index_list;
    for(std::tuple<int , int , int>missing_key : missing_keys) {
        bool exit = false;
        int missing_key_index       = std::get<0>(missing_key);
        int missing_key_piano_index = std::get<1>(missing_key);
        int missing_key_count       = std::get<2>(missing_key);

        Point p1 = white_keys_info.keys_rectangle_list[missing_key_index].center;
        Point p2 = white_keys_info.keys_rectangle_list[missing_key_index+1].center;
        double angle1 = white_keys_info.keys_rectangle_list[missing_key_index].angle;
        double angle2 = white_keys_info.keys_rectangle_list[missing_key_index+1].angle;

        filled_count += missing_key_count;
        if(filled_count > total_missing_count) {
            missing_key_count = (filled_count-total_missing_count);
            break;
        }
        // Internal division of the two points
        for(int j = 1; j < missing_key_count+1; j++) {
            int m = j , n = (missing_key_count+1)-j;
            // divided point
            Point p((m*p2.x+n*p1.x)/(m+n) , (m*p2.y+n*p1.y)/(m+n));
            circle(test_img , p , 2 , Scalar(0x00 , 0x00 , 0xff) , 1);
            // Construct a new rectangle. New rectangle has the same width and height as the neighboring rectangle
            // The angle of the new rectangle is the average of the two reference rectangle(ith and i+1th rectangles.)
            RotatedRect new_rect(p , Size(white_keys_info.keys_rectangle_list[missing_key_index].size.width , white_keys_info.keys_rectangle_list[missing_key_index].size.height) , (angle1+angle2)/2);
            white_keys_info.keys_rectangle_list.push_back(new_rect); // new ones are added

            newly_created_index_list.push_back(++missing_key_piano_index);
#ifdef DEBUG
            // draw the rectangles (why not?)
            std::vector<Point>rect_contour;
            rotated_rect_to_contour(new_rect , rect_contour);
            drawContours(test_img , std::vector<std::vector<Point>>({rect_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
#endif
        }
        if(exit) break;
    }
    std::sort(newly_created_index_list.begin() , newly_created_index_list.end());
    for(int i : newly_created_index_list) {
        white_keys_info.key_notes.insert(white_keys_info.key_notes.begin()+i , std::pair<int , int>(key_shape_list_notes[i] , i));
    }
    white_keys_info.dist_between_keys_list.clear();
    white_keys_info.cm_dist_from_bestfit_list.clear();
    write_keys_info(white_keys_info);

    imshow("newly_added" , test_img);
}

void PianoRecognition::doublecheck_white_keys(struct piano_keys_info &white_keys_info) {
    std::vector<int>discrepencies;
    int key_shape_list_notes[] = {10,12, 1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    for(int i = 0; i < white_keys_info.key_notes.size(); i++) {
        if(NOTE(white_keys_info.key_notes[i].first) != key_shape_list_notes[i]) discrepencies.push_back(i);
    }
    std::cout << "white discrepencies : " << discrepencies.size() << "\n";
    if(discrepencies.size() == 0) return;
}

void PianoRecognition::doublecheck_black_keys(struct piano_keys_info &black_keys_info) {
    int black_key_notes_template[] = {
        0x0A , 
        0x12 , 0x14 , 0x17 , 0x19 , 0x1B , 
        0x22 , 0x24 , 0x27 , 0x29 , 0x2B , 
        0x32 , 0x34 , 0x37 , 0x39 , 0x3B , 
        0x42 , 0x44 , 0x47 , 0x49 , 0x4B , 
        0x52 , 0x54 , 0x57 , 0x59 , 0x5B , 
        0x62 , 0x64 , 0x67 , 0x69 , 0x6B , 
        0x72 , 0x74 , 0x77 , 0x79 , 0x7B , 
    };
    std::vector<int>discrepencies;
    for(int i = 0; i < black_keys_info.key_notes.size(); i++) {
        if(black_keys_info.key_notes[i].first != black_key_notes_template[i]) discrepencies.push_back(i);
    }
    std::cout << "black discrepencies : " << discrepencies.size() << "\n";
    if(discrepencies.size() == 0) return;
    
}
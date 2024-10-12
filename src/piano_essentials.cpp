#include <piano_detection.hpp>

using namespace cv;
using namespace xfeatures2d;

void create_features_info(Mat img , std::vector<KeyPoint>&keypoints , Mat &descriptors) {
    Ptr<ORB>detector = ORB::create(img.size().width/2);
    detector->detectAndCompute(img , Mat() , keypoints , descriptors);
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

void relocate_rotated_rect_list(std::vector<cv::RotatedRect>&rect_list , int dx , int dy) {
    for(int i = 0; i < rect_list.size(); i++) {
        rect_list[i].center.x += dx;
        rect_list[i].center.y += dy;
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

double rotational_matrix_x(double x , double y , double theta , double x0 , double y0) {
    return ((x-x0)*cos(theta)-(y-y0)*sin(theta))+x0;
}

double rotational_matrix_y(double x , double y , double theta , double x0 , double y0) {
    return ((x-x0)*sin(theta)+(y-y0)*cos(theta))+y0;
}

Point2f rotational_matrix(Point2f p , double theta , Point2f pivot) {
    return Point2f(
        rotational_matrix_x(p.x , p.y , theta , pivot.x , pivot.y) , 
        rotational_matrix_y(p.x , p.y , theta , pivot.x , pivot.y)
    );
}

/// @brief Adjust the height of the rotated rectangle
/// @param rect The target rotated rectangle
/// @param new_height new height
/// @param direction true = Up (default), false = Down
void adjust_rotated_rect_height(RotatedRect &rect , int new_height , bool direction) {
    /* transform the location & height */
    // 1. Elongate the rectangle's size
    int d = (new_height-rect.size.height)/2;

    // 2. To match the location of the rectangle prior to the elongation, adjust the 
    //    location of the rectangle
    rect.center.x += (direction ? 1.0f : -1.0f)*d*sin(rect.angle*M_PI/180.0f);
    rect.center.y -= (direction ? 1.0f : -1.0f)*d*cos(rect.angle*M_PI/180.0f);
    
    rect.size.height = new_height;
}

/// @brief Adjust the width of the rotated rectangle
/// @param rect 
/// @param new_width 
/// @param direction false : left, true : right
void adjust_rotated_rect_width(RotatedRect &rect , int new_width , bool direction) {
    int d = (rect.size.width-new_width)/2;

    rect.center.x += (direction ? 1.0f : -1.0f)*d*cos(rect.angle*M_PI/180.0f);
    rect.center.y -= (direction ? 1.0f : -1.0f)*d*sin(rect.angle*M_PI/180.0f);

    rect.size.width = new_width;
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


/// @brief Write the keys info to the structure based on the informations provided
///        Necessary information consists: 
///         1. keys_info.keys_rectangle_list
/// @param keys_info 
void write_keys_info(piano_keys_info_t &keys_info) {
    std::vector<Point>center_points;
    for(RotatedRect r : keys_info.keys_rectangle_list) { center_points.push_back(r.center); }

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

    std::vector<double>dist_median_list , y_list;

    if(keys_info.keys_rectangle_list.size() <= 1) {
        std::cout << "No keys in the key_rectangle_list!!\n";
        return;
    }
    keys_info.cm_bestfit_b = bestfit_b;
    keys_info.cm_bestfit_a = bestfit_a;
    for(int i = 0; i < keys_info.keys_rectangle_list.size(); i++) {
        /* Calculate the distance between the keys(based on center of mass) */
        // skip the last one
        if(i >= keys_info.keys_rectangle_list.size()-1) { keys_info.dist_between_keys_list.push_back(-1); continue; }
        // distance between two centers
        double distance_between_keys = euclidean_distance(keys_info.keys_rectangle_list[i].center , keys_info.keys_rectangle_list[i+1].center);
        
        // y-coord
        y_list.push_back(keys_info.keys_rectangle_list[i].center.y);

        dist_median_list.push_back(distance_between_keys);
    }
}

void write_pivot_info(PianoInfo &piano_info , white_or_black_t white_or_black) {
    // if flipped, the pivot should be the higher point on the rectangle
    // if not flipped, the pivot should be the lower point on the rectangle
    piano_keys_info_t *keys_info = white_or_black ? (piano_keys_info_t *)&piano_info.white_keys_info : (piano_keys_info_t *)&piano_info.black_keys_info;
#ifdef DEBUG
    Mat pivot_image;
    cvtColor(keys_info.piano_image , pivot_image , COLOR_GRAY2BGR);
#endif
    for(int i = 0; i < keys_info->keys_rectangle_list.size(); i++) {
        Point2f pts[4];
        keys_info->keys_rectangle_list[i].points(pts);
        // 0,1 <--> 2,3
        // 1,2 <--> 3,0
        Point p1,p2;
        Point alternative_p1,alternative_p2;
        // shorter one is the width
        Point width_p1,width_p2; // pair 1
        Point width_p3,width_p4; // pair 2
        if(euclidean_distance(pts[0] , pts[1]) < euclidean_distance(pts[1] , pts[2])) {
            width_p1 = pts[0]; width_p2 = pts[1];
            width_p3 = pts[2]; width_p4 = pts[3];
        }
        else {
            width_p1 = pts[1]; width_p2 = pts[2];
            width_p3 = pts[3]; width_p4 = pts[0];
        }

        // compare which points are more farther 
        int avg_y_1 = (width_p1.y+width_p2.y)/2 , avg_y_2 = (width_p3.y+width_p4.y)/2;
        if(piano_info.flipped) {
            if(avg_y_1 < avg_y_2) { p1 = width_p1; p2 = width_p2; alternative_p1 = width_p3; alternative_p2 = width_p4; }
            else { p1 = width_p3; p2 = width_p4; alternative_p1 = width_p1; alternative_p2 = width_p2; }
        }
        else {
            if(avg_y_1 < avg_y_2) { p1 = width_p3; p2 = width_p4; alternative_p1 = width_p1; alternative_p2 = width_p2; }
            else { p1 = width_p1; p2 = width_p2; alternative_p1 = width_p3; alternative_p2 = width_p4; }
        }
        bool overlap = false;
        Point midpoint((p1.x+p2.x)/2 , (p1.y+p2.y)/2);
        Point midpoint_a((alternative_p1.x+alternative_p2.x)/2 , (alternative_p1.y+alternative_p2.y)/2);
        int width = keys_info->keys_rectangle_list[i].size.width;
        double dist = euclidean_distance(midpoint , midpoint_a);
        double m = (double)width/2.0f , n = dist-((double)width/2.0f);
        
        // use internal division to calculate the coordinate of the pivot
        Point pivot(((m*midpoint_a.x+n*midpoint.x)/(m+n)) , ((m*midpoint_a.y+n*midpoint.y)/(m+n)));

        for(int j = 0; j < keys_info->keys_rectangle_list.size(); j++) {
            if(j == i) continue;
            std::vector<Point>contour;
            RotatedRect rr = keys_info->keys_rectangle_list[j];
            rr.size.width *= 1.25; // slightly inflate the width of rectangle
            rotated_rect_to_contour(rr , contour);
            if(pointPolygonTest(contour , pivot , false) > 0) {
                overlap = true;
                break;
            }
        }

        if(overlap) {
            std::swap(p1 , alternative_p1);
            std::swap(p2 , alternative_p2);
        }
        // redeclare the midpoint
        midpoint = Point((p1.x+p2.x)/2 , (p1.y+p2.y)/2);
        midpoint_a = Point((alternative_p1.x+alternative_p2.x)/2 , (alternative_p1.y+alternative_p2.y)/2);
        // calculate the pivot from the two points
        
        // use internal division to calculate the coordinate of the pivot
        pivot = Point(((m*midpoint_a.x+n*midpoint.x)/(m+n)) , ((m*midpoint_a.y+n*midpoint.y)/(m+n)));
        
        keys_info->keys_rectangle_pivot.push_back(std::pair<Point , bool>(pivot , pivot.x > keys_info->keys_rectangle_list[i].center.x));
#ifdef DEBUG
        std::vector<Point>dc;
        rotated_rect_to_contour(keys_info.keys_rectangle_list[i] , dc);
        drawContours(pivot_image , std::vector<std::vector<Point>>({dc}) , -1 , Scalar(0x00 , 0x00 , 0xff) , 1);
        
        circle(pivot_image , p1 , 2 , Scalar(0x00 , 0xff , 0x00) , -1);
        circle(pivot_image , p2 , 2 , Scalar(0x00 , 0xff , 0x00) , -1);
        circle(pivot_image , midpoint , 2 , Scalar(0x00 , 0xff , 0xff) , -1);
        circle(pivot_image , pivot , 2 , Scalar(0x00 , 0x00 , 0xff) , -1);
        circle(pivot_image , midpoint_a , 2 , Scalar(0x00 , 0xff , 0xff) , -1);
#endif
    }
#ifdef DEBUG
    static int dn = 1;
    imshow("pivot_info"+std::to_string(dn++) , pivot_image);
#endif
}
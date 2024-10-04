#include <piano_key_detection.hpp>

using namespace cv;

/// @brief Create the rectangular contour area of the white keys from the image of piano.
/// @param piano_image The processed image of the piano. The image should only contain the piano and should be gray-scaled. 
/// @param keys_info 
void piano::detect_white_keys(Mat piano_image , struct piano_keys_info &keys_info , const struct piano_keys_info &black_keys_info) {
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
    keys_info.flipped = black_keys_info.flipped;

    copyMakeBorder(piano_image , piano_image_padding , padding , padding , padding , padding , BORDER_CONSTANT , Scalar(0));
    morphologyEx(piano_image_padding , piano_image_padding , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(3 , 3)));

    int truncated_height = 0;
    std::vector<Point>truncating_contour , truncating_best_fit_line_points;
    // add the corner points to the contour

    // sort the contours by x coordinates
    Vec4f black_keys_best_fit_line;
    for(RotatedRect rr : black_keys_info.keys_rectangle_list) {
        std::vector<Point>contour;
        rotated_rect_to_contour(rr , contour);
        for(int i = 0; i < contour.size(); i++) {
            contour[i].x += padding;
            contour[i].y += padding;
        }
        if(black_keys_info.flipped) { truncating_best_fit_line_points.push_back(contour[1]); truncating_best_fit_line_points.push_back(contour[2]); }
        else { truncating_best_fit_line_points.push_back(contour[0]); truncating_best_fit_line_points.push_back(contour[3]); }
    }
    fitLine(truncating_best_fit_line_points , black_keys_best_fit_line , DIST_L2 , 0 , 0.01 , 0.01);
    double bestfit_vx = black_keys_best_fit_line[0] , bestfit_vy = black_keys_best_fit_line[1] , bestfit_x0 = black_keys_best_fit_line[2] , bestfit_y0 = black_keys_best_fit_line[3];
    double bestfit_b = bestfit_vy/bestfit_vx;
    double bestfit_a = -(bestfit_b*bestfit_x0)+bestfit_y0;

    truncating_contour.insert(truncating_contour.begin() , Point(0 , bestfit_a));
    truncating_contour.insert(truncating_contour.begin()+1 , Point(0 , black_keys_info.flipped ? piano_image_padding.size().height : 0));
    
    truncating_contour.insert(truncating_contour.end() , truncating_best_fit_line_points.begin() , truncating_best_fit_line_points.end());
    std::sort(truncating_contour.begin() , truncating_contour.end() , [](const auto &a , const auto &b) { return (a.x < b.x); });

    truncating_contour.push_back(Point(piano_image_padding.size().width , bestfit_b*piano_image_padding.size().width+bestfit_a));
    truncating_contour.push_back(Point(piano_image_padding.size().width , black_keys_info.flipped ? piano_image_padding.size().height : 0));

    Mat testimg;
    morphologyEx(piano_image , testimg , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(3 , 3)) , Point(-1 , -1) , 3);

    piano_image_padding.copyTo(truncated);
    // draw the contour that covers all the black keys
    drawContours(truncated , std::vector<std::vector<Point>>({truncating_contour}) , -1 , 0xff , -1);

#ifdef DEBUG
    Mat debug1;
    cvtColor(truncated , debug1 , COLOR_GRAY2BGR);
    for(int i = 0; i < truncating_contour.size(); i++) {
        circle(debug1 , Point(truncating_contour[i].x+20 , truncating_contour[i].y+20) , 1 , Scalar(0x00 , 0xff , 0x00) , -1);
        putText(debug1 , std::to_string(i) , Point(truncating_contour[i].x+20 , truncating_contour[i].y+20) , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0xff , 0x00) , 2);
    }
    for(int i = 0; i < truncating_best_fit_line_points.size(); i++) {
        circle(debug1 , truncating_best_fit_line_points[i] , 1 , Scalar(0x00 , 0x00 , 0xff) , -1);
    }

    circle(debug1 , Point(0 , bestfit_a) , 2 , Scalar(0x00 , 0x00 , 0xff) , -1);
    circle(debug1 , Point(debug1.size().width , debug1.size().width*bestfit_b+bestfit_a) , 2 , Scalar(0x00 , 0x00 , 0xff) , -1);
    imshow("debug1" , debug1);

    debug1.release();
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

    // sort by x coordinate
    std::sort(keys_rect_list_1.begin() , keys_rect_list_1.end() , [](const auto &a , const auto &b) {
        return (a.center.x < b.center.x);
    });
#ifdef DEBUG
    imshow("white_piano_image" , piano_image);
#endif
    // calculate the best fit line of C.M. from the rectangles that are not height adjusted
    // this way we can remove the outliers prematurely
    std::vector<Point>premature_center_points;
    for(RotatedRect r : keys_rect_list_1) { premature_center_points.push_back(r.center); }
    Vec4f premature_best_fit_line;
    fitLine(premature_center_points , premature_best_fit_line , DIST_L2 , 0 , 0.01 , 0.01);
    // reuse previously used variable
    bestfit_vx = premature_best_fit_line[0] , bestfit_vy = premature_best_fit_line[1] , bestfit_x0 = premature_best_fit_line[2] , bestfit_y0 = premature_best_fit_line[3];
    bestfit_b = bestfit_vy/bestfit_vx;
    bestfit_a = -(bestfit_b*bestfit_x0)+bestfit_y0;
    std::vector<Point>bestfit_contour = {Point2f(0 , bestfit_a) , Point2f(0 , bestfit_a+1)
     , Point2f(piano_image_padding.size().width , (double)piano_image_padding.size().width*bestfit_b+bestfit_a+1)
     , Point2f(piano_image_padding.size().width , (double)piano_image_padding.size().width*bestfit_b+bestfit_a)};

    std::vector<RotatedRect>keys_rect_list_2;
    for(int i = 0; i < keys_rect_list_1.size(); i++) {
        std::vector<Point>contour;

        // slightly inflate the size for detection
        keys_rect_list_1[i].size.width  *= 1.5;
        keys_rect_list_1[i].size.height *= 1.5;

        Mat overlap_test1 = Mat::zeros(piano_image_padding.size() , CV_8UC1);
        Mat overlap_test2 = Mat::zeros(piano_image_padding.size() , CV_8UC1);
        rotated_rect_to_contour(keys_rect_list_1[i] , contour);
        drawContours(overlap_test1 , std::vector<std::vector<Point>>({bestfit_contour}) , -1 , Scalar(0xff) , -1);
        drawContours(overlap_test2 , std::vector<std::vector<Point>>({contour}) , -1 , Scalar(0xff) , -1);
        Mat test = overlap_test1 & overlap_test2;

        keys_rect_list_1[i].size.width  /= 1.5;
        keys_rect_list_1[i].size.height /= 1.5;
        
        if(countNonZero(test) >= 1) {
            keys_rect_list_2.push_back(keys_rect_list_1[i]);
        }

        overlap_test1.release();
        overlap_test2.release();
        test.release();
    }
#ifdef DEBUG
    Mat copy_piano_img2;
    cvtColor(piano_image_padding , copy_piano_img2 , COLOR_GRAY2BGR);
    for(int i = 0; i < keys_rect_list_2.size(); i++) {
        std::vector<Point>dc;
        rotated_rect_to_contour(keys_rect_list_2[i] , dc);
        drawContours(copy_piano_img2 , std::vector<std::vector<Point>>({dc}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    drawContours(copy_piano_img2 , std::vector<std::vector<Point>>({bestfit_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
    imshow("bestfit_line_shown" , copy_piano_img2);

    Mat copy_piano_img3;
    cvtColor(piano_image , copy_piano_img3 , COLOR_GRAY2BGR);
#endif

    std::vector<double>median_width_list , median_height_list;
    keys_info.mean_key_width = 0;
    keys_info.mean_key_height = 0;
    for(int i = 0; i < keys_rect_list_2.size(); i++) {
        // find the closest black keys from the key and set the key's height to align with black's height

        // index of the black key closest to the key
        int black_min_dist_from_key = 0;
        double min_dist = 0x7fffffff;
        for(int j = 0; j < black_keys_info.keys_rectangle_list.size(); j++) {
            double distance = euclidean_distance(keys_rect_list_2[i].center , black_keys_info.keys_rectangle_list[j].center);
            if(min_dist > distance) {
                min_dist = distance;
                black_min_dist_from_key = j;
            }
        }
        // found the closest black key
        RotatedRect closest_black = black_keys_info.keys_rectangle_list[black_min_dist_from_key];
        Point2f black_points[4] , white_points[4];
        closest_black.points(black_points);
        keys_rect_list_2[i].points(white_points);

        // adjust the height of the key
        Point2f black_p = (black_points[1].y < black_points[2].y) ? black_points[1] : black_points[2];
        Point2f white_p = (white_points[0].y > white_points[3].y) ? white_points[0] : white_points[3];
        if(black_keys_info.flipped) {
            black_p = (black_points[0].y > black_points[3].y) ? black_points[0] : black_points[3];
            white_p = (white_points[1].y < white_points[2].y) ? white_points[1] : white_points[2];
        }
        white_p.x -= padding;
        white_p.y -= padding;
        double distance = abs(black_p.y-white_p.y);
        adjust_rotated_rect_height(keys_rect_list_2[i] , distance , !black_keys_info.flipped);

        // inflate the width
        keys_rect_list_2[i].size.width += erode_iteration*2;

        // calculate mean and median
        keys_info.mean_key_width += keys_rect_list_2[i].size.width;
        keys_info.mean_key_height += keys_rect_list_2[i].size.height;
        median_width_list.push_back(keys_rect_list_2[i].size.width);
        median_height_list.push_back(keys_rect_list_2[i].size.height);

        keys_rect_list_2[i].center.x -= padding;
        keys_rect_list_2[i].center.y -= padding;
        keys_info.keys_rectangle_list.push_back(keys_rect_list_2[i]);
        
#ifdef DEBUG
        std::vector<Point>c1,c2;
        rotated_rect_to_contour(closest_black , c1);
        rotated_rect_to_contour(keys_rect_list_2[i] , c2);
        Scalar color = Scalar(rng.uniform(0 , 180) , rng.uniform(0 , 180) , rng.uniform(0 , 180));
        drawContours(copy_piano_img3 , std::vector<std::vector<Point>>({c1 , c2}) , -1 , color , 1);
#endif
    }
    keys_info.median_key_width = calculate_median(median_width_list);
    keys_info.median_key_height = calculate_median(median_height_list);
#ifdef DEBUG
    imshow("copy_piano_img3" , copy_piano_img3);
#endif
}

void piano::detect_black_keys(Mat piano_image , struct piano_keys_info &keys_info) {
    Mat only_black_keys , black_keys_enclosed , piano_image_padding;

    int rectangles_count = 0;

    int img_width = piano_image.size().width;
    int img_height = piano_image.size().height;
    int padding = 30;

    piano_image.copyTo(keys_info.piano_image);
    copyMakeBorder(piano_image , piano_image_padding , padding , padding , padding , padding , BORDER_CONSTANT , Scalar(0));

    // black key mask
    std::cout << "(detect_black_keys) 1. Performing close operation & thresholding... \n";
    // remove noises using gaussian blur
    GaussianBlur(piano_image_padding , piano_image_padding , Size(5 , 5) , 5);
    threshold(piano_image_padding , only_black_keys , -1 , 255 , THRESH_BINARY|THRESH_OTSU);
    // inRange(piano_image_padding , Scalar(0x00 , 0x00 , 0x00) , Scalar(0x33 , 0x33 , 0x33) , only_black_keys);
    // only_black_keys = ~only_black_keys;
    morphologyEx(only_black_keys , only_black_keys , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(2 , 2)));
    // dilate(only_black_keys , only_black_keys , cv::Mat() , Point(-1 , -1) , 1);

    imshow("only_black_keys" , only_black_keys);

    std::cout << "(detect_black_keys) 2. Performing convex hull... \n";
    std::vector<std::vector<Point>>contours;
    only_black_keys.copyTo(black_keys_enclosed);
    findContours(only_black_keys , contours , RETR_EXTERNAL , CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<Point>>hull(contours.size());

    // the longest contour is the contour that encloses the piano
    int i_max_arc_length = 0;
    int hull_max_arc_length = 0;
    for(int i = 0; i < contours.size(); i++) {
        int length = arcLength(contours[i] , true);
        if(hull_max_arc_length < length) {
            i_max_arc_length = i; hull_max_arc_length = length;
        }
        convexHull(contours[i] , hull[i]);
    }
    drawContours(black_keys_enclosed , hull , i_max_arc_length , 0xff , 10);
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
    // sort by x coordinate
    std::sort(keys_rect_list_1.begin() , keys_rect_list_1.end() , [](const auto &a , const auto &b) {
        return (a.center.x < b.center.x);
    });

    // check whether the image should be flipped
    bool flipped = false;
    std::vector<double>black_cm_y_list;
    int black_median_cm_y = 0;
    cvtColor(piano_image_padding , test_image , COLOR_GRAY2BGR);
    
    RotatedRect piano_rotated_rect = minAreaRect(hull[i_max_arc_length]);
    if(piano_rotated_rect.size.height > piano_rotated_rect.size.width) {
        std::swap(piano_rotated_rect.size.height , piano_rotated_rect.size.width);
        piano_rotated_rect.angle -= 90.0f;
    }
    RotatedRect piano_rotated_rect_upper(piano_rotated_rect) , piano_rotated_rect_lower(piano_rotated_rect);
    adjust_rotated_rect_height(piano_rotated_rect_upper , piano_rotated_rect.size.height/2 , false);
    adjust_rotated_rect_height(piano_rotated_rect_lower , piano_rotated_rect.size.height/2 , true);
    std::vector<Point>piano_rr_contour_upper , piano_rr_contour_lower;
    rotated_rect_to_contour(piano_rotated_rect_upper , piano_rr_contour_upper);
    rotated_rect_to_contour(piano_rotated_rect_lower , piano_rr_contour_lower);

    int upper_hit_count = 0;
    int lower_hit_count = 0;
    for(RotatedRect rr : keys_rect_list_1) {
        if(pointPolygonTest(piano_rr_contour_upper , rr.center , false) > 0) upper_hit_count++;
        if(pointPolygonTest(piano_rr_contour_lower , rr.center , false) > 0) lower_hit_count++;
    }
    flipped = upper_hit_count < lower_hit_count;
    keys_info.flipped = flipped;

    for(int i = 0; i < keys_rect_list_1.size(); i++) {
        if(keys_rect_list_1[i].size.width < keys_info.mean_key_width*0.55) continue;
        if(std::min(keys_rect_list_1[i].size.width , keys_rect_list_1[i].size.height) > keys_info.mean_key_width*1.6) continue;
        if(keys_rect_list_1[i].size.height < keys_info.mean_key_height) {
            adjust_rotated_rect_height(keys_rect_list_1[i] , keys_info.mean_key_height , !keys_info.flipped);
        }
#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        std::vector<Point>cr;
        rotated_rect_to_contour(keys_rect_list_1[i] , cr);
        drawContours(colorful , std::vector<std::vector<Point>>{cr} , -1 , color , -1);
#endif
        // compensate with the blur
        keys_rect_list_1[i].size.width -= 2;
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

static void remove_item_from_piano_info(struct piano_keys_info &keys_info , int i) {
    keys_info.keys_rectangle_list.erase(keys_info.keys_rectangle_list.begin()+i);
    if(i > 0 && i < keys_info.cm_dist_from_bestfit_list.size()-1) {
        // remove two components and recalibrate the distance
        keys_info.dist_between_keys_list[i] = euclidean_distance(keys_info.keys_rectangle_list[i-1].center , keys_info.keys_rectangle_list[i+1].center);
    }
    keys_info.dist_between_keys_list.erase(keys_info.dist_between_keys_list.begin()+i-1);
    keys_info.cm_dist_from_bestfit_list.erase(keys_info.cm_dist_from_bestfit_list.begin()+i);
}

template <typename T> static void region_divider(std::vector<T>&data_list , int number_of_regions , std::vector<std::vector<T>>&regions , std::vector<int>&region_indicator) {
    int count = data_list.size();
    double region_indexes[number_of_regions];
    for(int i = 1; i <= number_of_regions; i++) { region_indexes[i-1] = (1.0f/(double)number_of_regions)*((double)i)*count; }
    int region_index = 0;
    region_indicator.clear();
    regions.resize(number_of_regions);
    for(int i = 0; i < count; i++) {
        regions[region_index].push_back(data_list[i]);
        region_indicator.push_back(region_index);
        if(i >= floor(region_indexes[region_index])) { region_index++; }
    }
}

void piano::adjust_key_angles(struct piano_keys_info &keys_info) {
    double angle_median;
    double angle_mean = 0;
    double angle_sd;
    std::vector<double>rect_angle_list;
    for(RotatedRect r : keys_info.keys_rectangle_list) { rect_angle_list.push_back(r.angle); angle_mean += r.angle; }
    angle_mean /= keys_info.keys_rectangle_list.size();
    angle_median = calculate_median(rect_angle_list);
    angle_sd = calculate_standard_deviation(rect_angle_list , angle_mean);

    Mat image_copy = Mat::zeros(keys_info.piano_image.size() , CV_8UC3);
    cvtColor(keys_info.piano_image , image_copy , COLOR_GRAY2BGR);

    double Q1 = calculate_percentile(rect_angle_list , 0.25);
    double Q3 = calculate_percentile(rect_angle_list , 0.75);
    double left_whisker = Q1-1.5*(Q3-Q1);
    double right_whisker = Q3+1.5*(Q3-Q1);

    // calculate the regional median
    std::vector<std::vector<RotatedRect>>rect_divided_by_region;
    std::vector<int>angle_region_indicator;
    int region_size = 5;
    double angle_regional_median[region_size];
    region_divider<RotatedRect>(keys_info.keys_rectangle_list , region_size , rect_divided_by_region , angle_region_indicator);
    for(int i = 0; i < region_size; i++) {
        std::vector<double>median_list;
        for(RotatedRect rr : rect_divided_by_region[i]) { median_list.push_back(rr.angle); }
        if(median_list.size() != 0) {
            angle_regional_median[i] = calculate_median(median_list);
        }
    }

    for(int i = 0; i < keys_info.keys_rectangle_list.size(); i++) {
#ifdef DEBUG
        std::vector<Point>dc;
        rotated_rect_to_contour(keys_info.keys_rectangle_list[i] , dc);
        drawContours(image_copy , std::vector<std::vector<Point>>({dc}) , -1 , Scalar(0x00 , 0xff , 0xff) , 1);
#endif
        if(keys_info.keys_rectangle_list[i].angle >= right_whisker||keys_info.keys_rectangle_list[i].angle <= left_whisker) {
            // rotate the rectangle by its tip point
            Point2f pts[4];
            RotatedRect rr = keys_info.keys_rectangle_list[i];
            
            rr.points(pts);
            double target_angle = angle_regional_median[angle_region_indicator[i]];
            double delta_theta = target_angle-rr.angle;

            int x0 , y0;
            if(keys_info.keys_rectangle_pivot.size() == 0) {
                x0 = rr.center.x+(rr.size.height/2)*sin(rr.angle*M_PI/180.0f);
                y0 = rr.center.y-(rr.size.height/2)*cos(rr.angle*M_PI/180.0f);
            }
            else {
                x0 = keys_info.keys_rectangle_pivot[i].x;
                y0 = keys_info.keys_rectangle_pivot[i].y;
            }
            int x = keys_info.keys_rectangle_list[i].center.x , y = keys_info.keys_rectangle_list[i].center.y;

            int rotated_x = (x-x0)*cos(delta_theta*M_PI/180.0f)-(y-y0)*sin(delta_theta*M_PI/180.0f);
            int rotated_y = (x-x0)*sin(delta_theta*M_PI/180.0f)+(y-y0)*cos(delta_theta*M_PI/180.0f);
            rotated_x += x0; rotated_y += y0;
            circle(image_copy , Point(x0 , y0) , 1 , Scalar(0x00 , 0x00 , 0xff) , 2);

            circle(image_copy , Point(rr.center.x , rr.center.y) , 1 , Scalar(0xff , 0x00 , 0xff) , 2);
            circle(image_copy , Point(rotated_x , rotated_y) , 1 , Scalar(0xff , 0x00 , 0x00) , 2);

            keys_info.keys_rectangle_list[i].center.x = rotated_x;
            keys_info.keys_rectangle_list[i].center.y = rotated_y;
            
            keys_info.keys_rectangle_list[i].angle = target_angle;

#ifdef DEBUG
            std::vector<Point>dc;
            rotated_rect_to_contour(keys_info.keys_rectangle_list[i] , dc);
            circle(image_copy , Point(x0 , y0) , 2 , Scalar(0x00 , 0xff , 0x00) , -1);
            drawContours(image_copy , std::vector<std::vector<Point>>({dc}) , -1 , Scalar(0xff , 0xff , 0x00) , 1);
#endif
        }
    }
#ifdef DEBUG
    static int lol=1;
    imshow("adjust_white_outliers"+std::to_string(lol++) , image_copy);
#endif
}

void piano::adjust_key_widths(struct piano_keys_info &keys_info) {
    double width_median = keys_info.median_key_width;
    for(int i = 0; i < keys_info.keys_rectangle_list.size(); i++) {
        if(keys_info.keys_rectangle_list[i].size.width > 1.4*width_median) {
            std::cout << "outlier : " << i << "\n";
            adjust_rotated_rect_width(keys_info.keys_rectangle_list[i] , width_median);
        }
    }
}
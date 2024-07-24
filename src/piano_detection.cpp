#include "piano_detection.hpp"

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

/// @brief Filter the image to binary for contour processing
/// @param frame original frame
/// @return filtered frame
cv::Mat PianoRecognition::filter_piano_image(cv::Mat frame) {
    Mat blurred_frame , grayscaled_frame , binary_frame;
    medianBlur(frame , blurred_frame , 11);
    if(blurred_frame.type() != CV_8UC1) cvtColor(blurred_frame , grayscaled_frame , COLOR_RGB2GRAY);
    else grayscaled_frame = blurred_frame;

    threshold(grayscaled_frame , binary_frame , 90 , 255 , THRESH_OTSU|THRESH_BINARY);
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
    const int overlap_streak_threshold = 100;
    
    Mat debug_copy;
    while(1) {
        Mat frame;
        if(video.read(frame) == false) {
            std::cout << "video error!\n";
            return false;
        }

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
    const float match_thresh_ratio = 0.85;
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

        // check hit count
        if(boundingRect(cont_obj).area() >= img.size().area()*0.5) continue;
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
    std::sort(keys_info.keys_rectangle_list.begin() , keys_info.keys_rectangle_list.end() , [](const auto &a , const auto &b) {
        return (a.center.x < b.center.x);
    });

    std::vector<Point>center_points;
    for(RotatedRect r : keys_info.keys_rectangle_list) { center_points.push_back(r.center); }

    // calculate the best fits of the center masses
    Vec4f best_fit_line;
    fitLine(center_points , best_fit_line , DIST_L2 , 0 , 0.01 , 0.01);
    double bestfit_vx = best_fit_line[0] , bestfit_vy = best_fit_line[1] , bestfit_x0 = best_fit_line[2] , bestfit_y0 = best_fit_line[3];
    double bestfit_b = bestfit_vy/bestfit_vx;
    double bestfit_a = -(bestfit_b*bestfit_x0)+bestfit_y0;

    keys_info.mean_dist_from_bestfit = 0;
    keys_info.mean_dist_between_keys = 0;
    keys_info.median_dist_between_keys = 0;

    std::vector<double>dist_median_list;

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
        Point p1 = keys_info.keys_rectangle_list[i].center , p2 = keys_info.keys_rectangle_list[i+1].center;
        double distance_between_keys = sqrt(pow((p1.x-p2.x) , 2)+pow((p1.y-p2.y) , 2));
        
        keys_info.dist_between_keys_list.push_back(distance_between_keys);
        dist_median_list.push_back(distance_between_keys);
        keys_info.mean_dist_between_keys += distance_between_keys;
        std::cout << i << " : distance_between_keys = " << distance_between_keys << "\n";
    }
    std::sort(dist_median_list.begin() , dist_median_list.end());

    keys_info.median_dist_between_keys = (dist_median_list.size()%2)
        ? (dist_median_list[(dist_median_list.size()+1)/2])
        : ((dist_median_list[(dist_median_list.size()+1)/2]+dist_median_list[((dist_median_list.size()+1)/2)+1])/2);
    
    keys_info.mean_dist_from_bestfit /= keys_info.keys_rectangle_list.size();
    keys_info.mean_dist_between_keys /= (keys_info.keys_rectangle_list.size()-1);
    std::cout << "mean_dist_between_keys = " << keys_info.mean_dist_between_keys << "\n";
}

/// @brief Remove the outlier keys
/// @param keys_info piano keys info structure
void PianoRecognition::remove_outliers(struct piano_keys_info &keys_info) {
    // remove the outlier by comparing distance from the best-fit line
    for(int i = 0; i < keys_info.cm_dist_from_bestfit_list.size(); i++) {
        if(keys_info.cm_dist_from_bestfit_list[i] >= 1.8*keys_info.mean_dist_between_keys) {
            std::cout << "removing index " << i << "...\n";

            if(i > 0 && i < keys_info.cm_dist_from_bestfit_list.size()-1) {
                // remove two components and recalibrate
                Point p1 = keys_info.keys_rectangle_list[i-1].center , p2 = keys_info.keys_rectangle_list[i+1].center;
                keys_info.dist_between_keys_list[i] = sqrt(pow((p1.x-p2.x) , 2)+pow((p1.y-p2.y) , 2));
            }
            keys_info.dist_between_keys_list.erase(keys_info.dist_between_keys_list.begin()+i-1);

            keys_info.keys_rectangle_list.erase(keys_info.keys_rectangle_list.begin()+i);
            keys_info.cm_dist_from_bestfit_list.erase(keys_info.cm_dist_from_bestfit_list.begin()+i);
            i--;
            // adjust the means and medians
        }
    }
    
}

/// @brief Auto-fill the missing white keys based on the distance between the center of the keys
/// @param keys_info 
void PianoRecognition::white_auto_fill_keys(struct piano_keys_info &keys_info) {
#ifdef DEBUG
    Mat new_constructed_img;
#endif
    bool new_added = false;
    cvtColor(keys_info.piano_image , new_constructed_img , COLOR_GRAY2BGR);
    // remove the outlier by comparing distance from the best-fit line
    std::cout << "median distance : " << keys_info.median_dist_between_keys << "\n";
    for(int i = 0; i < keys_info.dist_between_keys_list.size(); i++) {
        std::cout << i << " : " << keys_info.dist_between_keys_list[i] << "\n";
        // compare with the nearby keys
        if(keys_info.dist_between_keys_list[i] >= keys_info.keys_rectangle_list[i].size.width*2.0) {
            // because the width of the key changes by the perspective of camera, use the key width as the nearest key
            // The rectangle does not cover the entirety of the key, so we inflate the width a little bit.
            // The value "1.3" is acquired from the empirical experience.

            int missing_key_count = round(keys_info.dist_between_keys_list[i]/(keys_info.keys_rectangle_list[i].size.width*1.3))-1;
            std::cout << "missing keys found!" << "\n";
            std::cout << "missing keys count : " << missing_key_count << "\n";

            Point p1 = keys_info.keys_rectangle_list[i].center;
            Point p2 = keys_info.keys_rectangle_list[i+1].center;
            double angle1 = keys_info.keys_rectangle_list[i].angle;
            double angle2 = keys_info.keys_rectangle_list[i+1].angle;
            
            // Internal division of the two points
            for(int j = 1; j < missing_key_count+1; j++) {
                int m = j , n = (missing_key_count+1)-j;
                // divided point
                Point p((m*p2.x+n*p1.x)/(m+n) , (m*p2.y+n*p1.y)/(m+n));
                circle(new_constructed_img , p , 2 , Scalar(0x00 , 0x00 , 0xff) , 1);
                // Construct a new rectangle. New rectangle has the same width and height as the neighboring rectangle
                // The angle of the new rectangle is the average of the two reference rectangle(ith and i+1th rectangles.)
                RotatedRect new_rect(p , Size(keys_info.keys_rectangle_list[i].size.width , keys_info.keys_rectangle_list[i].size.height) , (angle1+angle2)/2);
                keys_info.keys_rectangle_list.push_back(new_rect); // new ones are added
                new_added = true;
#ifdef DEBUG
                // draw the rectangles (why not?)
                std::vector<Point>rect_contour;
                rotated_rect_to_contour(new_rect , rect_contour);
                drawContours(new_constructed_img , std::vector<std::vector<Point>>({rect_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
#endif
            }
        }
    }

    if(!new_added) return;
    keys_info.dist_between_keys_list.clear();
    keys_info.cm_dist_from_bestfit_list.clear();
    write_keys_info(keys_info);
#ifdef DEBUG
    imshow("new_constructed_img" , new_constructed_img);
#endif
}

/// @brief Create the rectangular contour area of the white keys from the image of piano.
/// @param piano_image The processed image of the piano. The image should only contain the piano and should be gray-scaled. 
/// @param keys_info 
void PianoRecognition::detect_white_keys(Mat piano_image , struct piano_keys_info &keys_info) {
    Mat piano_image_padding , dilated , adaptive;
    int rectangles_count = 0;

    int padding = 20;

    piano_image.copyTo(keys_info.piano_image);
    int img_width = piano_image.size().width;
    int img_height = piano_image.size().height;
    std::cout << "(detect_white_keys) 1. creating dilated black-key mask... \n";
    // 1. create the dilated image, to mask the black keys

    dilate(piano_image , dilated , Mat() , Point(-1 , -1) , 4);
    morphologyEx(dilated , dilated , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(3 , 3)));
    threshold(dilated , dilated , -1 , 255 , THRESH_BINARY_INV|THRESH_OTSU);

    std::cout << "(detect_white_keys) 2. Performing adaptive thresholding... \n";
    // adaptive 
    adaptiveThreshold(piano_image , adaptive , 255 , ADAPTIVE_THRESH_GAUSSIAN_C , THRESH_BINARY , 11 , 5);

    std::cout << "(detect_white_keys) 3. OR operation & morphology... \n";

    Mat noise_removed = adaptive|dilated;
    erode(noise_removed , noise_removed , Mat() , Point(-1 , -1) , 2);

#ifdef DEBUG
    Mat outline_color;
    cvtColor(piano_image , outline_color , COLOR_GRAY2BGR);
    RNG rng((unsigned int)time(0));
#endif
    
    std::cout << "(detect_white_keys) 4. Finding the contours...\n";
    std::vector<std::vector<Point>>piano_contours_1 , piano_contours;
    findContours(noise_removed , piano_contours_1 , RETR_EXTERNAL , CHAIN_APPROX_SIMPLE);

    std::vector<RotatedRect>keys_rect_list_1;
    std::cout << "(detect_white_keys) 5. Finding the contours...\n";
    std::vector<double>width_median_list;
    std::vector<double>height_median_list;
    
    Mat first_contour;
    cvtColor(piano_image , first_contour , COLOR_GRAY2BGR);
    for(int i = 0; i < piano_contours_1.size(); i++) {
        /* filter out contours that are too small or too big */
        // if the arc length of the contour is smaller than the 1% of the image's width
        if(arcLength(piano_contours_1[i] , false) < 0.01*img_width) continue;
        // if the arc length of the contour is bigger than the sum of image's width and height
        if(arcLength(piano_contours_1[i] , false) >= img_width+img_height) continue;
        
        // check the bounding rect to see if width is longer than height
        // If width is longer than the height, it signifies that it is not a key. 
        Rect bounding_box = boundingRect(piano_contours_1[i]);
        if(bounding_box.width > bounding_box.height) continue;
        /* Get the bounding rectangles of the white keyboard */
        RotatedRect r = minAreaRect(piano_contours_1[i]);
        // make the shorter side to width and long sides to height
        if(r.size.width > r.size.height) {
            std::swap(r.size.width , r.size.height);
            r.angle -= 90.0f;
        }

        // push to the final list
        keys_rect_list_1.push_back(r);

        /* calculate the mean area, width and height */
        /* Also calculate the maximum height of the rectangles */
        keys_info.mean_key_area += r.size.area();
        keys_info.mean_key_width += r.size.width;
        keys_info.mean_key_height += r.size.height;
        keys_info.max_key_height = std::max(keys_info.max_key_height , (double)r.size.height);
        width_median_list.push_back(r.size.width);
        height_median_list.push_back(r.size.height);
        rectangles_count++;
    }
    std::sort(width_median_list.begin() , width_median_list.end());
    std::sort(height_median_list.begin() , height_median_list.end());
    keys_info.mean_key_area /= rectangles_count;
    keys_info.mean_key_width /= rectangles_count;
    keys_info.mean_key_height /= rectangles_count;

    // calculate the mean width and height
    keys_info.median_key_height = (height_median_list.size()%2)
        ? (height_median_list[(height_median_list.size()+1)/2])
        : ((height_median_list[(height_median_list.size()+1)/2]+height_median_list[((height_median_list.size()+1)/2)+1])/2);
    keys_info.median_key_width = (width_median_list.size()%2)
        ? (width_median_list[(width_median_list.size()+1)/2])
        : ((width_median_list[(width_median_list.size()+1)/2]+width_median_list[((width_median_list.size()+1)/2)+1])/2);
#ifdef DEBUG
    std::cout << "keys_info.mean_key_area : " << keys_info.mean_key_area << "\n";
    std::cout << "keys_info.mean_key_width : " << keys_info.mean_key_width << "\n";
    std::cout << "keys_info.mean_key_height : " << keys_info.mean_key_height << "\n";

    std::cout << "5. Removing the small contours & resizing... \n";
#endif
    int height_to_be_resized = keys_info.max_key_height;
    for(int i = 0; i < keys_rect_list_1.size(); i++) {
        std::vector<Point>bounding_rect_contour;
        // filter out anomalies
        // keys have fairly homogeneous width, so remove the keys that...
        //  - have the width longer than the 170% of the mean width
        //  - have the width shorter than the 30% of the mean width
        // Also, remove the keys that...
        //  - have smaller than 45% of the mean area

        // These percentage values are determined empirically. (There is no fancy formula that determines these values)
        if(keys_rect_list_1[i].size.width > 2.3*keys_info.mean_key_width) continue;
        if(keys_rect_list_1[i].size.width <= 0.8*keys_info.median_key_width) continue;
        if(keys_rect_list_1[i].size.area() < 0.45*keys_info.mean_key_area) continue;
        
        /* transform the location & height */
        // 1. Elongate the rectangle's size
        int d = (height_to_be_resized-keys_rect_list_1[i].size.height)/2;

        // 2. To match the location of the rectangle prior to the elongation, adjust the 
        //    location of the rectangle
        keys_rect_list_1[i].center.x += d*sin(keys_rect_list_1[i].angle*M_PI/180.0f);
        keys_rect_list_1[i].center.y -= d*cos(keys_rect_list_1[i].angle*M_PI/180.0f);
        
        keys_rect_list_1[i].size.height = height_to_be_resized;
        rotated_rect_to_contour(keys_rect_list_1[i] , bounding_rect_contour);

        // push the rectangle to the final list
        keys_info.keys_rectangle_list.push_back(keys_rect_list_1[i]);
#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        drawContours(outline_color , std::vector<std::vector<Point>>({bounding_rect_contour}) , -1 , color , -1);
#endif
    }

#ifdef DEBUG
    imshow("noise_removed" , noise_removed);
    imshow("adaptive" , adaptive);
    imshow("adaptive|dilated" , adaptive|dilated);
    imshow("colored" , outline_color);
    imwrite("finally.jpg" , outline_color);
#endif
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
    morphologyEx(piano_image_padding , only_black_keys , MORPH_CLOSE , getStructuringElement(MORPH_RECT , Size(3 , 3)));
    threshold(only_black_keys , only_black_keys , -1 , 255 , THRESH_BINARY|THRESH_OTSU);

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
    findContours(black_keys_enclosed , black_keys_contours_1 , RETR_TREE , CHAIN_APPROX_SIMPLE);

    std::cout << "(detect_black_keys) 3. Removing unnecessary rectangles... \n";
    RNG rng((unsigned int)time(0));
    Mat colorful;
    std::vector<RotatedRect>keys_rect_list_1;
    cvtColor(piano_image_padding , colorful , COLOR_GRAY2BGR);

    // prepare for the key info
    keys_info.mean_key_width = 0;
    keys_info.mean_key_height = 0;
    keys_info.mean_key_area = 0;
    keys_info.max_key_height = 0;
    for(int i = 0; i < black_keys_contours_1.size(); i++) {
        /* filter out contours that are too small or too big */
        // if the arc length of the contour is smaller than the 1% of the image's width
        if(arcLength(black_keys_contours_1[i] , false) < 0.01*img_width) continue;
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
        keys_info.mean_key_area += r.size.area();
        keys_info.mean_key_width += r.size.width;
        keys_info.mean_key_height += r.size.height;
        keys_info.max_key_height = std::max(keys_info.max_key_height , (double)r.size.height);
        rectangles_count++;
#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        std::vector<Point>cr;
        get_bounding_rect_contour(black_keys_contours_1[i] , cr);
        // drawContours(colorful , std::vector<std::vector<Point>>{cr} , -1 , color , 2);
#endif
    }
    keys_info.mean_key_area /= rectangles_count;
    keys_info.mean_key_width /= rectangles_count;
    keys_info.mean_key_height /= rectangles_count;
#ifdef DEBUG
    std::cout << "keys_info.mean_key_area : " << keys_info.mean_key_area << "\n";
    std::cout << "keys_info.mean_key_width : " << keys_info.mean_key_width << "\n";
    std::cout << "keys_info.mean_key_height : " << keys_info.mean_key_height << "\n";
#endif

    for(int i = 0; i < keys_rect_list_1.size(); i++) {
        if(keys_rect_list_1[i].size.width < 0.7*keys_info.mean_key_width||keys_rect_list_1[i].size.width > 1.8*keys_info.mean_key_width) continue;
        if(keys_rect_list_1[i].size.height < 0.7*keys_info.max_key_height||keys_rect_list_1[i].size.height > 1.8*keys_info.max_key_height) continue;

#ifdef DEBUG
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        std::vector<Point>cr;
        rotated_rect_to_contour(keys_rect_list_1[i] , cr);
        drawContours(colorful , std::vector<std::vector<Point>>{cr} , -1 , color , -1);
#endif
        // remove the padding
        keys_rect_list_1[i].center.x -= padding;
        keys_rect_list_1[i].center.y -= padding;
        // push the rectangle to the final list
        keys_info.keys_rectangle_list.push_back(keys_rect_list_1[i]);
    }
    // imshow("piano_image" , piano_image);
    // imshow("only_black_keys" , only_black_keys);
    // imshow("black_keys_enclosed" , black_keys_enclosed);
    // imshow("colorful" , colorful);
}


void PianoRecognition::recognize_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {

}
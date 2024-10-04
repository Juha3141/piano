#include <piano_area_detection.hpp>

using namespace cv;
using namespace xfeatures2d;

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

/* July 2024 */

#include <piano_area_detection.hpp>
#include <piano_key_detection.hpp>
#include <piano_note_detection.hpp>

#include <hand_detection_agent.hpp>
#include <midi_system.hpp>

using namespace cv;

void debug_print(struct piano_keys_info &keys_info , const char *win);
void debug_print_colorful(struct piano_keys_info &keys_info , const char *win);
void debug_print_both(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info , const char *win);
void debug_print_notes(struct piano_keys_info &white_keys_info , const char *win);

// not implemented yet
void frame_filter_func(Mat &frame) {}

int main(int argc , char **argv) {
    bool is_video = false;

    Mat typical_piano = imread("most_normal_piano.jpg" , IMREAD_GRAYSCALE);
    if(argc != 3) {
        std::cout << "piano [image/video] [option(img/vid)]\n";
        return -1;
    }
    if(strcmp(argv[2] , "img") == 0) is_video = false;
    else if(strcmp(argv[2] , "vid") == 0) is_video = true;
    else {
        std::cout << "piano [image/video] [option(img/vid)]\n";
        return -1;
    }

    PianoRecognition piano(argv[1] , 0 , 1280 , 720 , 60 , typical_piano);
    piano.set_frame_filter(frame_filter_func);

    std::vector<Point>contour , bounding_rect_contour , convex_hull;
    RotatedRect bounding_rect;
// if it's a video
if(is_video) {
    if(piano.process_piano_calibration() == false) {
        std::cout << "calibration failed!\n";
        return -1;
    }
}
else {
    Mat img = imread(argv[1] , IMREAD_COLOR);
    Mat resized;

    int width = 1024;
    resize(img , resized , Size(width , ((float)img.size().height*((float)width/(float)img.size().width)) ));
    img = resized;
    std::cout << "resized : " << img.size() << "\n";

    if(piano.recognize_piano(img , contour) == false) {
        std::cout << "recognizing piano failed!\n";
        return -1;
    }
    
    bounding_rect = minAreaRect(contour);
    piano.set_detected_piano_img(img , contour , bounding_rect);
}

    std::cout << "calibration done!" << "\n";
    Mat frame = piano.piano_bounding_rect_img;
    Mat mask = Mat::zeros(frame.size() , CV_8UC3);
    
    std::cout << "creating bounding rectangle... \n";
    get_bounding_rect_contour(piano.piano_bounding_contour , bounding_rect_contour);
    
    // To-do : find the better way
    convexHull(piano.piano_bounding_contour , convex_hull);
    std::cout << "drawing mask... ";
    drawContours(mask , std::vector<std::vector<Point>>({convex_hull}) , 0 , Scalar(0xff , 0xff , 0xff) , -1);

    Rect mask_bounding_rect = boundingRect(bounding_rect_contour);
    mask_bounding_rect.x = std::max(mask_bounding_rect.x , 0);
    mask_bounding_rect.y = std::max(mask_bounding_rect.y , 0);
    mask_bounding_rect.width = std::min(mask_bounding_rect.x+mask_bounding_rect.width , frame.size().width-1)-mask_bounding_rect.x;
    mask_bounding_rect.height = std::min(mask_bounding_rect.y+mask_bounding_rect.height , frame.size().height-1)-mask_bounding_rect.y;
    piano.piano_loc_x = mask_bounding_rect.x;
    piano.piano_loc_y = mask_bounding_rect.y;
    Mat only_piano = (frame & mask)(mask_bounding_rect);
    std::cout << "done!\n";

    // make it gray-scaled
    cvtColor(only_piano , only_piano , COLOR_BGR2GRAY);

    struct piano_keys_info white_keys_info;
    struct piano_keys_info black_keys_info;
    piano::detect_black_keys(only_piano , black_keys_info);
    piano::detect_white_keys(only_piano , white_keys_info , black_keys_info);
    // enlarge the black key's width
    write_keys_info(black_keys_info);
    write_keys_info(white_keys_info);

    std::cout << "adjusting white outliers... \n";
    piano::adjust_key_angles(white_keys_info);
    piano::adjust_key_widths(white_keys_info);
    std::cout << "adjusting black outliers... \n";
    piano::adjust_key_angles(black_keys_info);
    piano::adjust_key_widths(black_keys_info);
    debug_print_colorful(white_keys_info , "white_final");
    debug_print_colorful(black_keys_info , "black_final");

    std::cout << "detecting missing black keys... \n"; // !!
    piano::detect_missing_black_keys(black_keys_info , white_keys_info);
    debug_print(white_keys_info , "white_keys_bestfit_line");
    debug_print_both(white_keys_info , black_keys_info , "white_black_both");

    std::cout << "detecting missing white keys... \n";
    piano::detect_missing_white_keys(white_keys_info);
    std::cout << "detecting white key shapes... \n";
    piano::detect_white_key_shapes(white_keys_info , black_keys_info);
    std::cout << "detecting white key notes... `\n";
    piano::detect_white_key_notes(white_keys_info , black_keys_info);
    debug_print_both(white_keys_info , black_keys_info , "before_filling");
    std::cout << "filling the missing white keys out... \n";
    piano::fill_missing_white_keys(white_keys_info);
    std::cout << "double-checking the keys... \n";
    piano::doublecheck_white_keys(white_keys_info);
    piano::detect_black_key_notes(white_keys_info , black_keys_info);
    piano::doublecheck_black_keys(black_keys_info);
    debug_print_both(white_keys_info , black_keys_info , "win3");
    debug_print_notes(white_keys_info , "win4");

    std::cout << "white keys count : " << white_keys_info.keys_rectangle_list.size() << "\n";
    std::cout << "black keys count : " << black_keys_info.keys_rectangle_list.size() << "\n";

if(!is_video) {
    while(1) { if(waitKey(0) == 27) break; }
    return -1;
}

    std::cout << "initializing the agent...\n";
    if(hand_detection::initialize_agent(argv[1]) == false) {
        std::cout << "initializing failed!\n";
        return -1;
    }
    hand_detection::execute_agent();
    int piano_loc_x = piano.piano_loc_x;
    int piano_loc_y = piano.piano_loc_y;
    while(1) {
        Mat frame;
        hands_info_t hands_info;
        if(!hand_detection::fetch_hand_data(frame , hands_info)) continue;

        std::vector<RotatedRect>white_rectangle_copy , black_rectangle_copy;
        std::copy(white_keys_info.keys_rectangle_list.begin() , white_keys_info.keys_rectangle_list.end() , std::back_inserter(white_rectangle_copy));
        std::copy(black_keys_info.keys_rectangle_list.begin() , black_keys_info.keys_rectangle_list.end() , std::back_inserter(black_rectangle_copy));
        relocate_rotated_rect_list(white_rectangle_copy , piano_loc_x , piano_loc_y);
        relocate_rotated_rect_list(black_rectangle_copy , piano_loc_x , piano_loc_y);

        Scalar white_key_color(0x00 , 0xff , 0x00);
        Scalar black_key_color(0x00 , 0xff , 0xff);
        for(int c = 0; c < white_rectangle_copy.size(); c++) {
            std::vector<Point>rc;
            rotated_rect_to_contour(white_rectangle_copy[c] , rc);
            drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , white_key_color , 2);
            putText(frame , number_to_note_string(white_keys_info.key_notes[c].first) , Point(white_rectangle_copy[c].center.x , white_rectangle_copy[c].center.y+(white_rectangle_copy[c].size.height/3)) , FONT_HERSHEY_SIMPLEX , 0.3 , Scalar(0x00 , 0x00 , 0xff) , 1);
        }
        for(int c = 0; c < black_rectangle_copy.size(); c++) {
            std::vector<Point>rc;
            rotated_rect_to_contour(black_rectangle_copy[c] , rc);
            drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , black_key_color , 2);
        }
        bool hand_on_black_left = false;
        bool hand_on_black_right = false;
        if(hands_info.left_hand_visible) {
            for(int i : std::vector<int>({4 , 8 , 12 , 16 , 20})) {
                int x = hands_info.left_hand_landmarks_xlist[i] , y = hands_info.left_hand_landmarks_ylist[i];
                for(int c = 0; c < black_rectangle_copy.size(); c++) {
                    std::vector<Point>rc;
                    rotated_rect_to_contour(black_rectangle_copy[c] , rc);
                    if(pointPolygonTest(rc , Point(x , y) , false) > 0) {
                        drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
                        hand_on_black_left = true;
                    }
                }
                if(hand_on_black_left == false) { 
                    for(int c = 0; c < white_rectangle_copy.size(); c++) {
                        std::vector<Point>rc;
                        rotated_rect_to_contour(white_rectangle_copy[c] , rc);
                        if(pointPolygonTest(rc , Point(x , y) , false) > 0) {
                            drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
                        }
                    }
                }
                circle(frame , Point(x , y) , 2 , Scalar(0x00 , 0xff , 0x00));
                putText(frame , std::to_string(i) , Point(x , y)  , FONT_HERSHEY_SIMPLEX , 0.5 , Scalar(0x00 , 0x00 , 0xff) , 2);
            }
        }
        if(hands_info.right_hand_visible) {
            for(int i : std::vector<int>({4 , 8 , 12 , 16 , 20})) {
                int x = hands_info.right_hand_landmarks_xlist[i] , y = hands_info.right_hand_landmarks_ylist[i];
                for(int c = 0; c < black_rectangle_copy.size(); c++) {
                    std::vector<Point>rc;
                    rotated_rect_to_contour(black_rectangle_copy[c] , rc);
                    if(pointPolygonTest(rc , Point(x , y) , false) > 0) {
                        drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
                        hand_on_black_right = true;
                        std::cout << "finger #" << i << " : " << black_keys_info.key_notes[i].first << "\n";
                    }
                }
                if(!hand_on_black_right) {
                    for(int c = 0; c < white_rectangle_copy.size(); c++) {
                        std::vector<Point>rc;
                        rotated_rect_to_contour(white_rectangle_copy[c] , rc);
                        if(pointPolygonTest(rc , Point(x , y) , false) > 0) {
                            drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
                        }
                    }
                }
                circle(frame , Point(x , y) , 2 , Scalar(0xff , 0x0 , 0x00));
                putText(frame , std::to_string(i) , Point(x , y)  , FONT_HERSHEY_SIMPLEX , 0.5 , Scalar(0xff , 0x00 , 0x00) , 2);
            }
        }
        imshow("ai" , frame);
        if(hand_detection::check_agent_running() == false) {
            std::cout << "Agent killed! exiting the program...\n";
            return -1;
        }
        if(waitKey(1) == 27) break;
    }

    std::cout << "killing the child process... \n";
    hand_detection::end();
    return 0;
}

void debug_print(struct piano_keys_info &keys_info , const char *win) {
    Mat only_piano_copy;
    cvtColor(keys_info.piano_image , only_piano_copy , COLOR_GRAY2BGR);
    // draw the best fit line
    for(double x = 0; x < only_piano_copy.size().width; x += 1) {
        Point point(x , (keys_info.cm_bestfit_b*x)+keys_info.cm_bestfit_a);
        circle(only_piano_copy , point , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    for(int i = 0; i < keys_info.keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(keys_info.keys_rectangle_list[i] , rectangle_contour);

        drawContours(only_piano_copy , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0x00 , 0x00 , 0xff) , 1);
        circle(only_piano_copy , keys_info.keys_rectangle_list[i].center , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
        if(i < keys_info.keys_rectangle_list.size()-1) {
            line(only_piano_copy , keys_info.keys_rectangle_list[i].center , keys_info.keys_rectangle_list[i+1].center , Scalar(0x00 , 0xff , 0x00) , 1);
        }
    }
    imshow(win , only_piano_copy);
}

void debug_print_colorful(struct piano_keys_info &keys_info , const char *win) {
    Mat only_piano_copy;
    cvtColor(keys_info.piano_image , only_piano_copy , COLOR_GRAY2BGR);
    RNG rng((unsigned int)time(0));
    for(int i = 0; i < keys_info.keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        rotated_rect_to_contour(keys_info.keys_rectangle_list[i] , rectangle_contour);

        drawContours(only_piano_copy , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , color , -1);
        circle(only_piano_copy , keys_info.keys_rectangle_list[i].center , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    imshow(win , only_piano_copy);
}

void debug_print_both(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info , const char *win) {
    Mat image;
    cvtColor(white_keys_info.piano_image , image , COLOR_GRAY2BGR);
    for(RotatedRect r : white_keys_info.keys_rectangle_list) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(r , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    for(RotatedRect r : black_keys_info.keys_rectangle_list) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(r , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
    }
    imshow(win , image);
}

void debug_print_notes(struct piano_keys_info &white_keys_info , const char *win) {
    Mat image = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC3);
    cvtColor(white_keys_info.piano_image , image , COLOR_GRAY2BGR);
    if(white_keys_info.key_notes.size() != white_keys_info.keys_rectangle_list.size()) {
        std::cout << "key notes discrepency error!\n";
        return;
    }
    for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(white_keys_info.keys_rectangle_list[i] , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);

        // putText(image , std::to_string(i) , white_keys_info.keys_rectangle_list[i].center , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0x00 , 0x00) , 1);
        putText(image , number_to_note_string(white_keys_info.key_notes[i].first) , Point(white_keys_info.keys_rectangle_list[i].center.x , white_keys_info.keys_rectangle_list[i].center.y+(white_keys_info.keys_rectangle_list[i].size.height/3)) , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0x00 , 0xff) , 1);
        
    }
    imshow(win , image);
}
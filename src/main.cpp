/* July 2024 */

#include <piano_area_detection.hpp>
#include <piano_detection.hpp>
#include <piano_debug.hpp>

#include <hand_detection_agent.hpp>
#include <midi_system.hpp>

using namespace cv;

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

    PianoRecognition piano_recognizer(argv[1] , 0 , 1280 , 720 , 60 , typical_piano);
    piano_recognizer.set_frame_filter(frame_filter_func);

    std::vector<Point>contour , bounding_rect_contour , convex_hull;
    RotatedRect bounding_rect;
// if it's a video
if(is_video) {
    if(piano_recognizer.process_piano_calibration() == false) {
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

    if(piano_recognizer.recognize_piano(img , contour) == false) {
        std::cout << "recognizing piano_recognizer failed!\n";
        return -1;
    }
    
    bounding_rect = minAreaRect(contour);
    piano_recognizer.set_detected_piano_img(img , contour , bounding_rect);
}

    std::cout << "calibration done!" << "\n";
    Mat frame = piano_recognizer.piano_bounding_rect_img;
    Mat mask = Mat::zeros(frame.size() , CV_8UC3);
    
    std::cout << "creating bounding rectangle... \n";
    get_bounding_rect_contour(piano_recognizer.piano_bounding_contour , bounding_rect_contour);
    
    // To-do : find the better way
    convexHull(piano_recognizer.piano_bounding_contour , convex_hull);
    std::cout << "drawing mask... ";
    drawContours(mask , std::vector<std::vector<Point>>({convex_hull}) , 0 , Scalar(0xff , 0xff , 0xff) , -1);

    Rect mask_bounding_rect = boundingRect(bounding_rect_contour);
    mask_bounding_rect.x = std::max(mask_bounding_rect.x , 0);
    mask_bounding_rect.y = std::max(mask_bounding_rect.y , 0);
    mask_bounding_rect.width = std::min(mask_bounding_rect.x+mask_bounding_rect.width , frame.size().width-1)-mask_bounding_rect.x;
    mask_bounding_rect.height = std::min(mask_bounding_rect.y+mask_bounding_rect.height , frame.size().height-1)-mask_bounding_rect.y;
    piano_recognizer.piano_loc_x = mask_bounding_rect.x;
    piano_recognizer.piano_loc_y = mask_bounding_rect.y;
    Mat only_piano = (frame & mask)(mask_bounding_rect);
    std::cout << "done!\n";

    // make it gray-scaled
    cvtColor(only_piano , only_piano , COLOR_BGR2GRAY);

    /******** Piano key and note detection ********/

    PianoInfo piano_info;
    piano_info.set_piano_image(only_piano);

    piano_info.detect_black_keys();
    piano_info.detect_white_keys();
    // enlarge the black key's width

    std::cout << "adjusting white outliers... \n";
    PianoInfo::adjust_key_angles(piano_info.white_keys_info);
    PianoInfo::adjust_key_widths(piano_info.white_keys_info);

    std::cout << "adjusting black outliers... \n";
    PianoInfo::adjust_key_angles(piano_info.black_keys_info);
    PianoInfo::adjust_key_widths(piano_info.black_keys_info);

    std::cout << "detecting missing white spots... \n";
    piano_info.detect_white_missing_spots();

    debug_print_colorful(piano_info , white , "white_final");
    debug_print_colorful(piano_info , black , "black_final");

    std::cout << "detecting white key shapes... \n";
    piano_info.detect_white_key_shapes();
    std::cout << "detecting white key notes... `\n";
    piano_info.detect_white_key_notes();
    piano_info.adjust_wrong_white_notes();
    debug_print_both(piano_info , "before_filling");
    std::cout << "filling the missing white keys out... \n";
    piano_info.fill_missing_white_keys();


    piano_info.detect_black_key_notes();
    debug_print_notes(piano_info , "white_key_notes");

    std::cout << "white keys count : " << piano_info.white_keys_info.keys_rectangle_list.size() << "\n";
    std::cout << "black keys count : " << piano_info.black_keys_info.keys_rectangle_list.size() << "\n";

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
    int piano_loc_x = piano_recognizer.piano_loc_x;
    int piano_loc_y = piano_recognizer.piano_loc_y;
    while(1) {
        Mat frame;
        hands_info_t hands_info;
        if(!hand_detection::fetch_hand_data(frame , hands_info)) continue;

        std::vector<RotatedRect>white_rectangle_copy , black_rectangle_copy;
        std::copy(piano_info.white_keys_info.keys_rectangle_list.begin() , piano_info.white_keys_info.keys_rectangle_list.end() , std::back_inserter(white_rectangle_copy));
        std::copy(piano_info.black_keys_info.keys_rectangle_list.begin() , piano_info.black_keys_info.keys_rectangle_list.end() , std::back_inserter(black_rectangle_copy));
        relocate_rotated_rect_list(white_rectangle_copy , piano_loc_x , piano_loc_y);
        relocate_rotated_rect_list(black_rectangle_copy , piano_loc_x , piano_loc_y);

        Scalar white_key_color(0x00 , 0xff , 0x00);
        Scalar black_key_color(0x00 , 0xff , 0xff);
        for(int c = 0; c < white_rectangle_copy.size(); c++) {
            std::vector<Point>rc;
            rotated_rect_to_contour(white_rectangle_copy[c] , rc);
            drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , white_key_color , 2);
            putText(frame , number_to_note_string(piano_info.white_keys_info.key_notes[c].first) , Point(white_rectangle_copy[c].center.x , white_rectangle_copy[c].center.y+(white_rectangle_copy[c].size.height/3)) , FONT_HERSHEY_SIMPLEX , 0.3 , Scalar(0x00 , 0x00 , 0xff) , 1);
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
                        std::cout << "finger #" << i << " : " << piano_info.black_keys_info.key_notes[i].first << "\n";
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
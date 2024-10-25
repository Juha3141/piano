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
    std::cout << "detecting white key notes... \n";
    piano_info.detect_white_key_notes();
    piano_info.adjust_wrong_white_notes();

    debug_print_notes(piano_info , white , "before_filling");
    std::cout << "filling the missing white keys out... \n";
    piano_info.fill_missing_white_keys();

    piano_info.detect_black_key_notes();
    piano_info.elongate_white_keys();
    debug_print_notes(piano_info , white , "white_key_notes");
    debug_print_notes(piano_info , black , "black_key_notes");

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
    std::vector<RotatedRect>white_rectangle_copy , black_rectangle_copy;
    std::copy(piano_info.white_keys_info.keys_rectangle_list.begin() , piano_info.white_keys_info.keys_rectangle_list.end() , std::back_inserter(white_rectangle_copy));
    std::copy(piano_info.black_keys_info.keys_rectangle_list.begin() , piano_info.black_keys_info.keys_rectangle_list.end() , std::back_inserter(black_rectangle_copy));
    relocate_rotated_rect_list(white_rectangle_copy , piano_loc_x , piano_loc_y);
    relocate_rotated_rect_list(black_rectangle_copy , piano_loc_x , piano_loc_y);

    int music_sheet_index = 0;
    std::vector<std::vector<piano_note_info_t>>music_sheet = {
        {{TO_NOTE(NOTE_G , 3),0,0} , {TO_NOTE(NOTE_B , 3),0,0} , {TO_NOTE(NOTE_D , 4),0,0} , {TO_NOTE(NOTE_Fsharp , 4),0,0} , {TO_NOTE(NOTE_G , 4),0,0}}
    };
    std::cout << "NOTE TO PLAY : " << "\n";
    for(piano_note_info_t t : music_sheet[music_sheet_index]) {
        std::cout << number_to_note_string(t.note) << OCTAVE(t.note) << "(" << t.finger_number << ") ";
    }
    std::cout << "\n";
    while(1) {
        Mat frame;
        hands_info_t hands_info;
        if(!hand_detection::fetch_hand_data(frame , hands_info)) continue;

        
        // first  : indicates the note
        // second : right/left hand, true : right, false : left
        // third  : indicates the finger number
        std::vector<piano_note_info_t>white_finger_list;
        std::vector<piano_note_info_t>black_finger_list;

        std::vector<piano_note_info_t>total_finger_list;

        hand_detection::detect_key_landmark_overlaps(frame , hands_info , white_finger_list , black_finger_list , piano_info , white_rectangle_copy , black_rectangle_copy);
        // std::cout << "-----------------\n";

        std::vector<std::pair<int , bool>>correctly_placed_fingers; // fingers that are located on the keys to press

        hand_detection::compare_with_music_sheet(white_finger_list , total_finger_list , music_sheet[music_sheet_index] , correctly_placed_fingers);

        if(correctly_placed_fingers.size() != 0) {
            std::cout << " ----- good ones ----- \n";
            for(std::pair<int , bool>p : correctly_placed_fingers) {
                std::cout << "finger #" << p.first << "\n";
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
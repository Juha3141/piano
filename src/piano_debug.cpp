#include <piano_debug.hpp>

using namespace cv;

// true = white , false = black
void debug_print(const PianoInfo &piano_info , white_or_black_t white_or_black , const char *win) {
    Mat only_piano_copy;
    piano_keys_info_t *keys_info = white_or_black ? (piano_keys_info_t *)&piano_info.white_keys_info : (piano_keys_info_t *)&piano_info.black_keys_info;
    cvtColor(piano_info.piano_image , only_piano_copy , COLOR_GRAY2BGR);
    // draw the best fit line
    for(double x = 0; x < only_piano_copy.size().width; x += 1) {
        Point point(x , (keys_info->cm_bestfit_b*x)+keys_info->cm_bestfit_a);
        circle(only_piano_copy , point , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    for(int i = 0; i < keys_info->keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(keys_info->keys_rectangle_list[i] , rectangle_contour);

        drawContours(only_piano_copy , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0x00 , 0x00 , 0xff) , 1);
        circle(only_piano_copy , keys_info->keys_rectangle_list[i].center , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
        if(i < keys_info->keys_rectangle_list.size()-1) {
            line(only_piano_copy , keys_info->keys_rectangle_list[i].center , keys_info->keys_rectangle_list[i+1].center , Scalar(0x00 , 0xff , 0x00) , 1);
        }
    }
    imshow(win , only_piano_copy);
}

void debug_print_colorful(const PianoInfo &piano_info , white_or_black_t white_or_black , const char *win) {
    Mat only_piano_copy;
    piano_keys_info_t *keys_info = white_or_black ? (piano_keys_info_t *)&piano_info.white_keys_info : (piano_keys_info_t *)&piano_info.black_keys_info;
    cvtColor(piano_info.piano_image , only_piano_copy , COLOR_GRAY2BGR);
    RNG rng((unsigned int)time(0));
    for(int i = 0; i < keys_info->keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        rotated_rect_to_contour(keys_info->keys_rectangle_list[i] , rectangle_contour);

        drawContours(only_piano_copy , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , color , -1);
        circle(only_piano_copy , keys_info->keys_rectangle_list[i].center , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    imshow(win , only_piano_copy);
}

void debug_print_both(PianoInfo &piano_info , const char *win) {
    Mat image;
    cvtColor(piano_info.piano_image , image , COLOR_GRAY2BGR);
    for(RotatedRect r : piano_info.white_keys_info.keys_rectangle_list) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(r , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);
    }
    for(RotatedRect r : piano_info.black_keys_info.keys_rectangle_list) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(r , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
    }
    imshow(win , image);
}

void debug_print_notes(PianoInfo &piano_info , const char *win) {
    Mat image = Mat::zeros(piano_info.piano_image.size() , CV_8UC3);
    cvtColor(piano_info.piano_image , image , COLOR_GRAY2BGR);
    if(piano_info.white_keys_info.key_notes.size() != piano_info.white_keys_info.keys_rectangle_list.size()) {
        std::cout << "key notes discrepency error!\n";
        return;
    }
    for(int i = 0; i < piano_info.white_keys_info.keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(piano_info.white_keys_info.keys_rectangle_list[i] , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);

        // putText(image , std::to_string(i) , piano_info.white_keys_info.keys_rectangle_list[i].center , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0x00 , 0x00) , 1);
        int dx = -(piano_info.white_keys_info.keys_rectangle_list[i].size.height/3)*sin(piano_info.white_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
        int dy = (piano_info.white_keys_info.keys_rectangle_list[i].size.height/3)*cos(piano_info.white_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
        putText(image , number_to_note_string(piano_info.white_keys_info.key_notes[i].first) , Point(piano_info.white_keys_info.keys_rectangle_list[i].center.x+dx , piano_info.white_keys_info.keys_rectangle_list[i].center.y+dy) , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0x00 , 0xff) , 1);
        
    }
    imshow(win , image);
}

void debug_print_shapes(PianoInfo &piano_info , const char *win) {
    Mat image = Mat::zeros(piano_info.piano_image.size() , CV_8UC3);
    cvtColor(piano_info.piano_image , image , COLOR_GRAY2BGR);
    for(int i = 0; i < piano_info.white_keys_info.keys_rectangle_list.size(); i++) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(piano_info.white_keys_info.keys_rectangle_list[i] , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0xff , 0x00 , 0x00) , 1);

        // putText(image , std::to_string(i) , white_keys_info.keys_rectangle_list[i].center , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0x00 , 0x00) , 1);
        int dx = -(piano_info.white_keys_info.keys_rectangle_list[i].size.height/3)*sin(piano_info.white_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
        int dy = (piano_info.white_keys_info.keys_rectangle_list[i].size.height/3)*cos(piano_info.white_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
        putText(image , std::to_string(piano_info.white_keys_info.white_key_shapes[i]) , Point(piano_info.white_keys_info.keys_rectangle_list[i].center.x+dx , piano_info.white_keys_info.keys_rectangle_list[i].center.y+dy) , FONT_HERSHEY_SIMPLEX , 0.4 , Scalar(0x00 , 0x00 , 0xff) , 1);
        
    }
    imshow(win , image);
}
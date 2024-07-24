/* July 2024, 💀💀💀 */

#include "piano_detection.hpp"

using namespace cv;

// #define CALIBRATION_VIDEO

void debug_print(struct piano_keys_info &keys_info , const char *win);
void debug_print_both(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info , const char *win);

int main(int argc , char **argv) {
    Mat typical_piano = imread("most_normal_piano-2.jpg" , IMREAD_GRAYSCALE);
    if(argc < 2) {
        std::cout << "piano [video]\n";
        return -1;
    }

    PianoRecognition piano(argv[1] , CAP_V4L2 , 1280 , 720 , 60 , typical_piano);

    std::vector<Point>contour , bounding_rect_contour;
    RotatedRect bounding_rect;
#ifdef CALIBRATION_VIDEO
    if(piano.process_piano_calibration() == false) {
        std::cout << "calibration failed!\n";
        return -1;
    }
#else
    Mat img = imread(argv[1] , IMREAD_COLOR);
    if(argc == 3 && strcmp(argv[2] , "flip") == 0) {
        flip(img , img , 0);
    }
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
#endif

    std::cout << "calibration done!" << "\n";
    Mat frame = piano.piano_bounding_rect_img;
    Mat mask = Mat::zeros(frame.size() , CV_8UC3);
    
    std::cout << "creating bounding rectangle... \n";
    get_bounding_rect_contour(piano.piano_bounding_contour , bounding_rect_contour);
    std::cout << "drawing mask... ";
    std::vector<std::vector<Point>>dummy_c_l = {bounding_rect_contour};
    drawContours(mask , dummy_c_l , 0 , Scalar(0xff , 0xff , 0xff) , -1);
    
    Rect mask_bounding_rect = boundingRect(bounding_rect_contour);
    mask_bounding_rect.x = std::max(mask_bounding_rect.x , 0);
    mask_bounding_rect.y = std::max(mask_bounding_rect.y , 0);
    mask_bounding_rect.width = std::min(mask_bounding_rect.x+mask_bounding_rect.width , frame.size().width-1)-mask_bounding_rect.x;
    mask_bounding_rect.height = std::min(mask_bounding_rect.y+mask_bounding_rect.height , frame.size().height-1)-mask_bounding_rect.y;
    Mat only_piano = (frame & mask)(mask_bounding_rect);
    std::cout << "done!\n";

    // make it gray-scaled
    cvtColor(only_piano , only_piano , COLOR_BGR2GRAY);
    // smooth the image
    morphologyEx(only_piano , only_piano , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(3 , 3)));
    Mat template_piano = imread("bestpiano.png" , IMREAD_GRAYSCALE);
    imshow("only_piano" , only_piano);

    struct piano_keys_info white_keys_info;
    struct piano_keys_info black_keys_info;
    piano.detect_white_keys(only_piano , white_keys_info);
    piano.detect_black_keys(only_piano , black_keys_info);

    piano.write_keys_info(black_keys_info);
    piano.write_keys_info(white_keys_info);
    
    piano.remove_outliers(white_keys_info);
    piano.white_auto_fill_keys(white_keys_info);
    debug_print(white_keys_info , "win1");
    debug_print(black_keys_info , "win2");
    debug_print_both(white_keys_info , black_keys_info , "win3");

    while(1) { if(waitKey(0) == 27) break; }

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

void debug_print_both(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info , const char *win) {
    Mat image;
    cvtColor(white_keys_info.piano_image , image , COLOR_GRAY2BGR);
    for(RotatedRect r : white_keys_info.keys_rectangle_list) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(r , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
    }
    for(RotatedRect r : black_keys_info.keys_rectangle_list) {
        std::vector<Point>rectangle_contour;
        rotated_rect_to_contour(r , rectangle_contour);
        drawContours(image , std::vector<std::vector<Point>>({rectangle_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 2);
    }
    imshow(win , image);
}
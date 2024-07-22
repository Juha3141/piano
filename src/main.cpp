#include "piano_detection.hpp"

using namespace cv;

// #define CALIBRATION_VIDEO

int main(int argc , char **argv) {
    Mat typical_piano = imread("most_normal_piano-2.jpg" , IMREAD_GRAYSCALE);
    if(argc != 2) {
        std::cout << "piano [video]\n";
        return -1;
    }

    PianoRecognition piano(argv[1] , CAP_V4L2 , 1280 , 720 , 60 , typical_piano);

    std::vector<Point>contour , bounding_rect_contour;
#ifdef CALIBRATION_VIDEO
    if(piano.process_piano_calibration() == false) {
        std::cout << "calibration failed!\n";
        return -1;
    }
#else
    Mat img = imread(argv[1] , IMREAD_COLOR);
    Mat resized;
    resize(img , resized , Size(img.size().width*0.5 , img.size().height*0.5));
    img = resized;

    if(piano.recognize_piano(img , contour) == false) {
        std::cout << "recognizing piano failed!\n";
        return -1;
    }
    
    piano.set_detected_piano_img(img , contour , bounding_rect_contour);
#endif

    std::cout << "calibration done!" << "\n";
    Mat frame = piano.piano_bounding_rect_img;
    Mat mask = Mat::zeros(frame.size() , CV_8UC3);
    
    std::cout << "creating bounding rectangle... \n";
    get_bounding_rect_contour(piano.piano_bounding_contour , bounding_rect_contour);
    std::cout << "drawing mask... \n";
    std::vector<std::vector<Point>>dummy_c_l = {bounding_rect_contour};
    drawContours(mask , dummy_c_l , 0 , Scalar(0xff , 0xff , 0xff) , -1);
    Mat only_piano = (frame & mask);
    
    waitKey(0);
    
    Rect mask_bounding_rect = boundingRect(bounding_rect_contour);
    mask_bounding_rect.x = std::max(mask_bounding_rect.x , 0);
    mask_bounding_rect.y = std::max(mask_bounding_rect.y , 0);
    mask_bounding_rect.width = std::min(mask_bounding_rect.x+mask_bounding_rect.width , frame.size().width-1)-mask_bounding_rect.x;
    mask_bounding_rect.height = std::min(mask_bounding_rect.y+mask_bounding_rect.height , frame.size().height-1)-mask_bounding_rect.y;
    Mat mask_truncated = only_piano(mask_bounding_rect);
    namedWindow("win2" , WINDOW_NORMAL);
    namedWindow("win3" , WINDOW_NORMAL);
    namedWindow("win4" , WINDOW_NORMAL);
    resizeWindow("win2" , 1024 , 768);
    resizeWindow("win3" , 1024 , 768);
    resizeWindow("win4" , 1024 , 768);
    imshow("win2" , frame);
    imshow("win3" , only_piano);
    imshow("win4" , mask);
    imshow("win4" , mask_truncated);
    waitKey(0);
    
/*
    // Mat edges;
    // Canny(only_piano , edges , 100 , 200);
    Mat eroded , eroded_gray;
    Mat eroded_inrange;
    erode(only_piano , eroded , Mat() , Point(-1 , -1) , 3);
    cvtColor(eroded , eroded_gray , COLOR_BGR2GRAY);
    // inRange(eroded , Scalar(0x00 , 0x00 , 0x00) , Scalar(0x30 , 0x30 , 0x30))
    std::vector<std::vector<Point>>contours;
    
    namedWindow("winA" , WINDOW_NORMAL);
    namedWindow("winB" , WINDOW_NORMAL);
    resizeWindow("winA" , 1024 , 768);
    resizeWindow("winB" , 1024 , 768);

    imshow("winA" , frame);
    imshow("winB" , eroded_gray);
    // imshow("win3" , adjusted);
    // imshow("win3" , edges);
    while(1) { if(waitKey(0) == 27) break; }
*/
    return 0;
}
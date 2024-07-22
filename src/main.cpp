#include "piano_detection.hpp"

using namespace cv;

#define CALIBRATION_VIDEO

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

    int width = 1024;
    resize(img , resized , Size(width , ((float)img.size().height*((float)width/(float)img.size().width)) ));
    img = resized;
    std::cout << "resized : " << img.size() << "\n";

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

    Mat adaptive , dilated;

    // dilated
    dilate(only_piano , dilated , Mat() , Point(-1 , -1) , 4);
    morphologyEx(dilated , dilated , MORPH_OPEN , getStructuringElement(MORPH_RECT , Size(3 , 3)));
    threshold(dilated , dilated , 100 , 255 , THRESH_BINARY_INV|THRESH_OTSU);

    // adaptive 
    adaptiveThreshold(only_piano , adaptive , 255 , ADAPTIVE_THRESH_GAUSSIAN_C , THRESH_BINARY , 11 , 5);

    // or --> noise removed
    Mat noise_removed = adaptive|dilated;
    erode(noise_removed , noise_removed , Mat() , Point(-1 , -1) , 2);

    std::vector<std::vector<Point>>piano_contours;

    Mat outline_color;
    cvtColor(noise_removed , outline_color , COLOR_GRAY2BGR);
    RNG rng((unsigned int)time(0));
    // floodFill(canny , );
    findContours(noise_removed , piano_contours , RETR_TREE , CHAIN_APPROX_SIMPLE);
    namedWindow("canny_colored" , WINDOW_NORMAL);
    for(int i = 0; i < piano_contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        if(arcLength(piano_contours[i] , false) < 100) continue;

        drawContours(outline_color , piano_contours , i , color , -1);
        // std::vector<Vec2f>hull;
        // convexHull(canny_contours[i] , hull , true , true);
        imshow("canny_colored" , outline_color);
    }

    imshow("only_piano" , only_piano);
    imshow("noise_removed" , noise_removed);
    imshow("adaptive" , adaptive);
    imshow("dilated" , dilated);
    imshow("template" , template_piano);
//    imshow("inverted_piano" , adaptive);

    while(1) { if(waitKey(0) == 27) break; }

    return 0;
}
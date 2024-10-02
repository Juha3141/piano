// 1. create the dilated image, to mask the black keys

    dilate(piano_image , dilated , Mat() , Point(-1 , -1) , 3);
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
        Scalar color = Scalar(rng.uniform(0 , 255) , rng.uniform(0 , 255) , rng.uniform(0 , 255));
        drawContours(first_contour , piano_contours_1 , i , color , 1);
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
    imshow("first_contour" , first_contour);
    imshow("noise_removed" , noise_removed);
    imshow("adaptive" , adaptive);
    imshow("adaptive|dilated" , adaptive|dilated);
    imshow("colored" , outline_color);
    imwrite("finally.jpg" , outline_color);
#endif
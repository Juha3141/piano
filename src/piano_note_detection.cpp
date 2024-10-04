#include <piano_note_detection.hpp>

using namespace cv;
using namespace xfeatures2d;

void piano::detect_missing_white_keys(struct piano_keys_info &keys_info) {
#ifdef DEBUG
    Mat new_constructed_img;
    cvtColor(keys_info.piano_image , new_constructed_img , COLOR_GRAY2BGR);
#endif
    bool new_added = false;
    // remove the outlier by comparing distance from the best-fit line
    std::cout << "median distance : " << keys_info.median_dist_between_keys << "\n";
    std::vector<std::pair<RotatedRect , int>> *vect = new std::vector<std::pair<RotatedRect , int>>;
    for(int i = 0; i < keys_info.dist_between_keys_list.size(); i++) {
        // compare with the nearby keys
        vect->push_back(std::pair<RotatedRect , int>(keys_info.keys_rectangle_list[i] , i));
        if(keys_info.dist_between_keys_list[i] >= keys_info.keys_rectangle_list[i].size.width*1.5) {
            std::cout << "missing keys found!, index = " << i << "\n";
            keys_info.separated_keys_rectangle_list.push_back(vect);

            vect = new std::vector<std::pair<RotatedRect , int>>;
        }
    }
    keys_info.separated_keys_rectangle_list.push_back(vect);
    for(std::vector<std::pair<RotatedRect , int>>*vect : keys_info.separated_keys_rectangle_list) {
        std::cout << "----- consecutive part -----\n";
        for(std::pair<RotatedRect , int>p : *vect) {
            std::cout << p.second << " , ";
        }
        std::cout << "\n";
    }

    if(!new_added) return;
    keys_info.dist_between_keys_list.clear();
    keys_info.cm_dist_from_bestfit_list.clear();
    write_keys_info(keys_info);
}

void piano::detect_missing_black_keys(struct piano_keys_info &black_keys_info , struct piano_keys_info &white_keys_info) {
    if(black_keys_info.dist_between_keys_list.size() == 0) {
        std::cout << "dist_between_keys_list empty!\n";
        return;
    }
    int missing_black_keys = 36-black_keys_info.keys_rectangle_list.size();
    int black_key_distance_list[] = {
        2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,2,
        1,2,1,1,
    };
    int black_key_distance_cumulative[35] = {0 , };
    for(int i = 1; i < 35; i++) {
        black_key_distance_cumulative[i] = black_key_distance_cumulative[i-1]+black_key_distance_list[i];
    }
    
    Mat image = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC3);

    RNG rng((unsigned int)time(0));
    std::vector<int>key_count_list;
    for(int i = 0; i < black_keys_info.keys_rectangle_list.size()-1; i++) {
        // compare with the nearby keys
        std::cout << "-- " << i << "--\n";
        std::cout << "distance  : " << black_keys_info.dist_between_keys_list[i] << "\n";
        
        double width = white_keys_info.median_key_width;//*cos(black_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
        
        std::vector<Point2f>pts;
        black_keys_info.keys_rectangle_list[i].points(pts);
        int current_x = (pts[2].x+pts[3].x)/2;
        int current_y = (pts[2].y+pts[3].y)/2;
        for(int j = 0; j < white_keys_info.keys_rectangle_list.size(); j++) {
            std::vector<Point>contour;
            rotated_rect_to_contour(white_keys_info.keys_rectangle_list[j] , contour);
            if(pointPolygonTest(contour , Point2f(current_x , current_y) , false) > 0) {
                width = white_keys_info.keys_rectangle_list[j].size.width;//*cos(black_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
                break;
            }
        }
        Scalar color = Scalar(rng.uniform(0 , 180) , rng.uniform(0 , 180) , rng.uniform(0 , 180));
        
        std::cout << "width   : " << width << "\n";
        int key_count = round((double)black_keys_info.dist_between_keys_list[i]/width);
        std::cout << "key_count : " << key_count << "\n";
        bool match = (key_count == black_key_distance_list[i]);
        std::cout << "match     : " << match << "\n";
        key_count_list.push_back(key_count);
    }

    for(int i = 0; i < key_count_list.size(); i++) {
        if(key_count_list[i] == black_key_distance_list[i]) continue;
        
        std::cout << "black key #" << i << "\n";
        
    }
//  imshow("blackpoint" , image);
}

/// @brief Detect the shapes of the white keys
/// @param white_keys_info 
/// @param black_keys_info 
void piano::detect_white_key_shapes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {
    Mat white_mat(white_keys_info.piano_image.size() , CV_8UC1);
    Mat black_mat(white_keys_info.piano_image.size() , CV_8UC1);

#define KEY_NO_BLACK    1
#define KEY_BLACK_LEFT  2
#define KEY_BLACK_RIGHT 3
#define KEY_BLACK_BOTH  4

    int white_key_index = 0;
    std::vector<int>error_index_list;
    for(RotatedRect white_rect : white_keys_info.keys_rectangle_list) {
        bool left = false;
        bool right = false;

        std::vector<std::vector<Point>>overlapping_contours;
        Point2f white_rect_pts[4];
        white_rect.points(white_rect_pts);
        std::vector<Point>left_side_contour = {white_rect_pts[0] , white_rect_pts[0] , white_rect_pts[1] , white_rect_pts[1]};
        std::vector<Point>right_side_contour = {white_rect_pts[2] , white_rect_pts[2] , white_rect_pts[3] , white_rect_pts[3]};
        for(RotatedRect black_rect : black_keys_info.keys_rectangle_list) {
            std::vector<Point>intersect;
            if(rotatedRectangleIntersection(white_rect , black_rect , intersect) == INTERSECT_NONE) continue;
            
            Mat overlap_check_1 = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC1);
            Mat overlap_check_2 = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC1);
            Mat overlap_check_3 = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC1);
            std::vector<Point>black_rect_contour;
            rotated_rect_to_contour(black_rect , black_rect_contour);
            drawContours(overlap_check_1 , std::vector<std::vector<Point>>({black_rect_contour}) , -1 , 0xff , -1);
            drawContours(overlap_check_2 , std::vector<std::vector<Point>>({left_side_contour}) , -1 , 0xff , -1);
            drawContours(overlap_check_3 , std::vector<std::vector<Point>>({right_side_contour}) , -1 , 0xff , -1);

            int overlap_left_check = countNonZero(overlap_check_1 & overlap_check_2);
            int overlap_right_check = countNonZero(overlap_check_1 & overlap_check_3);
            if(overlap_left_check >= 1 && overlap_right_check >= 1) {
                std::cout << "error!" << "\n";
                if(white_rect.center.x > black_rect.center.x) {
                    left = true;
                }
                else {
                    right = true;
                }
                error_index_list.push_back(white_key_index);
            }
            else {
                if(overlap_left_check >= 1) { left = true; }
                if(overlap_right_check >= 1) { right = true; }
            }
        }

        Scalar color;
        int key_type;
        switch((left << 1)|right) {
            case 0:    key_type = KEY_NO_BLACK;    break;
            case 0b01: key_type = KEY_BLACK_RIGHT; break; // right
            case 0b11: key_type = KEY_BLACK_BOTH;  break; // both
            case 0b10: key_type = KEY_BLACK_LEFT;  break; // left
        }
        white_keys_info.white_key_shapes.push_back(key_type);
        white_key_index++;
    }
    std::cout << "----- white_keys_info.white_key_shapes -----\n";
    for(int i = 0; i < white_keys_info.white_key_shapes.size(); i++) {
        std::cout << white_keys_info.white_key_shapes[i] << " , ";
    }
    std::cout << "\n";
}

void piano::detect_white_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {
    double mean_y = 0;
    /* C : 1
     * D : 3
     * E : 5
     * F : 6
     * G : 8
     * A : 10
     * B : 12
     */
    /*                                   0    1              2              3              4              5              6              7              8*/
    int key_shape_list[]              = {3,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 3,4,2,3,4,4,2, 1};
    int key_shape_list_notes_octave[] = {0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 5,5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7, 8};
    int key_shape_list_notes[] = {10,12, 1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    for(RotatedRect rr : white_keys_info.keys_rectangle_list) { mean_y += rr.center.y; }
    mean_y /= (double)white_keys_info.keys_rectangle_list.size();

    std::cout << "mean_y : " << mean_y << "\n";
    double current_x = 0 , current_y = mean_y;
    current_x = white_keys_info.keys_rectangle_list[0].center.x;
    std::cout << "------ detect_white_key_notes ------\n";
    double previous_width = white_keys_info.keys_rectangle_list[0].size.width*cos(white_keys_info.keys_rectangle_list[0].angle*M_PI/180.0f);
    int previous_hit_index = 0;
    double distance = 0;
    int current_offset = 0;
    while(1) {
        bool detected = false;
        int detected_index = 0;
        for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
            std::vector<Point>contour;
            rotated_rect_to_contour(white_keys_info.keys_rectangle_list[i] , contour);
            if(pointPolygonTest(contour , Point2f(current_x , current_y) , false) >= 0) {
                std::cout << "hit , index = " << i << "\n";
                current_x = white_keys_info.keys_rectangle_list[i].center.x;
                current_y = white_keys_info.keys_rectangle_list[i].center.y;
                previous_width = white_keys_info.keys_rectangle_list[i].size.width*cos(white_keys_info.keys_rectangle_list[i].angle*M_PI/180.0f);
                
                detected = true;
                detected_index = i;
                break;
            }
        }
        // find the closest thing from the current offset
        if(detected == true) {
            int min_distance = 0x7fffffff;
            int min_dist_piano_index = -1;
            for(int i = 0; i < 52; i++) {
                if(key_shape_list[i] == white_keys_info.white_key_shapes[detected_index]) {
                    std::cout << "match : " << i << ", note : " << number_to_note_string(key_shape_list_notes[i]) << "\n";
                    std::cout << "distance between the offset : " << abs(current_offset-i) << "\n";
                    if(min_distance > abs(current_offset-i)) {
                        min_dist_piano_index = i;
                        min_distance = abs(current_offset-i);
                    }
                }
            }
            std::cout << "correlating index : " << min_dist_piano_index << " , note : " << number_to_note_string(key_shape_list_notes[min_dist_piano_index]) << "\n";
            current_offset = min_dist_piano_index;
            white_keys_info.key_notes.push_back(std::pair<int , int>((key_shape_list_notes[min_dist_piano_index])|(key_shape_list_notes_octave[min_dist_piano_index] << 8) , min_dist_piano_index));
        }
        if(detected_index == white_keys_info.keys_rectangle_list.size()-1 || current_x >= white_keys_info.piano_image.size().width) { break; }
        std::cout << "x : " << current_x << "\n";
        current_x += previous_width;
        current_offset++;
    }
    return;
}

void piano::detect_black_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info) {
    for(int b = 0; b < black_keys_info.keys_rectangle_list.size(); b++) {
        // first : index , second : note
        std::vector<std::pair<int , int>>intersecting_white_keys;
        for(int w = 0; w < white_keys_info.keys_rectangle_list.size(); w++) {
            std::vector<Point>intersect;
            int intersection_type = rotatedRectangleIntersection(black_keys_info.keys_rectangle_list[b] , white_keys_info.keys_rectangle_list[w] , intersect);
            if(intersection_type == INTERSECT_PARTIAL) {
                intersecting_white_keys.push_back(std::pair<int , int>(w , white_keys_info.key_notes[w].first));
            }
        }
        int black_note = 0;
        std::sort(intersecting_white_keys.begin() , intersecting_white_keys.end() , [](const auto &a , const auto &b) { return a.second < b.second; });
        if(intersecting_white_keys.size() == 1) {
            std::cout << "  size = 1" << "\n";
            switch((NOTE(intersecting_white_keys[0].second))) {
                case PIANO_KEY_F: black_note = intersecting_white_keys[0].second-1; break;
                case PIANO_KEY_B: black_note = intersecting_white_keys[0].second+1; break;
            }
        }
        if(intersecting_white_keys.size() == 2) {
            std::cout << "  size = 2" << "\n";
            if(NOTE(intersecting_white_keys[0].second+1) == NOTE(intersecting_white_keys[1].second-1)) {
                black_note = intersecting_white_keys[0].second+1;
            }
        }
        std::cout << "black index : " << b << " , note : " << number_to_note_string(black_note) << "(" << black_note << ")\n";
        black_keys_info.key_notes.push_back(std::pair<int , int>(black_note , b));
    }
}

void piano::fill_missing_white_keys(struct piano_keys_info &white_keys_info) {
    int total_missing_count = 52-white_keys_info.keys_rectangle_list.size();
    int filled_count = 0;
    if(total_missing_count == 0) return;
    // first  : missing_key_index
    // second : missing_key_piano_index (0-52)
    // third  : missing_key_count
    std::vector<std::tuple<int , int , int>>missing_keys;
    int index_dist_between_keys = 0;
    for(int i = 1; i < white_keys_info.key_notes.size(); i++) {
        index_dist_between_keys = white_keys_info.key_notes[i].second-white_keys_info.key_notes[i-1].second;
        if(index_dist_between_keys != 1) {
            std::cout << "missing keys found! index : " << i-1 << " , count : " << index_dist_between_keys-1 << "\n";
            missing_keys.push_back(std::tuple<int , int , int>(i-1 , white_keys_info.key_notes[i-1].second , index_dist_between_keys-1));
        }
    }

    // first : index , second : number of missing keys
    if(missing_keys.size() == 0) return;
    std::cout << "missing keys : " << missing_keys.size() << "\n";
    // if(white_keys_info.keys_rectangle_list.size() == 52) return;
    int key_shape_list_notes[] = {10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    int key_shape_list_notes_octave[] = {0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 5,5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7, 8};

    Mat test_img = Mat::zeros(white_keys_info.piano_image.size() , CV_8UC3);
    std::vector<int>newly_created_index_list;
    for(std::tuple<int , int , int>missing_key : missing_keys) {
        bool exit = false;
        int missing_key_index       = std::get<0>(missing_key);
        int missing_key_piano_index = std::get<1>(missing_key);
        int missing_key_count       = std::get<2>(missing_key);

        Point p1 = white_keys_info.keys_rectangle_list[missing_key_index].center;
        Point p2 = white_keys_info.keys_rectangle_list[missing_key_index+1].center;
        double angle1 = white_keys_info.keys_rectangle_list[missing_key_index].angle;
        double angle2 = white_keys_info.keys_rectangle_list[missing_key_index+1].angle;

        filled_count += missing_key_count;
        if(filled_count > total_missing_count) {
            missing_key_count = (filled_count-total_missing_count);
            break;
        }
        // Internal division of the two points
        for(int j = 1; j < missing_key_count+1; j++) {
            int m = j , n = (missing_key_count+1)-j;
            // divided point
            Point p((m*p2.x+n*p1.x)/(m+n) , (m*p2.y+n*p1.y)/(m+n));
            circle(test_img , p , 2 , Scalar(0x00 , 0x00 , 0xff) , 1);
            // Construct a new rectangle. New rectangle has the same width and height as the neighboring rectangle
            // The angle of the new rectangle is the average of the two reference rectangle(ith and i+1th rectangles.)
            RotatedRect new_rect(p , Size(white_keys_info.keys_rectangle_list[missing_key_index].size.width , white_keys_info.keys_rectangle_list[missing_key_index].size.height) , (angle1+angle2)/2);
            white_keys_info.keys_rectangle_list.push_back(new_rect); // new ones are added

            newly_created_index_list.push_back(++missing_key_piano_index);
#ifdef DEBUG
            // draw the rectangles (why not?)
            std::vector<Point>rect_contour;
            rotated_rect_to_contour(new_rect , rect_contour);
            drawContours(test_img , std::vector<std::vector<Point>>({rect_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
#endif
        }
        if(exit) break;
    }
    std::sort(newly_created_index_list.begin() , newly_created_index_list.end());
    for(int i : newly_created_index_list) {
        white_keys_info.key_notes.insert(white_keys_info.key_notes.begin()+i , std::pair<int , int>(key_shape_list_notes[i] , i));
    }
    white_keys_info.dist_between_keys_list.clear();
    white_keys_info.cm_dist_from_bestfit_list.clear();
    write_keys_info(white_keys_info);

    imshow("newly_added" , test_img);
}

void piano::doublecheck_white_keys(struct piano_keys_info &white_keys_info) {
    std::vector<int>discrepencies;
    int key_shape_list_notes[] = {10,12, 1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    for(int i = 0; i < white_keys_info.key_notes.size(); i++) {
        if(NOTE(white_keys_info.key_notes[i].first) != key_shape_list_notes[i]) discrepencies.push_back(i);
    }
    std::cout << "white discrepencies : " << discrepencies.size() << "\n";
    if(discrepencies.size() == 0) return;
}

void piano::doublecheck_black_keys(struct piano_keys_info &black_keys_info) {
    int black_key_notes_template[] = {
        0x0A , 
        0x12 , 0x14 , 0x17 , 0x19 , 0x1B , 
        0x22 , 0x24 , 0x27 , 0x29 , 0x2B , 
        0x32 , 0x34 , 0x37 , 0x39 , 0x3B , 
        0x42 , 0x44 , 0x47 , 0x49 , 0x4B , 
        0x52 , 0x54 , 0x57 , 0x59 , 0x5B , 
        0x62 , 0x64 , 0x67 , 0x69 , 0x6B , 
        0x72 , 0x74 , 0x77 , 0x79 , 0x7B , 
    };
    std::vector<int>discrepencies;
    for(int i = 0; i < black_keys_info.key_notes.size(); i++) {
        if(black_keys_info.key_notes[i].first != black_key_notes_template[i]) discrepencies.push_back(i);
    }
    std::cout << "black discrepencies : " << discrepencies.size() << "\n";
    if(discrepencies.size() == 0) return;
    
}
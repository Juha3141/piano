#include <piano_detection.hpp>

using namespace cv;
using namespace xfeatures2d;

#define DEBUG

// shape of the keys
#define KEY_NO_BLACK    0
#define KEY_BLACK_LEFT  1
#define KEY_BLACK_RIGHT 2
#define KEY_BLACK_BOTH  3

void PianoInfo::create_white_adjusted_cm_list(void) {
    std::vector<RotatedRect>rotated_white_rectangles;
    std::vector<RotatedRect>rotated_black_rectangles;

    Mat demo_image;
    cvtColor(piano_image , demo_image , COLOR_GRAY2BGR);
    
    this->key_adjusted_cm_list.clear();
    // rotate the positions of the rectangles
    int average_white_key_cm_y = 0;
    for(int i = 0; i < this->white_keys_info.keys_rectangle_list.size(); i++) {
        RotatedRect rr(this->white_keys_info.keys_rectangle_list[i]);
        rr.center = rotational_matrix(rr.center , (-this->piano_bounding_rect.angle)*M_PI/180.0f , this->piano_bounding_rect.center);
        rr.angle -= this->piano_bounding_rect.angle;
        rotated_white_rectangles.push_back(rr);
        average_white_key_cm_y += rr.center.y;

        this->key_adjusted_cm_list.push_back(std::tuple<bool , Point , int , double>(true , rr.center , i , 0));
    
#ifdef DEBUG
        std::vector<Point>contour;
        rotated_rect_to_contour(rr , contour);
        drawContours(demo_image , std::vector<std::vector<Point>>({contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
        circle(demo_image , rr.center , 2 , Scalar(0xff , 0x00 , 0x00) , -1);
#endif
    }
    average_white_key_cm_y /= this->white_keys_info.keys_rectangle_list.size();

    for(int i = 0; i < this->black_keys_info.keys_rectangle_list.size(); i++) {
        RotatedRect rr(this->black_keys_info.keys_rectangle_list[i]);
        rr.center = rotational_matrix(rr.center , (-this->piano_bounding_rect.angle)*M_PI/180.0f , this->piano_bounding_rect.center);
        rr.angle -= this->piano_bounding_rect.angle;
        rotated_white_rectangles.push_back(rr);
        Point moved_point(rr.center.x-abs(rr.center.y-average_white_key_cm_y)*tan(rr.angle*M_PI/180.0f) , average_white_key_cm_y);

        this->key_adjusted_cm_list.push_back(std::tuple<bool , Point , int , double>(false , moved_point , i , 0));

#ifdef DEBUG
        std::vector<Point>contour;
        rotated_rect_to_contour(rr , contour);
        drawContours(demo_image , std::vector<std::vector<Point>>({contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
        circle(demo_image , rr.center , 2 , Scalar(0x00 , 0x00 , 0xff) , -1);
        circle(demo_image , moved_point , 2 , Scalar(0x00 , 0xff , 0xff) , -1);
#endif
    }
    std::sort(this->key_adjusted_cm_list.begin() , this->key_adjusted_cm_list.end() , 
        [](const std::tuple<bool , Point , int , double>&a , const std::tuple<bool , Point , int , double>&b) {
            return (std::get<1>(a).x < std::get<1>(b).x); 
        });
    for(int i = 0; i < this->key_adjusted_cm_list.size()-1; i++) {
        double distance = euclidean_distance(std::get<1>(this->key_adjusted_cm_list[i]) , std::get<1>(this->key_adjusted_cm_list[i+1]));
        std::get<3>(this->key_adjusted_cm_list[i]) = distance;
    }
#ifdef DEBUG
    imshow("create_white_adjusted_cm_list" , demo_image);
#endif
}

/// @brief Detect the shapes of the white keys
void PianoInfo::detect_white_key_shapes(void) {
    if(this->key_adjusted_cm_list.size() == 0) create_white_adjusted_cm_list();

    bool missing_spots_bitmap[this->white_keys_info.keys_rectangle_list.size()];
    memset(missing_spots_bitmap , false , this->white_keys_info.keys_rectangle_list.size()*sizeof(bool));
    for(std::pair<int , int>p : this->white_keys_info.missing_key_spots_list) {
        missing_spots_bitmap[p.first] = true;
        missing_spots_bitmap[p.second] = true;
    }

    white_keys_info.white_key_shapes.resize(white_keys_info.keys_rectangle_list.size());
    for(int i = 0; i < this->key_adjusted_cm_list.size(); i++) {
        if(std::get<0>(this->key_adjusted_cm_list[i]) == false) continue;
        bool left = false;
        bool right = false;

        int white_index = std::get<2>(this->key_adjusted_cm_list[i]);
        // check where are the black keys
        if(i > 0 && std::get<0>(this->key_adjusted_cm_list[i-1]) == false) left = true;
        if(i < this->key_adjusted_cm_list.size()-1 && std::get<0>(this->key_adjusted_cm_list[i+1]) == false) right = true;

        Mat testing_ground_1 , testing_ground_2;
        // if the white key is detected to be missing, double-check with more robust method
        if(missing_spots_bitmap[white_index]) {
            std::cout << "missing spots... white #" << white_index << "\n";
            RotatedRect black_left_rr , black_right_rr;
            std::vector<Point>white_rr_contour;
            std::vector<Point>black_left_rr_contour , black_right_rr_contour;

            testing_ground_1 = Mat::zeros(this->piano_image.size() , CV_8UC1);
            testing_ground_2 = Mat::zeros(this->piano_image.size() , CV_8UC1);
            rotated_rect_to_contour(this->white_keys_info.keys_rectangle_list[std::get<2>(this->key_adjusted_cm_list[i])] , white_rr_contour);
            if(left) {
                black_left_rr = this->black_keys_info.keys_rectangle_list[std::get<2>(this->key_adjusted_cm_list[i-1])];
                rotated_rect_to_contour(black_left_rr , black_left_rr_contour);

                drawContours(testing_ground_1 , std::vector<std::vector<Point>>({white_rr_contour}) , -1 , Scalar(0xff) , -1);
                drawContours(testing_ground_2 , std::vector<std::vector<Point>>({black_left_rr_contour}) , -1 , Scalar(0xff) , -1);
                Mat overlapping = testing_ground_1 & testing_ground_2;
                if(countNonZero(overlapping) == 0) {
                    left = false;
                    std::cout << "left false detection, white #" << std::get<2>(this->key_adjusted_cm_list[i]) << "\n";
                }
                overlapping.release();
            }
            testing_ground_1 = Mat::zeros(this->piano_image.size() , CV_8UC1);
            testing_ground_2 = Mat::zeros(this->piano_image.size() , CV_8UC1);
            if(right) {
                black_right_rr = this->black_keys_info.keys_rectangle_list[std::get<2>(this->key_adjusted_cm_list[i+1])];
                rotated_rect_to_contour(black_right_rr , black_right_rr_contour);

                drawContours(testing_ground_1 , std::vector<std::vector<Point>>({white_rr_contour}) , -1 , Scalar(0xff) , -1);
                drawContours(testing_ground_2 , std::vector<std::vector<Point>>({black_right_rr_contour}) , -1 , Scalar(0xff) , -1);
                
                Mat overlapping = testing_ground_1 & testing_ground_2;
                if(countNonZero(overlapping) == 0) {
                    right = false;
                    std::cout << "right false detection, white #" << std::get<2>(this->key_adjusted_cm_list[i]) << "\n";
                }
                overlapping.release();
            }
            testing_ground_1.release();
            testing_ground_2.release();
        }

        // if the image is flipped, the key shape must also be flipped
        if(this->flipped) std::swap(left , right);

        int key_shape = (right << 1|left);
        std::cout << "White key #" << std::get<2>(this->key_adjusted_cm_list[i]) << " : " << key_shape << "\n";
        white_keys_info.white_key_shapes[std::get<2>(this->key_adjusted_cm_list[i])] = key_shape;
    }
}

void PianoInfo::detect_white_key_notes(void) {
    /* C : 1
     * D : 3
     * E : 5
     * F : 6
     * G : 8
     * A : 10
     * B : 12
     */
    /*                                   0    1              2              3              4              5              6              7              8*/
    int key_shape_list[]              = {2, 1,  2,3,1,2,3,3, 1,  2,3,1,2,3,3, 1,  2,3,1,2,3,3, 1,  2,3,1,2,3,3, 1,  2,3,1,2,3,3, 1,  2,3,1,2,3,3, 1,  2,3,1,2,3,3, 1,  0};
    int key_shape_list_notes_octave[] = {0, 0,  1,1,1,1,1,1, 1,  2,2,2,2,2,2, 2,  3,3,3,3,3,3, 3,  4,4,4,4,4,4, 4,  5,5,5,5,5,5, 5,  6,6,6,6,6,6, 6,  7,7,7,7,7,7, 7,  8};
    int key_shape_list_notes[] =        {10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1};
    
    std::cout << "------ detect_white_key_notes ------\n";
    int current_offset = 0;
    if(this->flipped) {
        std::reverse(std::begin(key_shape_list) , std::end(key_shape_list));
        std::reverse(std::begin(key_shape_list_notes_octave) , std::end(key_shape_list_notes_octave));
        std::reverse(std::begin(key_shape_list_notes) , std::end(key_shape_list_notes));
    }

    for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
        double distance = 0;
        if(i != white_keys_info.keys_rectangle_list.size()-1) distance = euclidean_distance(white_keys_info.keys_rectangle_list[i].center , white_keys_info.keys_rectangle_list[i+1].center);
        double width = white_keys_info.keys_rectangle_list[i].size.width;
        std::cout << "current_offset : " << current_offset << "\n";
        
        std::cout << "white_keys_info.white_key_shapes[" << i << "] = " << white_keys_info.white_key_shapes[i] << "\n";
        int min_distance = 0x7fffffff;
        int min_dist_piano_index = -1;
        for(int j = 0; j < sizeof(key_shape_list)/sizeof(int); j++) {
            if(key_shape_list[j] == white_keys_info.white_key_shapes[i]) {
                if(min_distance >= abs(current_offset-j)) {
                    min_dist_piano_index = j;
                    min_distance = abs(current_offset-j);
                }
            }
        }
        if(min_dist_piano_index == -1) std::cout << "error!\n";

        int note = key_shape_list_notes[min_dist_piano_index];
        int octave = key_shape_list_notes_octave[min_dist_piano_index];
        std::cout << i << " : " << number_to_note_string(key_shape_list_notes[min_dist_piano_index]) << "(Octave : " << key_shape_list_notes_octave[min_dist_piano_index] << ")\n"; 
        white_keys_info.key_notes.push_back(std::pair<int , int>((octave << 4)|(note & 0x0f) , min_dist_piano_index));
        std::cout << "distance = " << distance << "\n";
        std::cout << "width    = " << width << "\n";
        
        // if distance is really small --> just increment the current offset
        if(distance < width) current_offset++; 
        else current_offset += floor(distance/width);
    }
    return;
}

void PianoInfo::detect_black_key_notes(void) {
    if(key_adjusted_cm_list.size() == 0) create_white_adjusted_cm_list();
    this->black_keys_info.key_notes.resize(this->black_keys_info.keys_rectangle_list.size());
    for(int i = 0; i < this->key_adjusted_cm_list.size(); i++) {
        if(std::get<0>(this->key_adjusted_cm_list[i]) == true) continue;
        std::cout << "" << "\n";

        int black_index = std::get<2>(this->key_adjusted_cm_list[i]);
        int white_index_left  = std::get<2>(this->key_adjusted_cm_list[i-1]);
        int white_index_right = std::get<2>(this->key_adjusted_cm_list[i+1]);
        int white_index_left_note = this->white_keys_info.key_notes[white_index_left].first;
        int white_index_right_note = this->white_keys_info.key_notes[white_index_right].first;
        std::cout << "Left side note = " << number_to_note_string(this->white_keys_info.key_notes[white_index_left].first) << "\n";
        std::cout << "Right side note = " << number_to_note_string(this->white_keys_info.key_notes[white_index_right].first) << "\n";
        int note = (NOTE(white_index_left_note)+NOTE(white_index_right_note))/2;
        note |= (OCTAVE(white_index_left_note) << 4);

        std::cout << "Note = " << number_to_note_string(note) << "\n";
        this->black_keys_info.key_notes[black_index].first = note;
        this->black_keys_info.key_notes[black_index].second = 0;
    }
}

static int note_to_piano_index(int octave_and_note) {
    const int numerical_note_to_note_index[] = {
        -1 , 0 , // 1
        -1 , 1 , // 3
        -1 , 2 , // 5
         3 ,     // 6
        -1 , 4 , // 8
        -1 , 5 , // 10
        -1 , 6   // 12
    };
    const int octave_to_note_index[] = {
        0 , 2 , 9 , 16 , 23 , 30 , 37 , 44 , 51
    };
    int note = NOTE(octave_and_note);
    int octave = OCTAVE(octave_and_note);
    int note_index = numerical_note_to_note_index[note];
    if(octave == 0) { note_index -= 5; }
    if(note_index < 0) { note_index = 0; }
    return octave_to_note_index[octave]+note_index;
}

void PianoInfo::adjust_wrong_white_notes(void) {
    int key_shape_list_notes[] =        {10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1,3,5,6,8,10,12, 1};
    std::vector<std::vector<std::pair<int , int>>>consecutive_key_notes_list;
    
    int index = 0;
    consecutive_key_notes_list.resize(white_keys_info.missing_key_spots_list.size()+1);
    int k;
    for(k = 0; k < white_keys_info.missing_key_spots_list.size(); k++) {
        std::cout << "consecutive part : " << index << " ~ " << white_keys_info.missing_key_spots_list[k].first << "\n";
        for(int i = index; i <= white_keys_info.missing_key_spots_list[k].first; i++) {
            consecutive_key_notes_list[k].push_back(std::pair<int , int>(white_keys_info.key_notes[i].first , i));
        }
        index = white_keys_info.missing_key_spots_list[k].second;
    }
    for(int i = index; i < white_keys_info.keys_rectangle_list.size(); i++) {
        consecutive_key_notes_list[k].push_back(std::pair<int , int>(white_keys_info.key_notes[i].first , i));
    }
    std::cout << "consecutive part : " << index << " ~ " << white_keys_info.keys_rectangle_list.size()-1 << "\n";

    // if flipped, reverse the note list
    if(this->flipped) std::reverse(std::begin(key_shape_list_notes) , std::end(key_shape_list_notes));
    // very inefficient but will do the job
    for(std::vector<std::pair<int , int>>consecutive : consecutive_key_notes_list) {
        std::cout << "consecutive part ---- \n";
        int max_matching_index = 0;
        int max_matching = 0;
        for(std::pair<int , int>p : consecutive) {
            std::cout << number_to_note_string(p.first) << "(" << OCTAVE(p.first) << "),";
        }
        std::cout << "\b \n";
        for(int i = 0; i <= (sizeof(key_shape_list_notes)/sizeof(int))-consecutive.size(); i++) {
            int matching = 0;
            for(int j = 0; j < consecutive.size(); j++) {
                if(NOTE(consecutive[j].first) == key_shape_list_notes[i+j]) matching++;
            }
            if(matching > max_matching) {
                max_matching_index = i;
                max_matching = matching;
            }
        }
        std::cout << "max_matching_index = " << max_matching_index << "\n";
        std::cout << "max_matching =       " << max_matching << "(" << consecutive.size()-max_matching << " discrepencies)\n";

        if(max_matching == 0) continue;

        // adjust discrepencies
        for(int i = 0; i < consecutive.size(); i++) {
            white_keys_info.key_notes[consecutive[i].second].first = 
                (white_keys_info.key_notes[consecutive[i].second].first & 0xf0)|(key_shape_list_notes[max_matching_index+i] & 0x0f);
            white_keys_info.key_notes[consecutive[i].second].second = note_to_piano_index(white_keys_info.key_notes[consecutive[i].second].first);
        }
    }
}

typedef struct missing_key_info_s {
    int missing_key_index_1 , missing_key_index_2;
    int missing_key_piano_index_1 , missing_key_piano_index_2;

    int missing_key_count;
}missing_key_info_t;

static void detect_missing_keys_from_notes_list(white_piano_keys_info_t &white_keys_info , std::vector<missing_key_info_t>&missing_keys) {
    int index_dist_between_keys = 0;
    for(int i = 1; i < white_keys_info.key_notes.size(); i++) {
        index_dist_between_keys = white_keys_info.key_notes[i].second-white_keys_info.key_notes[i-1].second;
        if(index_dist_between_keys != 1) {
            std::cout << "missing keys found! index : " << i-1 << " , count : " << index_dist_between_keys-1 << "\n";
            missing_key_info_t mkeyinfo;
            mkeyinfo.missing_key_index_1 = i-1;
            mkeyinfo.missing_key_index_2 = i;
            mkeyinfo.missing_key_piano_index_1 = white_keys_info.key_notes[i-1].second;
            mkeyinfo.missing_key_piano_index_2 = white_keys_info.key_notes[i].second;
            mkeyinfo.missing_key_count = index_dist_between_keys-1;

            missing_keys.push_back(mkeyinfo);
            std::cout << "missing key index based on 0-52 scale : " << white_keys_info.key_notes[i-1].second << "\n";
        }
    }
}

static void detect_missing_keys_from_notes_list_flipped(white_piano_keys_info_t &white_keys_info , std::vector<missing_key_info_t>&missing_keys) {
    int index_dist_between_keys = 0;
    for(int i = white_keys_info.key_notes.size()-2; i >= 0; i--) {
        index_dist_between_keys = white_keys_info.key_notes[i].second-white_keys_info.key_notes[i+1].second;
        if(index_dist_between_keys != 1) {
            std::cout << "missing keys found! index : " << i+1 << " , count : " << index_dist_between_keys-1 << "\n";
            missing_key_info_t mkeyinfo;
            mkeyinfo.missing_key_index_1 = i;
            mkeyinfo.missing_key_index_2 = i+1;
            mkeyinfo.missing_key_piano_index_1 = white_keys_info.key_notes[i].second;
            mkeyinfo.missing_key_piano_index_2 = white_keys_info.key_notes[i+1].second;
            mkeyinfo.missing_key_count = index_dist_between_keys-1;

            missing_keys.push_back(mkeyinfo);
            std::cout << "missing key index based on 0-52 scale : " << white_keys_info.key_notes[i+1].second << "\n";
        }
    }
}

void PianoInfo::fill_missing_white_keys(void) {
    int total_missing_count = 52-white_keys_info.keys_rectangle_list.size();
    int filled_count = 0;
    if(total_missing_count == 0) return;
    // first  : two indexes of missing keys
    // second : missing_key_piano_index (0-52)
    // third  : missing_key_count
    std::vector<missing_key_info_t>missing_keys;

    if(flipped) detect_missing_keys_from_notes_list_flipped(white_keys_info , missing_keys);
    else        detect_missing_keys_from_notes_list(white_keys_info , missing_keys);
    // first : index , second : number of missing keys
    if(missing_keys.size() == 0) return;
    std::sort(missing_keys.begin() , missing_keys.end() , [](const missing_key_info_t &a , const missing_key_info_t &b) {
        return (a.missing_key_index_1 < b.missing_key_index_1);
    });

#ifdef DEBUG
    std::cout << "missing keys : " << missing_keys.size() << "\n";
    std::cout << "total_missing_count = " << total_missing_count << "\n";
    Mat test_img = Mat::zeros(piano_image.size() , CV_8UC3);
    Mat test_img_2;
    cvtColor(piano_image , test_img_2 , COLOR_GRAY2BGR);
#endif
    
    int key_shape_list_notes[] = {10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    int key_shape_list_notes_octave[] = {0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 5,5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7, 8};

    int k = 0;
    for(int i = 0; i < missing_keys.size(); i++) {
        int missing_key_1             = missing_keys[i].missing_key_index_1;
        int missing_key_2             = missing_keys[i].missing_key_index_2;
        int missing_key_piano_index_1 = missing_keys[i].missing_key_piano_index_1;
        int missing_key_piano_index_2 = missing_keys[i].missing_key_piano_index_2;
        int missing_key_count         = missing_keys[i].missing_key_count;
#ifdef DEBUG
        std::cout << "missing_key_note_1 = " << number_to_note_string(key_shape_list_notes[missing_keys[i].missing_key_piano_index_1]) << "\n";
        std::cout << "missing_key_note_2 = " << number_to_note_string(key_shape_list_notes[missing_keys[i].missing_key_piano_index_2]) << "\n";
#endif

        std::vector<std::pair<int , int>>new_keys_list;
        for(int j = std::min(missing_key_piano_index_1 , missing_key_piano_index_2)+1;
                j < std::max(missing_key_piano_index_1 , missing_key_piano_index_2); j++) {
            new_keys_list.push_back(std::pair<int , int>(key_shape_list_notes[j] , j));
            int note = key_shape_list_notes[j] , octave = key_shape_list_notes_octave[j];
#ifdef DEBUG
            std::cout << number_to_note_string(note) << "(" << octave << "),";
#endif
        }
        if(this->flipped) std::reverse(new_keys_list.begin() , new_keys_list.end());
        
#ifdef DEBUG
        std::cout << "\n";
        std::cout << "missing_key_1 = " << missing_key_1 << "\n";
        std::cout << "missing_key_2 = " << missing_key_2 << "\n";
        std::vector<Point>c1 , c2;
        rotated_rect_to_contour(this->white_keys_info.keys_rectangle_list[missing_key_1] , c1);
        rotated_rect_to_contour(this->white_keys_info.keys_rectangle_list[missing_key_2] , c2);
        drawContours(test_img_2 , std::vector<std::vector<Point>>({c1}) , -1 , Scalar(0xff , 0x00 , 0x00));
        drawContours(test_img_2 , std::vector<std::vector<Point>>({c2}) , -1 , Scalar(0xff , 0xff , 0x00));
        putText(test_img_2 , "1" , this->white_keys_info.keys_rectangle_list[missing_key_1].center , FONT_HERSHEY_SIMPLEX , 0.5 , Scalar(0xff , 0x00 , 0x00) , 2);
        putText(test_img_2 , "2" , this->white_keys_info.keys_rectangle_list[missing_key_2].center , FONT_HERSHEY_SIMPLEX , 0.5 , Scalar(0xff , 0xff , 0x00) , 2);
#endif

        auto iter = white_keys_info.key_notes.begin()+std::max(missing_key_1 , missing_key_2)+k;
        white_keys_info.key_notes.insert(iter , new_keys_list.begin() , new_keys_list.end());

        std::cout << "\n";
        k += missing_key_count;
    }

    for(auto mkeyinfo : missing_keys) {
        bool exit = false;
        int missing_key_index_1 = mkeyinfo.missing_key_index_1;
        int missing_key_index_2 = mkeyinfo.missing_key_index_2;
        int missing_key_count = mkeyinfo.missing_key_count;

        std::cout << "missing_key_count = " << missing_key_count << "\n";
        filled_count += missing_key_count;
        if(filled_count > total_missing_count) {
            missing_key_count = total_missing_count;
            exit = true;
        }
        // index_1 : more located on "left side", index_2 : located on "right side"
        if(white_keys_info.keys_rectangle_list[missing_key_index_1].center.x > white_keys_info.keys_rectangle_list[missing_key_index_2].center.x) {
            std::swap(missing_key_index_1 , missing_key_index_2);
        }

        Point2f pts_1[4] , pts_2[4];
        white_keys_info.keys_rectangle_list[missing_key_index_1].points(pts_1);
        white_keys_info.keys_rectangle_list[missing_key_index_2].points(pts_2);

        Point p1((pts_1[2].x+pts_1[3].x)/2 , (pts_1[2].y+pts_1[3].y)/2);
        Point p2((pts_2[0].x+pts_2[1].x)/2 , (pts_2[0].y+pts_2[1].y)/2);
        double angle1 = white_keys_info.keys_rectangle_list[missing_key_index_1].angle , angle2 = white_keys_info.keys_rectangle_list[missing_key_index_2].angle;
        double delta_angle = (angle2-angle1)/(missing_key_count+1);

        double angle = angle1;
        double width = white_keys_info.keys_rectangle_list[missing_key_index_1].size.width;
        double delta_width = (white_keys_info.keys_rectangle_list[missing_key_index_2].size.width-white_keys_info.keys_rectangle_list[missing_key_index_1].size.width)/(missing_key_count+1);
        // Internal division of the two points
        std::cout << "delta_angle = " << delta_angle << "\n";
        std::cout << "delta_width = " << delta_width << "\n";
        RotatedRect latest_rect(white_keys_info.keys_rectangle_list[missing_key_index_1]);
        for(int j = 1; j <= missing_key_count-1; j++) {
            int m = j , n = (missing_key_count)-j;
            // divided point
            Point p_right((m*p2.x+n*p1.x)/(m+n) , (m*p2.y+n*p1.y)/(m+n));
            Point p_cm(p_right.x-((width/2)*cos(angle*M_PI/180.0f)) , p_right.y+((width/2)*sin(angle*M_PI/180.0f)));
            std::cout << "p_right = " << p_right << "\n";
            std::cout << "p_cm    = " << p_cm << "\n";
            circle(test_img , p_right , 2 , Scalar(0xff , 0x00 , 0x00) , 1);
            circle(test_img , p_cm , 2 , Scalar(0x00 , 0x00 , 0xff) , 1);
            // Construct a new rectangle. New rectangle has the same width and height as the neighboring rectangle
            // The angle of the new rectangle is the average of the two reference rectangle(ith and i+1th rectangles.)
            RotatedRect new_rect(p_cm , Size(width , white_keys_info.keys_rectangle_list[missing_key_index_1].size.height) , angle);
            latest_rect = new_rect;
            white_keys_info.keys_rectangle_list.push_back(new_rect); // new ones are added
#ifdef DEBUG
            // draw the rectangles (why not?)
            std::vector<Point>rect_contour;
            rotated_rect_to_contour(new_rect , rect_contour);
            drawContours(test_img , std::vector<std::vector<Point>>({rect_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
#endif
            angle += delta_angle;
            width += delta_width;
        }
        // one additional key
        Point2f latest_rect_pts[4];
        latest_rect.points(latest_rect_pts);
        Point p_final((latest_rect_pts[2].x+latest_rect_pts[3].x)/2 , (latest_rect_pts[2].y+latest_rect_pts[3].y)/2);
        Point final_new_rect_cm((p_final.x+p2.x)/2 , (p_final.y+p2.y)/2);

        std::cout << "p_final           = " << p_final << "\n";
        std::cout << "p2           = " << p2 << "\n";
        std::cout << "final_new_rect_cm = " << final_new_rect_cm << "\n";
        RotatedRect final_new_rect(final_new_rect_cm , Size(width , white_keys_info.keys_rectangle_list[missing_key_index_1].size.height) , angle);
        white_keys_info.keys_rectangle_list.push_back(final_new_rect);
#ifdef DEBUG
        std::vector<Point>rect_contour;
        rotated_rect_to_contour(final_new_rect , rect_contour);
        circle(test_img , final_new_rect_cm , 2 , Scalar(0x00 , 0x00 , 0xff) , 1);
        drawContours(test_img , std::vector<std::vector<Point>>({rect_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
#endif

        if(exit) break;
    }
    std::sort(white_keys_info.keys_rectangle_list.begin() , white_keys_info.keys_rectangle_list.end() , [](const RotatedRect &a,const RotatedRect &b) {
        return a.center.x < b.center.x;
    });
    white_keys_info.dist_between_keys_list.clear();
    white_keys_info.missing_key_count_list.clear();
    white_keys_info.missing_key_spots_list.clear();
    white_keys_info.keys_rectangle_pivot.clear();

    this->key_adjusted_cm_list.clear();

    write_keys_info(white_keys_info);
    write_pivot_info(*this , white);

#ifdef DEBUG
    imshow("newly_added" , test_img);
    imshow("range" , test_img_2);
#endif
}

void PianoInfo::elongate_white_keys(void) {
    for(int i = 0; i < white_keys_info.keys_rectangle_list.size(); i++) {
        double old_height = white_keys_info.keys_rectangle_list[i].size.height;
        adjust_rotated_rect_height(white_keys_info.keys_rectangle_list[i] , old_height*1.1 , flipped);
    }
}
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

    // rotate the positions of the rectangles
    int average_white_key_cm_y = 0;
    for(int i = 0; i < this->white_keys_info.keys_rectangle_list.size(); i++) {
        RotatedRect rr(this->white_keys_info.keys_rectangle_list[i]);
        rr.center = rotational_matrix(rr.center , (-this->piano_bounding_rect.angle)*M_PI/180.0f , this->piano_bounding_rect.center);
        rr.angle -= this->piano_bounding_rect.angle;
        rotated_white_rectangles.push_back(rr);
        average_white_key_cm_y += rr.center.y;

        this->key_adjusted_cm_list.push_back(std::tuple<bool , Point , int>(true , rr.center , i));
    
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

        this->key_adjusted_cm_list.push_back(std::tuple<bool , Point , int>(false , moved_point , i));

#ifdef DEBUG
        std::vector<Point>contour;
        rotated_rect_to_contour(rr , contour);
        drawContours(demo_image , std::vector<std::vector<Point>>({contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
        circle(demo_image , rr.center , 2 , Scalar(0x00 , 0x00 , 0xff) , -1);
        circle(demo_image , moved_point , 2 , Scalar(0x00 , 0xff , 0xff) , -1);
#endif
    }
    std::sort(this->key_adjusted_cm_list.begin() , this->key_adjusted_cm_list.end() , 
        [](const std::tuple<bool , Point , int>&a , const std::tuple<bool , Point , int>&b) {
            return (std::get<1>(a).x < std::get<1>(b).x); 
        });
#ifdef DEBUG
    imshow("create_white_adjusted_cm_list" , demo_image);
#endif
}

/// @brief Detect the shapes of the white keys
void PianoInfo::detect_white_key_shapes(void) {
    if(this->key_adjusted_cm_list.size() == 0) create_white_adjusted_cm_list();

    std::vector<int>number_of_black_between_white;
    white_keys_info.white_key_shapes.resize(white_keys_info.keys_rectangle_list.size());
    for(int i = 0; i < this->key_adjusted_cm_list.size()-1; i++) {
        if(std::get<0>(this->key_adjusted_cm_list[i]) == false) continue;
        bool left = false;
        bool right = false;
        // check where are the black keys
        if(i > 0 && std::get<0>(this->key_adjusted_cm_list[i-1]) == false) left = true;
        if(i < this->key_adjusted_cm_list.size()-1 && std::get<0>(this->key_adjusted_cm_list[i+1]) == false) right = true;

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

void PianoInfo::fill_missing_white_keys(void) {
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
            std::cout << "missing key index based on 0-52 scale : " << white_keys_info.key_notes[i-1].second << "\n";
        }
    }

    // first : index , second : number of missing keys
    if(missing_keys.size() == 0) return;
    std::cout << "missing keys : " << missing_keys.size() << "\n";
    // if(white_keys_info.keys_rectangle_list.size() == 52) return;
    int key_shape_list_notes[] = {10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1,3,5,6,8,10,12,1};
    int key_shape_list_notes_octave[] = {0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3, 4,4,4,4,4,4,4, 5,5,5,5,5,5,5, 6,6,6,6,6,6,6, 7,7,7,7,7,7,7, 8};

    Mat test_img = Mat::zeros(piano_image.size() , CV_8UC3);

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
#ifdef DEBUG
            // draw the rectangles (why not?)
            std::vector<Point>rect_contour;
            rotated_rect_to_contour(new_rect , rect_contour);
            drawContours(test_img , std::vector<std::vector<Point>>({rect_contour}) , -1 , Scalar(0x00 , 0xff , 0x00) , 1);
#endif
        }
        if(exit) break;
    }
    std::sort(white_keys_info.keys_rectangle_list.begin() , white_keys_info.keys_rectangle_list.end() , [](const RotatedRect &a,const RotatedRect &b) {
        return a.center.x < b.center.x;
    });
    int k = 0;
    for(int i = 0; i < missing_keys.size(); i++) {
        int missing_key_index       = std::get<0>(missing_keys[i]);
        int missing_key_piano_index = std::get<1>(missing_keys[i]);
        int missing_key_count       = std::get<2>(missing_keys[i]);
        for(int j = 1; j <= missing_key_count; j++) {
            int note = (key_shape_list_notes_octave[missing_key_piano_index+j] << 4)|(key_shape_list_notes[missing_key_piano_index+j] & 0x0f);
            white_keys_info.key_notes.insert(white_keys_info.key_notes.begin()+k+missing_key_index+j , std::pair<int , int>(note , missing_key_piano_index+i));
            k++;
        }
    }
    white_keys_info.dist_between_keys_list.clear();
    white_keys_info.missing_key_count_list.clear();
    white_keys_info.missing_key_spots_list.clear();
    white_keys_info.keys_rectangle_pivot.clear();

    this->key_adjusted_cm_list.clear();

    write_keys_info(white_keys_info);
    write_pivot_info(*this , white);
    imshow("newly_added" , test_img);
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

    if(this->flipped) std::reverse(std::begin(key_shape_list_notes) , std::end(key_shape_list_notes));
    // very inefficient but will do the job
    for(std::vector<std::pair<int , int>>consecutive : consecutive_key_notes_list) {
        std::cout << "consecutive part ---- \n";
        int max_matching_index = 0;
        int max_matching = 0;
        for(std::pair<int , int>p : consecutive) {
            std::cout << number_to_note_string(p.first) << ",";
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
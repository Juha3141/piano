#include <hand_detection_agent.hpp>
#include <string>

// keys for the shared memory
#define SHM_INFOMEM_KEY 3141591
#define SHM_IMGMEM_KEY  3141592

#define SHM_INFOMEM_SIZE 4096
#define SHM_IMGMEM_SIZE  1920*1080*3

using namespace cv;

bool hand_detection::initialize_agent(const char *video_device) {
    GlobalHandDetectionSystem *global_hand_info = GlobalHandDetectionSystem::get_self();
    strcpy(global_hand_info->video_device , video_device);

    int shmid_imgmem = shmget(SHM_IMGMEM_KEY , SHM_IMGMEM_SIZE , IPC_CREAT|0666);
    int shmid_infomem = shmget(SHM_INFOMEM_KEY , SHM_INFOMEM_SIZE , IPC_CREAT|0666);
    
    global_hand_info->shm_addr_imgmem = shmat(shmid_imgmem , 0x00 , 0x00);
    global_hand_info->shm_addr_infomem = shmat(shmid_infomem , 0x00 , 0x00);
    if(global_hand_info->shm_addr_imgmem == (void *)-1) return false;
    if(global_hand_info->shm_addr_infomem == (void *)-1) return false;
    return true;
}

void hand_detection::execute_agent(void) {
    GlobalHandDetectionSystem *global_hand_info = GlobalHandDetectionSystem::get_self();
    std::cout << "waiting for the agent to send the data...\n";
    pid_t pid = fork();
    if(pid == 0) {
        execl("/usr/bin/python3" , "/usr/bin/python3" , "./hand_detection_python.py" , "--debug" , "0" , "--video" , global_hand_info->video_device , (char*)0x00);
        exit(0);
    }
    global_hand_info->agent_process_id = pid;
    
    memset(global_hand_info->shm_addr_infomem , 0 , SHM_INFOMEM_SIZE);
    // wait for the data to arrive
    while(((unsigned long *)global_hand_info->shm_addr_infomem)[0] == 0) { }
    std::cout << "global_hand_info->agent_process_id = " << global_hand_info->agent_process_id << "\n";
    
    global_hand_info->screen_width = ((hands_info_t *)global_hand_info->shm_addr_infomem)->width;
    global_hand_info->screen_height = ((hands_info_t *)global_hand_info->shm_addr_infomem)->height;
    global_hand_info->screen_channels = ((hands_info_t *)global_hand_info->shm_addr_infomem)->channels;
    memset(global_hand_info->shm_addr_imgmem , 0 , global_hand_info->screen_width*global_hand_info->screen_height*global_hand_info->screen_channels);
    std::cout << "screen_width = " << global_hand_info->screen_width << "\n";
    std::cout << "screen_height = " << global_hand_info->screen_height << "\n";
    std::cout << "screen_channels = " << global_hand_info->screen_channels << "\n";
}

/// @brief Fetch the hand information from the 
/// @param 
bool hand_detection::fetch_hand_data(Mat &current_frame , hands_info_t &hands) {
    GlobalHandDetectionSystem *global_hand_info = GlobalHandDetectionSystem::get_self();
    if(((unsigned long *)global_hand_info->shm_addr_imgmem)[0] == 0x00) return false;

    current_frame = Mat(Size(global_hand_info->screen_width , global_hand_info->screen_height) , CV_8UC3);
    hands_info_t *hands_info = (hands_info_t *)global_hand_info->shm_addr_infomem;

    memcpy(current_frame.ptr(0) , global_hand_info->shm_addr_imgmem , global_hand_info->screen_width*global_hand_info->screen_height*global_hand_info->screen_channels);
    memcpy(&hands , hands_info , sizeof(hands_info_t));
    return true;
}

void hand_detection::end(void) {
    GlobalHandDetectionSystem *global_hand_info = GlobalHandDetectionSystem::get_self();
    // if(global_hand_info->shm_addr_imgmem) shmdt(global_hand_info->shm_addr_imgmem);
    // if(global_hand_info->shm_addr_infomem) shmdt(global_hand_info->shm_addr_infomem);

    kill(global_hand_info->agent_process_id , SIGKILL);
}

bool hand_detection::check_agent_running(void) {
    return getpgid(GlobalHandDetectionSystem::get_self()->agent_process_id) != -1;
}

static void subroutine_polygon_test(Mat &frame , hands_info_t &hands , 
    std::vector<std::vector<Point>>&relocated_white_contours , std::vector<std::vector<Point>>&relocated_black_contours , 
    std::vector<piano_note_info_t>&on_finger_white_list , std::vector<piano_note_info_t>&on_finger_black_list , 
    PianoInfo &piano_info , bool is_right) {

// only the landmarks of the fingertips
    int finger_number = 1;
    bool hand_on_black_left = false;
    for(int i : std::vector<int>({4 , 8 , 12 , 16 , 20})) {
        int x , y;
        if(is_right) { x = hands.right_hand_landmarks_xlist[i]; y = hands.right_hand_landmarks_ylist[i]; }
        else { x = hands.left_hand_landmarks_xlist[i];  y = hands.left_hand_landmarks_ylist[i]; }
        // detect the black key first
        for(int c = 0; c < relocated_black_contours.size(); c++) {
            if(pointPolygonTest(relocated_black_contours[c] , Point(x , y) , false) >= 0) {
                drawContours(frame , std::vector<std::vector<Point>>({relocated_black_contours[c]}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
                hand_on_black_left = true;
                // left hand
                on_finger_black_list.push_back({piano_info.black_keys_info.key_notes[c].first , is_right , finger_number});
            }
        }
        if(hand_on_black_left == false) { 
            for(int c = 0; c < relocated_white_contours.size(); c++) {
                if(pointPolygonTest(relocated_white_contours[c] , Point(x , y) , false) >= 0) {
                    drawContours(frame , std::vector<std::vector<Point>>({relocated_white_contours[c]}) , -1 , Scalar(0xff , 0x00 , 0x00) , 2);
                    on_finger_white_list.push_back({piano_info.white_keys_info.key_notes[c].first , is_right , finger_number});
                }
            }
        }
        Scalar color;
        if(is_right) color = Scalar(0x00 , 0xff , 0x00);
        else color = Scalar(0xff , 0x00 , 0x00);
        circle(frame , Point(x , y) , 2 , color);
        putText(frame , std::to_string(i) , Point(x , y)  , FONT_HERSHEY_SIMPLEX , 0.5 , color , 2);
        finger_number++;
    }
}

void hand_detection::detect_key_landmark_overlaps(Mat &frame , hands_info_t &hands , 
    std::vector<piano_note_info_t>&on_finger_white_list , std::vector<piano_note_info_t>&on_finger_black_list , 
    PianoInfo &piano_info , std::vector<RotatedRect>&relocated_white_rects , std::vector<RotatedRect>&relocated_black_rects) {

    Scalar white_key_color(0x00 , 0xff , 0x00);
    Scalar black_key_color(0x00 , 0xff , 0xff);
    /****** Print the contours of the rectangles ******/
    std::vector<std::vector<Point>>relocated_white_contours;
    std::vector<std::vector<Point>>relocated_black_contours;
    for(int c = 0; c < relocated_white_rects.size(); c++) {
        std::vector<Point>rc;
        rotated_rect_to_contour(relocated_white_rects[c] , rc);
        relocated_white_contours.push_back(rc);

        drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , white_key_color , 1.5);
        putText(frame , number_to_note_string(piano_info.white_keys_info.key_notes[c].first) , 
            Point(relocated_white_rects[c].center.x , relocated_white_rects[c].center.y+(relocated_white_rects[c].size.height/3)) , 
            FONT_HERSHEY_SIMPLEX , 0.3 , Scalar(0x00 , 0x00 , 0xff) , 1);
    }
    
    for(int c = 0; c < relocated_black_rects.size(); c++) {
        std::vector<Point>rc;
        rotated_rect_to_contour(relocated_black_rects[c] , rc);
        relocated_black_contours.push_back(rc);
        drawContours(frame , std::vector<std::vector<Point>>({rc}) , -1 , black_key_color , 1.5);
    }

    bool hand_on_black_left = false;
    bool hand_on_black_right = false;
    
    if(hands.left_hand_visible)  subroutine_polygon_test(frame , hands , relocated_white_contours , relocated_black_contours , on_finger_white_list , on_finger_black_list , piano_info , false);
    if(hands.right_hand_visible) subroutine_polygon_test(frame , hands , relocated_white_contours , relocated_black_contours , on_finger_white_list , on_finger_black_list , piano_info , true);
}

void hand_detection::compare_with_music_sheet(std::vector<piano_note_info_t>&white_finger_list , std::vector<piano_note_info_t>&black_finger_list , std::vector<piano_note_info_t>&music_sheet , std::vector<std::pair<int , bool>>&correctly_placed_fingers) {
    std::vector<piano_note_info_t>total_finger_list;
    for(piano_note_info_t n : white_finger_list) { total_finger_list.push_back(n); }
    for(piano_note_info_t n : black_finger_list) { total_finger_list.push_back(n); }
    
    for(piano_note_info_t n1 : total_finger_list) {
        for(piano_note_info_t n2 : music_sheet) {
            if(n1.note == n2.note) {
                if(n2.finger_number == 0) {
                    correctly_placed_fingers.push_back(std::pair<int , bool>(n1.finger_number , n1.is_right));
                }
                else if(n1.finger_number == n2.finger_number && n1.is_right == n2.is_right) {
                    correctly_placed_fingers.push_back(std::pair<int , bool>(n1.finger_number , n1.is_right));
                }
            }
        }
    }
}
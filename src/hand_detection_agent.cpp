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
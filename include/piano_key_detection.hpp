/* file piano_key_detection.hpp
 * Suite of functions that detects the bounding rectangles of the piano keys
 */
#ifndef _PIANO_KEY_DETECTION_HPP_
#define _PIANO_KEY_DETECTION_HPP_

#include <piano_essentials.hpp>

namespace piano {
    // black keys are more easy to recognize --> Recognize black keys fist and recognize the white keyma
    void detect_black_keys(cv::Mat piano_image , struct piano_keys_info &keys_info);
    void detect_white_keys(cv::Mat piano_image , struct piano_keys_info &keys_info , const struct piano_keys_info &black_keys_info);
    
    void adjust_key_angles(struct piano_keys_info &keys_info);
    void adjust_key_widths(struct piano_keys_info &keys_info);

    void detect_white_missing_spots(struct piano_keys_info &white_keys_info);
    void create_white_consecutive_key_notes_list(struct piano_keys_info &white_keys_info);
};

#endif
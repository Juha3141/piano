#ifndef _PIANO_NOTE_DETECTION_HPP_
#define _PIANO_NOTE_DETECTION_HPP_

#include <piano_essentials.hpp>

namespace piano {
    
    void create_white_adjusted_cm_list(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);

    void detect_white_key_shapes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);
    void detect_white_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);
    void detect_black_key_notes(struct piano_keys_info &white_keys_info , struct piano_keys_info &black_keys_info);
    
    void detect_missing_black_keys(struct piano_keys_info &black_keys_info , struct piano_keys_info &white_keys_info);

    void adjust_wrong_white_notes(struct piano_keys_info &white_keys_info);
    // void adjust_wrong_black_notes(struct piano_keys_info &black_keys_info);
    
    void fill_missing_white_keys(struct piano_keys_info &white_keys_info);
}

#endif
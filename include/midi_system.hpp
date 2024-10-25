#ifndef _MIDI_SYSTEM_HPP_
#define _MIDI_SYSTEM_HPP_

#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <cstring>

#include <piano_detection.hpp>

typedef struct piano_note_info_s {
    int note;
    bool is_right;
    int finger_number;
}piano_note_info_t;

#define TO_NOTE(note , octave) ((octave << 4)|note)

void construct_piano_file_map(void);
void play_piano_note(int note);

#endif
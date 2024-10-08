#ifndef _PIANO_DEBUG_HPP_
#define _PIANO_DEBUG_HPP_

#include <piano_detection.hpp>

void debug_print(const PianoInfo &piano_info , white_or_black_t white_or_black , const char *win);
void debug_print_colorful(const PianoInfo &piano_info , white_or_black_t white_or_black , const char *win);
void debug_print_both(PianoInfo &piano_info , const char *win);
void debug_print_notes(PianoInfo &piano_info , const char *win);
void debug_print_shapes(PianoInfo &piano_info , const char *win);

#endif
#include <midi_system.hpp>

static std::map<int , std::string>piano_file_map;

void construct_piano_file_map(void) {
    /* A 0 */ piano_file_map[0x0A] = "pianoaudio/448573__tedagame__a0.ogg";
    /* A#0 */ piano_file_map[0x0B] = "pianoaudio/448579__tedagame__a0.ogg";
    /* B 0 */ piano_file_map[0x0C] = "pianoaudio/448565__tedagame__b0.ogg";

    /* C 1 */ piano_file_map[0x11] = "pianoaudio/448544__tedagame__c1.ogg";
    /* C#1 */ piano_file_map[0x12] = "pianoaudio/448544__tedagame__c1.ogg";
    /* D 1 */ piano_file_map[0x13] = "pianoaudio/448606__tedagame__d1.ogg";
    /* D#1 */ piano_file_map[0x14] = "pianoaudio/448542__tedagame__d1.ogg";
    /* E 1 */ piano_file_map[0x15] = "pianoaudio/448616__tedagame__e1.ogg";
    /* F 1 */ piano_file_map[0x16] = "pianoaudio/448581__tedagame__f1.ogg";
    /* F#1 */ piano_file_map[0x17] = "pianoaudio/448586__tedagame__f1.ogg";
    /* G 1 */ piano_file_map[0x18] = "pianoaudio/448557__tedagame__g1.ogg";
    /* G#1 */ piano_file_map[0x19] = "pianoaudio/448591__tedagame__g1.ogg";
    /* A 1 */ piano_file_map[0x1A] = "pianoaudio/448578__tedagame__a1.ogg";
    /* A#1 */ piano_file_map[0x1B] = "pianoaudio/448572__tedagame__a1.ogg";
    /* B 1 */ piano_file_map[0x1C] = "pianoaudio/448564__tedagame__b1.ogg";

    
    /* C 2 */ piano_file_map[0x21] = "pianoaudio/448547__tedagame__c2.ogg";
    /* C#2 */ piano_file_map[0x22] = "pianoaudio/448541__tedagame__c2.ogg";
    /* D 2 */ piano_file_map[0x23] = "pianoaudio/448607__tedagame__d2.ogg";
    /* D#2 */ piano_file_map[0x24] = "pianoaudio/448600__tedagame__d2.ogg";
    /* E 2 */ piano_file_map[0x25] = "pianoaudio/448615__tedagame__e2.ogg";
    /* F 2 */ piano_file_map[0x26] = "pianoaudio/448588__tedagame__f2.ogg";
    /* F#2 */ piano_file_map[0x27] = "pianoaudio/448587__tedagame__f2.ogg";
    /* G 2 */ piano_file_map[0x28] = "pianoaudio/448558__tedagame__g2.ogg";
    /* G#2 */ piano_file_map[0x29] = "pianoaudio/448590__tedagame__g2.ogg";
    /* A 2 */ piano_file_map[0x2A] = "pianoaudio/448563__tedagame__a2.ogg";
    /* A#2 */ piano_file_map[0x2B] = "pianoaudio/448571__tedagame__a2.ogg";
    /* B 2 */ piano_file_map[0x2C] = "pianoaudio/448569__tedagame__b2.ogg";

    
    /* C 3 */ piano_file_map[0x31] = "pianoaudio/448546__tedagame__c3.ogg";
    /* C#3 */ piano_file_map[0x32] = "pianoaudio/448538__tedagame__c3.ogg";
    /* D 3 */ piano_file_map[0x33] = "pianoaudio/448608__tedagame__d3.ogg";
    /* D#3 */ piano_file_map[0x34] = "pianoaudio/448601__tedagame__d3.ogg";
    /* E 3 */ piano_file_map[0x35] = "pianoaudio/448614__tedagame__e3.ogg";
    /* F 3 */ piano_file_map[0x36] = "pianoaudio/448589__tedagame__f3.ogg";
    /* F#3 */ piano_file_map[0x37] = "pianoaudio/448584__tedagame__f3.ogg";
    /* G 3 */ piano_file_map[0x38] = "pianoaudio/448559__tedagame__g3.ogg";
    /* G#3 */ piano_file_map[0x39] = "pianoaudio/448593__tedagame__g3.ogg";
    /* A 3 */ piano_file_map[0x3A] = "pianoaudio/448562__tedagame__a3.ogg";
    /* A#3 */ piano_file_map[0x3B] = "pianoaudio/448570__tedagame__a3.ogg";
    /* B 3 */ piano_file_map[0x3C] = "pianoaudio/448568__tedagame__b3.ogg";

    
    /* C 4 */ piano_file_map[0x41] = "pianoaudio/448549__tedagame__c4.ogg";
    /* C#4 */ piano_file_map[0x42] = "pianoaudio/448539__tedagame__c4.ogg";
    /* D 4 */ piano_file_map[0x43] = "pianoaudio/448609__tedagame__d4.ogg";
    /* D#4 */ piano_file_map[0x44] = "pianoaudio/448602__tedagame__d4.ogg";
    /* E 4 */ piano_file_map[0x45] = "pianoaudio/448613__tedagame__e4.ogg";
    /* F 4 */ piano_file_map[0x46] = "pianoaudio/448595__tedagame__f4.ogg";
    /* F#4 */ piano_file_map[0x47] = "pianoaudio/448585__tedagame__f4.ogg";
    /* G 4 */ piano_file_map[0x48] = "pianoaudio/448552__tedagame__g4.ogg";
    /* G#4 */ piano_file_map[0x49] = "pianoaudio/448592__tedagame__g4.ogg";
    /* A 4 */ piano_file_map[0x4A] = "pianoaudio/448561__tedagame__a4.ogg";
    /* A#4 */ piano_file_map[0x4B] = "pianoaudio/448577__tedagame__a4.ogg";
    /* B 4 */ piano_file_map[0x4C] = "pianoaudio/448536__tedagame__b4.ogg";

    
    /* C 5 */ piano_file_map[0x51] = "pianoaudio/448548__tedagame__c5.ogg";
    /* C#5 */ piano_file_map[0x52] = "pianoaudio/448532__tedagame__c5.ogg";
    /* D 5 */ piano_file_map[0x53] = "pianoaudio/448619__tedagame__d5.ogg";
    /* D#5 */ piano_file_map[0x54] = "pianoaudio/448603__tedagame__d5.ogg";
    /* E 5 */ piano_file_map[0x55] = "pianoaudio/448612__tedagame__e5.ogg";
    /* F 5 */ piano_file_map[0x56] = "pianoaudio/448594__tedagame__f5.ogg";
    /* F#5 */ piano_file_map[0x57] = "pianoaudio/448582__tedagame__f5.ogg";
    /* G 5 */ piano_file_map[0x58] = "pianoaudio/448553__tedagame__g5.ogg";
    /* G#5 */ piano_file_map[0x59] = "pianoaudio/448599__tedagame__g5.ogg";
    /* A 5 */ piano_file_map[0x5A] = "pianoaudio/448560__tedagame__a5.ogg";
    /* A#5 */ piano_file_map[0x5B] = "pianoaudio/448576__tedagame__a5.ogg";
    /* B 5 */ piano_file_map[0x5C] = "pianoaudio/448537__tedagame__b5.ogg";

    
    /* C 6 */ piano_file_map[0x61] = "pianoaudio/448551__tedagame__c6.ogg";
    /* C#6 */ piano_file_map[0x62] = "pianoaudio/448533__tedagame__c6.ogg";
    /* D 6 */ piano_file_map[0x63] = "pianoaudio/448618__tedagame__d6.ogg";
    /* D#6 */ piano_file_map[0x64] = "pianoaudio/448604__tedagame__d6.ogg";
    /* E 6 */ piano_file_map[0x65] = "pianoaudio/448611__tedagame__e6.ogg";
    /* F 6 */ piano_file_map[0x66] = "pianoaudio/448597__tedagame__f6.ogg";
    /* F#6 */ piano_file_map[0x67] = "pianoaudio/448583__tedagame__f6.ogg";
    /* G 6 */ piano_file_map[0x68] = "pianoaudio/448554__tedagame__g6.ogg";
    /* G#6 */ piano_file_map[0x69] = "pianoaudio/448598__tedagame__g6.ogg";
    /* A 6 */ piano_file_map[0x6A] = "pianoaudio/448567__tedagame__a6.ogg";
    /* A#6 */ piano_file_map[0x6B] = "pianoaudio/448575__tedagame__a6.ogg";
    /* B 6 */ piano_file_map[0x6C] = "pianoaudio/448534__tedagame__b6.ogg";

    
    /* C 7 */ piano_file_map[0x71] = "pianoaudio/448550__tedagame__c7.ogg";
    /* C#7 */ piano_file_map[0x72] = "pianoaudio/448545__tedagame__c7.ogg";
    /* D 7 */ piano_file_map[0x73] = "pianoaudio/448617__tedagame__d7.ogg";
    /* D#7 */ piano_file_map[0x74] = "pianoaudio/448605__tedagame__d7.ogg";
    /* E 7 */ piano_file_map[0x75] = "pianoaudio/448610__tedagame__e7.ogg";
    /* F 7 */ piano_file_map[0x76] = "pianoaudio/448596__tedagame__f7.ogg";
    /* F#7 */ piano_file_map[0x77] = "pianoaudio/448580__tedagame__f7.ogg";
    /* G 7 */ piano_file_map[0x78] = "pianoaudio/448555__tedagame__g7.ogg";
    /* G#7 */ piano_file_map[0x79] = "pianoaudio/448556__tedagame__g7.ogg";
    /* A 7 */ piano_file_map[0x7A] = "pianoaudio/448566__tedagame__a7.ogg";
    /* A#7 */ piano_file_map[0x7B] = "pianoaudio/448574__tedagame__a7.ogg";
    /* B 7 */ piano_file_map[0x7C] = "pianoaudio/448535__tedagame__b7.ogg";
    
    
    /* C 8 */ piano_file_map[0x81] = "pianoaudio/448543__tedagame__c8.ogg";
}
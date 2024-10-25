#ifndef _HAND_COMMUNICATION_SYSTEM_HPP_
#define _HAND_COMMUNICATION_SYSTEM_HPP_

#include <iostream>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

typedef struct data_sending_from_center_s {
	bool motor_on;
	int finger_number;
}data_sending_from_center_t;

typedef struct data_received_from_hand_s {
	bool left[5], right[5];
}data_received_from_hand_t;

int open_serial_communication(const char *serial_port);
bool configure_serial_communication(int fd , int speed);

#endif
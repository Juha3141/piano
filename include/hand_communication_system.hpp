#ifndef _HAND_COMMUNICATION_SYSTEM_HPP_
#define _HAND_COMMUNICATION_SYSTEM_HPP_

struct HandInformation {
	bool left_motor_on, right_motor_on;
	int left_finger_number, right_finger_number;
};

struct CentralInformation {
	bool left[5], right[5];
};

#endif
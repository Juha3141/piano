#include <hand_communication_system.hpp>

int open_serial_communication(const char *serial_port) {
    int fd = open(serial_port , O_RDWR|O_NOCTTY|O_SYNC);
    if(fd < 0) {
        std::cout << "Error opening port " << serial_port << " : " << strerror(errno);
        return -1;
    }
    return fd;
}

bool configure_serial_communication(int fd , int speed) {
    struct termios tty;
    if(tcgetattr(fd , &tty) != 0) {
        return false;
    }
    cfsetospeed(&tty , speed);
    cfsetispeed(&tty , speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE)|CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    
    tty.c_oflag = 0;                    // no remapping
    tty.c_cc[VMIN] = 0;                 // 
    tty.c_cc[VTIME] = 5;                // 0.5 seconds read timeout
    tty.c_iflag &= ~(IXON|IXOFF|IXANY); // shut off xon/xoff ctl
    tty.c_cflag |= (CLOCAL|CREAD);      // ignore modem controls
    
    tty.c_cflag &= ~(PARENB|PARODD); // no parity bit
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    if(tcsetattr(fd , TCSANOW , &tty) != 0) {
        return false;
    }
    return true;
} 

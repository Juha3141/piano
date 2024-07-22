SOURCESFOLDER = src
HEADERSFOLDER = include
BINARYFOLDER  = bin

TARGET = piano

CC = g++
CCOPTIONS    = -m64 
LIBS         = $(shell pkg-config opencv4 --libs)
INCLUDEPATHS = -I $(HEADERSFOLDER) -I /usr/local/include/opencv4 $(shell pkg-config opencv4 --cflags)

clean: 
	rm -rf $(TARGET)
	rm -rf $(BINARYFOLDER)/*.*

all: $(subst .cpp,.obj,$(wildcard $(SOURCESFOLDER)/*.cpp))
	$(CC) $(wildcard $(BINARYFOLDER)/*.obj) -o $(TARGET) $(LIBS)

run:
#	gnome-terminal -- ./$(TARGET)
	./$(TARGET)

run_vid_init:
	v4l2-ctl -d /dev/video0
	./$(TARGET)

%.obj: %.cpp
	$(CC) $(CCOPTIONS) -c $< -o $(subst $(SOURCESFOLDER),$(BINARYFOLDER),$@) $(INCLUDEPATHS)

.PHONY: all
# --- macros
CC=/usr/local/openmpi-3.0.1/bin/mpic++
CFLAGS= -std=c++11 -lm -O3 -I /usr/local/opencv-3.4.1/include
LIBS = -L /usr/local/opencv-3.4.1/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

OBJFILES = filter.o haar_filter.o image.o main.o
TARGET = train

# --- targets
all: $(TARGET)

$(TARGET): $(OBJFILES) 
	$(CC) -o $(TARGET) $(OBJFILES) $(LIBS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<
       
# --- remove binary and executable files
clean:
	rm -f $(OBJFILES) $(TARGET) *~


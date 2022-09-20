CC = g++
CFLAGS = -c -O2 -Wall -std=c++17 -DHAVE_CSTDDEF  -g
LDFLAGS = 
POSTLIBS = -pthread 
SOURCES = $(shell find ./ -name '*.cpp')
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = main

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(COININC)  

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(COININC)  

clean:
	rm -rf *.o
	rm -rf main

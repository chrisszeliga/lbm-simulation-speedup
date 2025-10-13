# Define the C++ compiler to use
CXX = mpicxx

# Define any compile-time flags
CXXFLAGS = -std=c++14 -Wall -Wextra -pedantic -Wno-unused-parameter -fopenmp -Ofast

# Define the C++ source files
SOURCES = lbmSimulation.cc

# Define the C++ object files
OBJECTS = $(SOURCES:.cc=.o)

# Define the executable file 
MAIN = lbmSimulation

.PHONY: all clean

all: $(MAIN)
	@echo Congratulations $(MAIN) has been compiled!

$(MAIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(MAIN) $(OBJECTS)

.cc.o:
	$(CXX) $(CXXFLAGS) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)
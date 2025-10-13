# Compiler
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -fopenmp -Ofast

# Find all .cc files and create a list of executables
SOURCES := $(wildcard *.cc)
EXECUTABLES := $(SOURCES:.cc=)

# Default target: build all executables
all: $(EXECUTABLES)

# Rule to compile each .cc file into an executable
%: %.cc
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean rule to remove all executables
clean:
	rm -f $(EXECUTABLES)

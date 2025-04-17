MPICXX = mpicxx
CXXFLAGS = -Wall -std=c++11 -O3# Compiler flags
SRCS = GaussMain.cpp# Source files
OBJS = $(SRCS:.cpp=.o)# Object files
TARGET = mpi_program# Executable name
# Default target
all: $(TARGET) run
# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
  
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@
# Clean up generated files

run: $(TARGET)
	mpirun -np 4 $(TARGET) 64 1

clean:
	rm -f $(OBJS) $(TARGET)
# Phony targets
.PHONY: all clean run


# TODO: Generate proper header dependencies
# TODO: automake?


#======================================================================
# Project sources
HEADERS := $(wildcard *.h)
SOURCES := $(wildcard *.cpp *.cu)
OBJECTS := $(SOURCES:%.cpp=%.o)
OBJECTS := $(OBJECTS:%.cu=%.o)

#======================================================================
# CUDA configuration
NVCC = nvcc
CUDA_LIB_FLAGS = -L/usr/local/cuda-6.5/lib64 -lcudart


#======================================================================
# wx configuration
WX_CXX_FLAGS = $(shell wx-config --cppflags)
WX_LIB_FLAGS = $(shell wx-config --libs --gl-libs)

# For proper thread operation, need to call XInitThreads
WX_LIB_FLAGS += -lX11

#======================================================================
# OpenGL configuration
GL_LIB_FLAGS = -L/usr/lib -lGLU -lGL

#======================================================================
# OpenCV configuration
CV_LIB_FLAGS = $(shell pkg-config --libs opencv)

#======================================================================
# Boost configuration
BOOST_LIB_FLAGS = -lboost_thread -lboost_system

#======================================================================
# All flags and targets
CXX_FLAGS += $(WX_CXX_FLAGS)
LIB_FLAGS += $(WX_LIB_FLAGS)
LIB_FLAGS += $(GL_LIB_FLAGS)
LIB_FLAGS += $(CV_LIB_FLAGS)
LIB_FLAGS += $(BOOST_LIB_FLAGS)
LIB_FLAGS += $(CUDA_LIB_FLAGS)

default: run

run: cuda-video-gui
	(cd ../bin_data/; ../src/cuda-video-gui )

cuda-video-gui: $(HEADERS) $(OBJECTS)
	$(CXX) $(OBJECTS) $(LIB_FLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CXX_FLAGS) -c -o $@ $<



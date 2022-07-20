CXX=g++
ifeq ($(DEBUG), 1)
OFLAGS=-g
else
OFLAGS=-O3 -fopenmp
endif

THIRDPARTY_DIR=/absolute/path/to/third_parties/folder
CUDA_ROOT=/path/to/cuda/root  # e.g., /usr/local/cuda-11.1

CXXFLAGS=-DWHICH_FUNC=1 -pthread -std=c++14 -DCGAL_EIGEN3_ENABLED -DWHICH_FUNC=1 \
		 -I. -I./mesh/include -I./project/include -I./linear-algebra/include -I./deps -I./deps/rectangle-bin-pack/include \
		 -I/usr/include \
		 -I${THIRDPARTY_DIR}/third_parties/eigen-3.4 \
		 -I/usr/include/suitesparse \
		 -I${THIRDPARTY_DIR}/third_parties/boost_1_75_0/build/include \
		 -I${THIRDPARTY_DIR}/third_parties/CGAL-5.1.5/install/include
LDPATH=-L${CUDA_ROOT}/lib64
LDFLAGS=-lpng -lmetis -lcholmod -lm -pthread -lgomp -lblas -lcudart -lcublas  -lumfpack -lamd -lsuitesparseconfig

SRC_FILES := $(wildcard ./*.cpp)
SRC_FILES+=$(wildcard mesh/src/*.cpp)
SRC_FILES+=$(wildcard project/src/*.cpp)
SRC_FILES+=$(wildcard linear-algebra/src/*.cpp)
SRC_FILES+=$(wildcard deps/rectangle-bin-pack/src/*.cpp)

H_FILES := $(wildcard ./*.h)
H_FILES+=$(wildcard mesh/src/*.h)
H_FILES+=$(wildcard project/src/*.h)
H_FILES+=$(wildcard linear-algebra/src/*.h)
H_FILES+=$(wildcard deps/nlohmann/*.hpp)
H_FILES+=$(wildcard deps/rectangle-bin-pack/src/*.h)

VPATH := $(sort $(dir $(SRC_FILES)))

SRC_FILES_NOPATH = $(filter %.cpp,$(subst /, ,$(SRC_FILES)))

OBJ_FILES_NOPATH := $(patsubst %.cpp,%.o,$(SRC_FILES_NOPATH))

OBJ_FILES_PATH=$(OBJ_FILES_NOPATH:%.o=obj/%.o)

.PHONY: all
all: ; $(info $$OBJ_FILES_PATH is [${OBJ_FILES_PATH}])echo Hello world

tex_init: $(OBJ_FILES_PATH)
	$(CXX) $(LDPATH) -o tex_init $(OBJ_FILES_PATH) $(LDFLAGS)

obj/%.o: %.cpp $(H_FILES) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OFLAGS) -flax-vector-conversions -c $< -o $@

clean:
	rm -f obj/*
	rm -f tex_init

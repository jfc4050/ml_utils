OSX=0
PYTHON_VERSION=3.7
SRC=ml_utils
PKG=ml_utils

CPP=g++-7
CPP_FLAGS= -std=c++14 -O3 -shared -fPIC -w -Wfatal-errors
LINKER_FLAGS= -Wl,-undefined=dynamic_lookup

ifeq ($(OSX), 1)
CPP_FLAGS+= -mmacosx-version-min=10.7
endif

INCLUDES=$(addprefix -I${CONDA_PREFIX},\
	/include\
	/include/python$(PYTHON_VERSION)m\
	/lib/python$(PYTHON_VERSION)/site-packages/numpy/core/include\
)

all: $(addprefix $(PKG)/, bbox_utils.so prediction_filtering.so)

$(PKG)/bbox_utils.so: $(SRC)/bbox_utils.cpp
	$(CPP) $(CPP_FLAGS) $(INCLUDES) $< -o $@

$(PKG)/prediction_filtering.so: $(SRC)/prediction_filtering.cpp $(PKG)/bbox_utils.so
	$(CPP) $(CPP_FLAGS) $(INCLUDES) $^ -o $@
cmake_minimum_required(VERSION 2.8.12)
project(ml_utils)
set(SOURCE_DIR "ml_utils")
include_directories(${SOURCE_DIR})
set(SOURCES "${SOURCE_DIR}/boundingboxes.cpp")


include_directories(lib/pybind11/include)
include_directories(lib/xtl/include)
include_directories(lib/xtensor/include)
include_directories(lib/xtensor-python/include)

add_subdirectory(lib/pybind11)
pybind11_add_module(boundingboxes ${SOURCES} "${SOURCE_DIR}/boundingboxes.cpp")

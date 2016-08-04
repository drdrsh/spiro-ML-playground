cmake_minimum_required(VERSION 3.5)

project(ImageTools)

configure_file( ${CMAKE_SOURCE_DIR}/converter_runner.py  ${CMAKE_BINARY_DIR}/converter_runner.py COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/trimmer_runner.py  ${CMAKE_BINARY_DIR}/trimmer_runner.py COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/dicom_to_numpy.py  ${CMAKE_BINARY_DIR}/dicom_to_numpy.py COPYONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(ImageConverter main.cpp)
target_link_libraries(ImageConverter ${ITK_LIBRARIES})

add_executable(LungSegment LungSegment.cpp)
target_link_libraries(LungSegment ${ITK_LIBRARIES})



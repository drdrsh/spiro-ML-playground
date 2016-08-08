cmake_minimum_required(VERSION 3.5)

project(ImageTools)

configure_file( ${CMAKE_SOURCE_DIR}/converter_runner.py  ${CMAKE_BINARY_DIR}/converter_runner.py COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/trimmer_runner.py    ${CMAKE_BINARY_DIR}/trimmer_runner.py   COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/dicom_to_numpy.py    ${CMAKE_BINARY_DIR}/dicom_to_numpy.py   COPYONLY)

set(Boost_INCLUDE_DIR C:/Libraries/boost_1_61_0/)
set(BOOST_LIBRARYDIR C:/Libraries/boost_1_61_0/stage/libs/x64)
SET(Boost_USE_STATIC_LIBS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

FIND_PACKAGE(Boost 1.61.0 COMPONENTS filesystem program_options system REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
LINK_DIRECTORIES(${LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS})

add_executable(ImageMultiRes ImageMultiRes.cpp)
target_link_libraries(ImageMultiRes ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(LungSegment LungSegment.cpp)
target_link_libraries(LungSegment ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(DICOMGrid DICOMGrid.cpp)
target_link_libraries(DICOMGrid ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(ImageDownsample ImageDownsample.cpp)
target_link_libraries(ImageDownsample ${ITK_LIBRARIES} ${Boost_LIBRARIES})

add_executable(ImageAugment ImageAugment.cpp)
target_link_libraries(ImageAugment ${ITK_LIBRARIES} ${Boost_LIBRARIES})

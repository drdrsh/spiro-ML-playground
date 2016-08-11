cmake_minimum_required(VERSION 3.5)

project(ImageTools)

configure_file( ${CMAKE_SOURCE_DIR}/convert.py  ${CMAKE_BINARY_DIR}/convert.py  COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/segment.py  ${CMAKE_BINARY_DIR}/segment.py  COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/numpyify.py ${CMAKE_BINARY_DIR}/numpyify.py COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/augment.py  ${CMAKE_BINARY_DIR}/augment.py  COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/testset.py  ${CMAKE_BINARY_DIR}/testset.py  COPYONLY)

install(
    FILES convert.py segment.py numpyify.py testset.py augment.py
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION ${CMAKE_BINARY_DIR}
)

file(COPY ${CMAKE_SOURCE_DIR}/APPIL_DNN DESTINATION ${CMAKE_BINARY_DIR})

#set(Boost_INCLUDE_DIR C:/Libraries/boost_1_61_0/)
#set(BOOST_LIBRARYDIR C:/Libraries/boost_1_61_0/stage/libs/x64)
SET(Boost_USE_STATIC_LIBS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

FIND_PACKAGE(Boost 1.58.0 COMPONENTS filesystem program_options system REQUIRED )
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

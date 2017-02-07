cmake_minimum_required(VERSION 3.1)
project(First_project)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/LightLDAWrapper.cpp src/LightLDAWrapper.h)
add_executable(First_project ${SOURCE_FILES})
find_package(Boost 1.58	COMPONENTS
        filesystem
        REQUIRED)
set(CMAKE_VERBOSE_MAKEFILE ON)
link_directories(${Boost_LIBRARIES})
target_link_libraries (First_project ${Boost_LIBRARIES})


#cmake_minimum_required(VERSION 3.1) #versian, karevora
#project (First_project5) #anuny
#set(CMAKE_CXX_STANDARD 11) # ogtagorcel c++11
#find_package(Boost 1.58	COMPONENTS
#        filesystem
#        REQUIRED)
#
#
#set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}) # vortex sarqi executable
##mongo db library
#include_directories(/usr/include/boost) #inchic dependance-i unes vortexic includ ani
#link_directories(${Boost_LIBRARIES}) # nuyny linkagi hamar
##project header
#include_directories( ./include) # ete failumt include unes et includnery vortexic include ani
#add_executable(First_project
#        src/main.cpp src/LightLDAWrapper.cpp) # executable sarqi. 1.cragrit anuny, 2.vor fily build ani
#target_link_libraries (First_project ${Boost_LIBRARIES}) # konkret voronq link ani

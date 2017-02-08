cmake_minimum_required(VERSION 3.1)
project(First_project)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES_CREATEMODEL src/LightLDA_createmodel/main.cpp src/LightLDA_createmodel/LightLDAWrapper.cpp src/LightLDA_createmodel/LightLDAWrapper.h)
add_executable(create_model ${SOURCE_FILES_CREATEMODEL})
find_package(Boost 1.58	COMPONENTS
        filesystem
        REQUIRED)
set(CMAKE_VERBOSE_MAKEFILE ON)
link_directories(${Boost_LIBRARIES})
target_link_libraries(create_model ${Boost_LIBRARIES})


add_executable(get_output src/LightLDA_getoutput/main.cpp src/LightLDA_getoutput/LightLDA_getoutput.cpp src/LightLDA_getoutput/LightLDA_getoutput.h)
find_package(Boost 1.58 COMPONENTS
        filesystem
        REQUIRED)
set(CMAKE_VERBOSE_MAKEFILE ON)
link_directories(${Boost_LIBRARIES})
target_link_libraries(get_output ${Boost_LIBRARIES})


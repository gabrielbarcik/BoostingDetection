cmake_minimum_required(VERSION 3.5)
project(Project)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(OpenCV REQUIRED)

file(GLOB SOURCES "src/*.cpp")

add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})

